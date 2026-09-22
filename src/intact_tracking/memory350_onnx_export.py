"""Full frozen-tracker + Memory350 encoder + residual deployment graph."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from intact_tracking import memory350_deploy
from intact_tracking.memory350_checkpoint import (
    dependencies_from_legacy_checkpoint, embedded_tracker, load_policy_context,
)
from intact_tracking.memory350_deploy import (
    INPUT_FIELDS, INPUT_NAME, INPUT_WIDTH, OUTPUT_NAMES, TRACKER_FIELDS, TRACKER_WIDTH, input_layout,
)
from intact_tracking.memory350_tracker_action_policy import VERSION, TrackerActionResidualActor
from intact_tracking.rollout.mjlab_adapter import _sha256


@contextmanager
def exportable_attention():
    # Avoid fused aten::_transformer_encoder_layer_fwd in the legacy exporter.
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def split_input(value):
    offset, result = 0, {}
    for name, shape in INPUT_FIELDS:
        width = int(np.prod(shape))
        result[name] = value[:, offset:offset + width].reshape(value.shape[0], *shape)
        offset += width
    return result


def tracker_observations(value):
    offset, data = 0, {}
    for name, width in TRACKER_FIELDS:
        data[name] = value[:, offset:offset + width]
        offset += width
    return TensorDict(data, [value.shape[0]])


class Memory350ResidualExport(torch.nn.Module):
    input_size = INPUT_WIDTH
    deploy_input_names = [INPUT_NAME]

    def __init__(self, actor, context):
        super().__init__()
        if (actor.tracker.obs_dim != TRACKER_WIDTH or actor.dynamics_latent_dim != 320
                or context.state_mean.numel() != 122 or context.action_mean.numel() != 29):
            raise ValueError("Export requires proprio122/SPV5-2/history5 tracker-action architecture")
        self.preprocessing = actor.tracker.as_onnx()
        self.preprocessing.mlp = torch.nn.Identity()
        self.preprocessing.deterministic_output = torch.nn.Identity()
        self.base_mlp = copy.deepcopy(actor.tracker.mlp)
        self.base_output = actor.tracker.distribution.as_deterministic_output_module()
        self.encoder = copy.deepcopy(context.encoder)
        self.residual_mlp = copy.deepcopy(actor.residual_mlp)
        self.residual_output_mode, self.residual_scale = actor.residual_output_mode, actor.residual_scale
        self.latent_input_mode = actor.latent_input_mode
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(name, getattr(context, name).detach().cpu().clone())

    def _normalize(self, value):
        return torch.cat(((value[..., :122] - self.state_mean) / self.state_std,
                          (value[..., 122:151] - self.action_mean) / self.action_std,
                          (value[..., 151:] - self.state_mean) / self.state_std), dim=-1)

    def forward(self, observation):
        fields = split_input(observation)
        short = self._normalize(fields["short_interactions"])
        long = self._normalize(fields["long_interactions"])
        latent = self.encoder(short[..., :122], short[..., 122:151], short[..., 151:],
                              fields["short_valid"] > .5, long, fields["long_valid"] > .5)
        history = torch.cat((fields["previous_latents"].flatten(1), latent), dim=-1)
        if self.latent_input_mode == "zero":
            history = torch.zeros_like(history)
            latent = torch.zeros_like(latent)
        features = self.preprocessing(fields["tracker_observation"])
        base = self.base_output(self.base_mlp(features))
        residual = self.residual_mlp(torch.cat((features, history, base), dim=-1))
        if self.residual_output_mode == "bounded":
            residual = self.residual_scale * residual.tanh()
        return base + residual, latent


def load_export_models(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["residual_policy"]["version"] != VERSION:
        raise ValueError("Only the proprio122/history5/tracker-action policy is supported")
    if "inference_bundle_version" not in state:
        state.update(dependencies_from_legacy_checkpoint(state))
    tracker = embedded_tracker(state)
    context = load_policy_context(state, device="cpu")
    agent = OmegaConf.to_container(state["cfg"].agent, resolve=True)
    if agent.get("clip_actions") is not None:
        raise ValueError("This export contract requires an unclipped policy action chain")
    kwargs = copy.deepcopy(agent["actor"])
    if kwargs.pop("class_name") != "intact_tracking.memory350_tracker_action_policy:TrackerActionResidualActor":
        raise ValueError("Unsupported actor class")
    value = torch.zeros(1, TRACKER_WIDTH)
    value[:, 0] = 1
    obs = tracker_observations(value)
    obs.set("dynamics_latent", torch.zeros(1, 320))
    # The tracker constructor checks this training target's width. It is never
    # read by inference and is not an exported input.
    from intact_tracking.environment.mdp.spv5 import SPV5_REFERENCE_TARGET_DIM
    target_group = kwargs["tracker_actor_kwargs"].get("reference_encoder_target_group", "reference_encoder_target")
    obs.set(target_group, torch.zeros(1, SPV5_REFERENCE_TARGET_DIM))
    obs.set(kwargs["tracker_actor_kwargs"].get("estimator_target_group", "estimator_target"), torch.zeros(1, 1))
    obs.set(kwargs["tracker_actor_kwargs"].get("foot_contact_target_group", "foot_contact_target"), torch.zeros(1, 2))
    actor = TrackerActionResidualActor(obs, agent["obs_groups"], "actor", 29,
                                      tracker_state_dict=tracker["actor_state_dict"], **kwargs)
    actor.load_state_dict(state["actor_state_dict"], strict=True)
    actor.eval().requires_grad_(False)
    module = Memory350ResidualExport(actor, context).cpu().eval().requires_grad_(False)
    return state, actor, context, module


@torch.inference_mode()
def reference_outputs(value, actor, context):
    fields = split_input(value)
    def normalize(raw):
        return torch.cat(((raw[..., :122] - context.state_mean) / context.state_std,
                          (raw[..., 122:151] - context.action_mean) / context.action_std,
                          (raw[..., 151:] - context.state_mean) / context.state_std), -1)
    short, long = normalize(fields["short_interactions"]), normalize(fields["long_interactions"])
    latent = context.encoder(short[..., :122], short[..., 122:151], short[..., 151:],
                             fields["short_valid"] > .5, long, fields["long_valid"] > .5)
    if actor.latent_input_mode == "zero":
        latent = torch.zeros_like(latent)
    obs = tracker_observations(fields["tracker_observation"])
    obs.set("dynamics_latent", torch.cat((fields["previous_latents"].flatten(1), latent), -1))
    return actor(obs), latent


def audit_inputs(seed=20260920):
    generator = torch.Generator().manual_seed(seed)
    cases = []
    for short_count, long_count in ((0, 0), (1, 0), (50, 0), (50, 1), (50, 30), (0, 30), (17, 12)):
        value = torch.randn(1, INPUT_WIDTH, generator=generator) * .1
        fields = split_input(value)
        quat = fields["tracker_observation"][:, :4]
        quat.div_(quat.norm(dim=-1, keepdim=True))
        fields["short_valid"].zero_()
        fields["long_valid"].zero_()
        if short_count:
            fields["short_valid"][:, -short_count:] = 1
        if long_count:
            fields["long_valid"][:, -long_count:] = 1
        if short_count == 0:
            fields["previous_latents"].zero_()
        cases.append((f"short{short_count}_long{long_count}", value))
    return cases


def export_policy(checkpoint, output, *, run_name=None):
    """Publish ONNX+JSON only after checker and CPU runtime parity succeed."""
    import onnx
    import onnxruntime as ort

    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh versioned export directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    state, actor, context, module = load_export_models(checkpoint)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        cases = audit_inputs()
        path = temporary / "policy.onnx"
        with exportable_attention(), torch.inference_mode():
            module(cases[0][1])  # Build the constant robot-kinematics cache before tracing.
            torch.onnx.export(module, (cases[0][1],), str(path), export_params=True,
                              opset_version=18, input_names=[INPUT_NAME],
                              output_names=list(OUTPUT_NAMES), dynamic_axes={}, dynamo=False)
        onnx.checker.check_model(str(path), full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        audits = []
        with torch.inference_mode():
            for name, value in cases:
                expected = reference_outputs(value, actor, context)
                actual_torch = module(value)
                start = time.perf_counter()
                actual_onnx = session.run(list(OUTPUT_NAMES), {INPUT_NAME: value.numpy()})
                elapsed = time.perf_counter() - start
                errors = {}
                for key, want, got_torch, got_onnx in zip(OUTPUT_NAMES, expected, actual_torch, actual_onnx, strict=True):
                    torch.testing.assert_close(got_torch, want, atol=2e-4, rtol=2e-4)
                    np.testing.assert_allclose(got_onnx, want.numpy(), atol=2e-4, rtol=2e-4)
                    errors[key] = float(np.max(np.abs(got_onnx - want.numpy())))
                audits.append({"case": name, "max_absolute_errors": errors, "cpu_seconds": elapsed})
        metadata = build_metadata(state, checkpoint, module, run_name=run_name)
        metadata.update(onnx_sha256=_sha256(path), validation={
            "passed": True, "atol": 2e-4, "rtol": 2e-4, "cases": audits,
            "provider": "CPUExecutionProvider", "cpu_threads": 1,
            "torch_version": str(torch.__version__), "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
        })
        (temporary / "policy.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (temporary / "deploy_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copyfile(memory350_deploy.__file__, temporary / "policy_runtime.py")
        temporary.rename(output)
        return {"directory": str(output), "checkpoint": str(checkpoint),
                "completed_updates": state["completed_updates"], "validation": metadata["validation"]}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_metadata(state, checkpoint, module, *, run_name=None):
    meta = state["residual_policy"]
    result = {
        "format": "motion_tracking_sim2real_policy",
        "deployment_contract": "memory350_proprio122_history5_onnx_v1",
        "run_name": run_name or meta.get("arguments", {}).get("wandb_name") or checkpoint.parent.name,
        "run_path": str(checkpoint.parent), "iteration": int(state["iter"]),
        "completed_updates": int(state["completed_updates"]), "checkpoint": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint), "in_keys": [INPUT_NAME],
        "out_keys": list(OUTPUT_NAMES), "in_shapes": [[[1, INPUT_WIDTH]]],
        "out_shapes": [[[1, 29]], [[1, 64]]], "num_actions": 29, "opset": 18,
        "dtype": "float32", "input_layout": input_layout(),
        "precision_contract": "FP32 ONNX; standard unfused attention. CUDA BF16/fused training inference is not bit-identical.",
        "tracker_observation_layout": [{"name": n, "width": w} for n, w in TRACKER_FIELDS],
        "tracker_sha256": meta["tracker_sha256"], "context_sha256": meta["context_sha256"],
        "context_normalization": "embedded in ONNX; supply raw observations and raw interactions",
        "latent_input_mode": module.latent_input_mode,
        "context_input_contract": state["frozen_context"]["payload"]["context_input_contract"],
        "memory": {"short_steps": 50, "chunk_steps": 10, "long_chunks": 30,
                   "layout": "oldest to newest; left padded; float32 masks 0/1",
                   "interaction": ["proprio122_before", "command29", "proprio122_after"],
                   "long_chunks_disjoint_from_short": True,
                   "pending_eviction_tail_max_steps": 9,
                   "boundary": "exclude crossing transition; flush completed old chunks; discard incomplete tail",
                   "episode_or_motion_reset": "retain complete chunks; clear short and latent history",
                   "physical_session_reset": "clear all memory and latent history",
                   "latent_history": "previous four latent64 frames, oldest to newest; current frame computed inside ONNX"},
        "action_semantics": {"output": "deterministic frozen tracker mean + residual mean",
                             "residual_output_mode": module.residual_output_mode,
                             "residual_scale": module.residual_scale,
                             "pd_target": "default_joint_pos + action_scale * action",
                             "clipping": None, "auxiliary_dr_head": "training only; not needed for action inference"},
        "standalone_runtime": "policy_runtime.py; requires numpy and onnxruntime",
        "stock_tracker_client_compatible": False,
    }
    result.update(robot_metadata(state))
    return result


def robot_metadata(state):
    # Build only the CPU robot specification, never a rollout or motion dataset.
    import re
    from intact_tracking.environment.config import _build_action, _build_robot

    task = OmegaConf.create(state["frozen_tracker"]["cfg"]["task"])
    config = _build_robot(task)
    robot = config.build()
    model = robot.spec.compile()
    joints = list(robot.joint_names)
    action = _build_action(task)["joint_pos"]
    if (not action.use_default_offset or action.clip is not None
            or getattr(action, "raw_action_clip", None) is not None):
        raise ValueError("Deployment metadata requires default joint offsets and an unclipped action chain")
    actuator_map = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    indices = [actuator_map[n] for n in joints]

    def resolve(value, default=0.):
        if not isinstance(value, dict):
            return [float(value)] * len(joints)
        result = []
        for name in joints:
            matches = [float(v) for pattern, v in value.items() if re.fullmatch(pattern, name)]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous joint metadata for {name}")
            result.append(matches[0] if matches else default)
        return result

    command = task.command.get("command", task.command)
    return {"joint_names": joints, "joint_stiffness": model.actuator_gainprm[indices, 0].tolist(),
            "joint_damping": (-model.actuator_biasprm[indices, 2]).tolist(),
            "default_joint_pos": resolve(config.init_state.joint_pos),
            "action_scale": resolve(action.scale, default=1.), "command_names": ["motion"],
            "observation_names": [n for n, _ in TRACKER_FIELDS],
            "anchor_body_name": str(command.anchor_body_name), "body_names": list(command.body_names),
            "control_dt": float(task.decimation) * float(task.sim.timestep),
            "pd_parameters": "compiled nominal model; no training DR applied"}
