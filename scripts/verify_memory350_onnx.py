"""Compare the standalone ONNX client to the real wrapper during live simulation."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from intact_tracking.cli.memory350_proprio_native_policy_eval import configure
from intact_tracking.cli import memory350_policy_eval as evaluation
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.memory350_checkpoint import embedded_tracker, load_policy_context
from intact_tracking.memory350_deploy import Memory350Policy, TRACKER_FIELDS
from intact_tracking.memory350_onnx_export import load_export_models, reference_outputs, tracker_observations
from intact_tracking.memory350_policy_precision import configure_policy_precision
from intact_tracking.memory350_proprio_policy import ProprioNativePolicyWrapper
from intact_tracking.rollout.mjlab_adapter import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--motion-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=420)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    configure_policy_precision("fp32")
    # CUDA's fused Transformer inference path differs measurably from the
    # standard FP32 graph even without autocast. ONNX exports the unfused graph.
    torch.backends.mha.set_fastpath_enabled(False)
    configure()
    client = Memory350Policy(args.export_dir)
    metadata = client.metadata
    if metadata["checkpoint_sha256"] != _sha256(args.checkpoint):
        raise ValueError("Export/checkpoint mismatch")
    state, cpu_actor, cpu_context, cpu_module = load_export_models(args.checkpoint)
    actor = copy.deepcopy(cpu_actor)
    tracker = embedded_tracker(state)
    prepared = prepare_rollout(checkpoint_file="/unused/embedded_tracker.pt", num_envs=4,
                               motion_file=None, motion_path=evaluation.FULL_DATASET,
                               checkpoint_config=tracker["cfg"])
    evaluation.configure_limb_dr(prepared.env, 90260920, profile=state["residual_policy"]["dr_profile"])
    command = prepared.env.commands["motion"]
    command.motion_manifest_file = str(args.motion_manifest.resolve())
    command.sampling_mode = "uniform"
    prepared.env.episode_length_s = 2.4  # Exercise resets and old-chunk retention repeatedly.
    prepared.env.auto_reset, prepared.env.seed = True, 90260920
    env = evaluation.ManagerBasedRlEnv(cfg=prepared.env, device="cuda:0")
    records = {key: [] for key in ("observations", "reset_boundary", "commands", "expected_action", "expected_latent")}
    max_action, max_latent, resets = 0., 0., 0
    try:
        context = load_policy_context(state, device="cuda:0")
        wrapped = ProprioNativePolicyWrapper(env, prepared.clip_actions, context, latent_history_frames=5)
        wrapped.context.use_bfloat16 = False
        wrapped.latent_history.clear()
        wrapped._latent = wrapped._record_latent(wrapped._encode())
        actor.to("cuda:0").eval().requires_grad_(False)
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        assert list(robot.joint_names) == metadata["joint_names"]
        np.testing.assert_allclose(robot.data.default_joint_pos[0].cpu(), metadata["default_joint_pos"], atol=1e-7)
        np.testing.assert_allclose(action_term._scale[0].cpu(), metadata["action_scale"], atol=1e-7)
        actuator_ids = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
        indices = [actuator_ids[name] for name in robot.joint_names]
        np.testing.assert_allclose(env.sim.mj_model.actuator_gainprm[indices, 0], metadata["joint_stiffness"])
        np.testing.assert_allclose(-env.sim.mj_model.actuator_biasprm[indices, 2], metadata["joint_damping"])
        obs, boundary = wrapped.get_observations(), False
        with torch.inference_mode():
            for step in range(args.steps):
                value = torch.cat([obs[name][0] for name, _ in TRACKER_FIELDS]).cpu().numpy()
                expected = actor(obs)
                previous_latents = client.previous_latents.copy()
                if boundary:
                    previous_latents.fill(0)
                actual = client.step(value, reset_boundary=boundary)
                wanted_action = expected[0].cpu().numpy()
                wanted_latent = obs["dynamics_latent"][0, -64:].cpu().numpy()
                try:
                    np.testing.assert_allclose(actual, wanted_action, atol=2e-4, rtol=2e-4)
                except AssertionError:
                    packed = np.concatenate((value, *(v.reshape(-1) for v in client.history.tensors()),
                                             previous_latents.reshape(-1)))[None]
                    tensor = torch.from_numpy(packed)
                    cpu_export_action, cpu_export_latent = cpu_module(tensor)
                    cpu_reference_action, cpu_reference_latent = reference_outputs(tensor, cpu_actor, cpu_context)
                    cpu_obs = tracker_observations(torch.from_numpy(value[None]))
                    gpu_history = obs["dynamics_latent"][:1].cpu()
                    cpu_obs.set("dynamics_latent", gpu_history)
                    cpu_actor_action = cpu_actor(cpu_obs)[0].numpy()
                    debug = {"step": step, "boundary": boundary,
                             "onnx_vs_cpu_export": float(np.max(np.abs(actual-cpu_export_action[0].numpy()))),
                             "cpu_export_vs_cpu_actor": float((cpu_export_action-cpu_reference_action).abs().max()),
                             "cpu_vs_gpu_actor_same_latent": float(np.max(np.abs(cpu_actor_action-wanted_action))),
                             "cpu_vs_gpu_latent": float(np.max(np.abs(cpu_reference_latent[0].numpy()-wanted_latent))),
                             "previous_latent_history_error": float(np.max(np.abs(previous_latents.reshape(-1)-gpu_history[0,:256].numpy())))}
                    (args.output_dir/'failure_diagnostics.json').write_text(json.dumps(debug,indent=2)+'\n')
                    np.savez_compressed(args.output_dir/'failure_input.npz',input=packed,gpu_action=wanted_action,
                                        gpu_latent=wanted_latent,gpu_history=gpu_history.numpy())
                    print(json.dumps(debug),flush=True)
                    raise
                np.testing.assert_allclose(client.previous_latents[-1], wanted_latent, atol=2e-4, rtol=2e-4)
                max_action = max(max_action, float(np.max(np.abs(actual - wanted_action))))
                max_latent = max(max_latent, float(np.max(np.abs(client.previous_latents[-1] - wanted_latent))))
                for key, value_to_store in zip(records, (value, boundary, actual, wanted_action, wanted_latent), strict=True):
                    records[key].append(np.array(value_to_store, copy=True))
                expected[0] = torch.from_numpy(actual).to(env.device)
                action_term.record_policy_mean(expected)
                obs, _, dones, extras = wrapped.step(expected)
                boundary = bool((dones.bool() | extras["motion_resample_boundary"])[0])
                resets += int(boundary)
                if (step + 1) % 100 == 0:
                    print(json.dumps({"steps": step + 1, "max_action_error": max_action,
                                      "max_latent_error": max_latent, "resets": resets}), flush=True)
    finally:
        env.close()
    samples = args.output_dir / "rollout.npz"
    np.savez_compressed(samples, **{key: np.stack(value) for key, value in records.items()})
    program = '''
import builtins, importlib.util, json, sys
import numpy as np
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'mujoco', 'mjlab', 'intact_tracking'):
        raise RuntimeError('Deployment tried to import a training/simulation dependency: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location('standalone_runtime', sys.argv[1] + '/policy_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
client = module.Memory350Policy(sys.argv[1])
data = np.load(sys.argv[2])
maximum = 0.
for i, obs in enumerate(data['observations']):
    actual = client.step(obs, reset_boundary=bool(data['reset_boundary'][i]),
                         command_applied=data['commands'][i-1] if i else None)
    np.testing.assert_allclose(actual, data['expected_action'][i], atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(client.previous_latents[-1], data['expected_latent'][i], atol=2e-4, rtol=2e-4)
    maximum = max(maximum, float(np.max(np.abs(actual-data['expected_action'][i]))))
print(json.dumps({'steps': len(data['observations']), 'max_action_error': maximum,
                  'torch_and_simulator_imports_forbidden': True}))
'''
    independent = json.loads(subprocess.check_output(
        [sys.executable, "-I", "-c", program, str(args.export_dir.resolve()), str(samples.resolve())], text=True))
    report = {"passed": True, "checkpoint": str(args.checkpoint.resolve()), "steps": args.steps,
              "max_action_absolute_error": max_action, "max_latent_absolute_error": max_latent,
              "reset_count": resets, "long_chunks_at_end": len(client.history.chunks),
              "robot_pd_and_action_metadata_match_live_environment": True,
              "reference_precision": "FP32 policy and context encoder; fused MHA fastpath disabled",
              "independent_process": independent}
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
