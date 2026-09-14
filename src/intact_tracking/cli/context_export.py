"""Export and numerically audit a self-contained CPU student inference artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from intact_tracking.adaptation_context_memory import CONTEXT_MEAN, update_context_mean
from intact_tracking.adaptation_sensors import configure_context_sensors
from intact_tracking.cli.adaptation_eval import DATASET, TRACKER, configure_physics, load_actor
from intact_tracking.cli.residual_policy_train import _seed_everything
from intact_tracking.context_export import (
    ContextInferenceModule,
    StatefulContextInferenceModule,
    deployable_fields,
)
from intact_tracking.environment.runtime import prepare_rollout
from intact_tracking.rollout.mjlab_adapter import _sha256


def independent_cpu_audits(output, repeats=8):
    """Fresh interpreters, only PyTorch; enforce the CPU execution contract."""
    program = """
import json, sys, torch
from pathlib import Path
torch.set_num_threads(1)
path = Path(sys.argv[1])
model = torch.jit.load(str(path / 'policy_cpu.ts'), map_location='cpu').eval()
data = torch.load(path / 'audit_samples.pt', map_location='cpu', weights_only=True)
maximum = 0.0
with torch.inference_mode():
    for size in (1, 2, 5, len(data['input'])):
        actions = []
        for start in range(0, len(data['input']), size):
            sl = slice(start, start + size)
            if 'previous_mean' in data:
                action, mean, count = model(data['input'][sl], data['previous_mean'][sl], data['previous_count'][sl], data['reset'][sl])
                torch.testing.assert_close(mean, data['expected_mean'][sl], atol=2e-4, rtol=2e-4)
                torch.testing.assert_close(count, data['expected_count'][sl], atol=0, rtol=0)
            else:
                action = model(data['input'][sl])
            actions.append(action)
        actual = torch.cat(actions)
        torch.testing.assert_close(actual, data['expected'], atol=2e-4, rtol=2e-4)
        maximum = max(maximum, float((actual - data['expected']).abs().max()))
    if 'previous_mean' in data:
        width = data['sequence_width']
        mean, count = torch.zeros_like(data['previous_mean'][:width]), torch.zeros_like(data['previous_count'][:width])
        for step in range(data['sequence_steps']):
            sl = slice(step * width, (step + 1) * width)
            action, mean, count = model(data['input'][sl], mean, count, data['reset'][sl])
            torch.testing.assert_close(action, data['expected'][sl], atol=2e-4, rtol=2e-4)
            maximum = max(maximum, float((action - data['expected'][sl]).abs().max()))
print(json.dumps({'max_absolute_action_difference': maximum, 'cpu_threads': 1}))
"""
    return [
        json.loads(subprocess.check_output([sys.executable, "-I", "-c", program, str(output.resolve())], text=True))
        for _ in range(repeats)
    ]


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Export output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(15443)
    prepared = prepare_rollout(
        checkpoint_file=TRACKER, num_envs=8, motion_path=DATASET, motion_file=None
    )
    configure_physics(prepared.env, "dr")
    actor_cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["cfg"].agent.actor
    if actor_cfg.get("context_imu_accel", False):
        configure_context_sensors(prepared.env)
    env = ManagerBasedRlEnv(cfg=prepared.env, device="cuda:0")
    inputs, expected = [], []
    memory_samples = {key: [] for key in (
        "previous_mean", "previous_count", "reset", "expected_mean", "expected_count"
    )}
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=prepared.clip_actions)
        obs = wrapped.get_observations()
        actor = load_actor(args.checkpoint, prepared, obs, wrapped)
        fields = deployable_fields(actor)
        memory = actor.context_latent_mean
        mean = torch.zeros(8, actor.adaptation_latent_dim, device=env.device)
        count = torch.zeros(8, 1, device=env.device)
        reset = torch.zeros(8, device=env.device, dtype=torch.bool)

        def sample_action(observation, previous_mean, previous_count, resets):
            deployable = observation.select(*(name for name, _ in fields))
            next_mean, next_count = previous_mean, previous_count
            if memory:
                latent = actor.instant_context_latent(deployable)
                next_mean, next_count = update_context_mean(
                    previous_mean, previous_count, latent, resets, actor.context_mean_horizon
                )
                deployable.set(CONTEXT_MEAN, next_mean)
                for key, value in zip(memory_samples, (
                    previous_mean, previous_count, resets, next_mean, next_count
                ), strict=True):
                    memory_samples[key].append(value.detach().cpu().clone())
            action = actor(deployable)
            inputs.append(torch.cat([deployable[name] for name, _ in fields], -1).cpu())
            expected.append(action.cpu())
            return action, next_mean, next_count

        with torch.inference_mode():
            for _ in range(12):
                action, mean, count = sample_action(obs, mean, count, reset)
                env.action_manager.get_term("joint_pos").record_policy_mean(action)
                obs, _, reset, _ = wrapped.step(action)
            if memory:
                # Additional numeric-only states exercise the prefix/EMA
                # boundary and selective resets. Never applied to the robot.
                horizon = actor.context_mean_horizon
                counts = count.new_tensor([0, 1, horizon - 1, horizon] * 2)[:, None]
                resets = reset.new_tensor([False] * 4 + [True] * 4)
                sample_action(obs, mean, counts, resets)
            module_cls = StatefulContextInferenceModule if memory else ContextInferenceModule
            module = module_cls(actor).cpu().eval().requires_grad_(False)
    finally:
        env.close()
    sample = torch.cat(inputs)
    reference = torch.cat(expected)
    audit_data = {"input": sample, "expected": reference}
    if memory:
        audit_data.update({key: torch.cat(values) for key, values in memory_samples.items()})
        audit_data.update(sequence_steps=12, sequence_width=8)
    arguments = (
        (sample, audit_data["previous_mean"], audit_data["previous_count"], audit_data["reset"])
        if memory else (sample,)
    )

    def check(output):
        action = output[0] if memory else output
        torch.testing.assert_close(action, reference, atol=2e-4, rtol=2e-4)
        if memory:
            torch.testing.assert_close(output[1], audit_data["expected_mean"], atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(output[2], audit_data["expected_count"], atol=0, rtol=0)
        return action

    with torch.no_grad():
        check(module(*arguments))
        # Warm the constant robot-kinematics cache before tracing. Check multiple
        # batch sizes and states to detect baked-in batch shapes or observations.
        module(*(value[:2] for value in arguments))
        traced = torch.jit.trace(
            module, tuple(value[:2] for value in arguments),
            check_inputs=[tuple(value[2:3] for value in arguments), tuple(value[-8:] for value in arguments)],
        )
        traced = torch.jit.freeze(traced.eval())
        traced.save(str(output / "policy_cpu.ts"))
        restored = torch.jit.load(str(output / "policy_cpu.ts"), map_location="cpu")
        actual = check(restored(*arguments))
    torch.save(audit_data, output / "audit_samples.pt")
    independent_audits = independent_cpu_audits(output)
    manifest = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "artifact_sha256": _sha256(output / "policy_cpu.ts"),
        "input_fields_in_order": list(fields),
        "input_shape": ["batch", sum(width for _, width in fields)],
        "output_shape": ["batch", 29],
        "device": "cpu",
        "audit_samples": len(sample),
        "max_absolute_action_difference": float((actual - reference).abs().max()),
        "required_cpu_threads": 1,
        "gpu_reference_precision": "FP32 matmul and FP32 cuDNN convolution (TF32 disabled for numerical audit)",
        "independent_process_audits": independent_audits,
        "privileged_inference_fields": [],
        "stateful_context_memory": memory,
        "memory_contract": {
            "latent_dim": actor.adaptation_latent_dim,
            "horizon": actor.context_mean_horizon,
            "mean_only": actor.context_mean_only,
            "controller_code": "causal mean only" if actor.context_mean_only else "current latent concatenated with causal mean",
            "inputs": ["sensor_tensor", "previous_mean[B,D]", "count[B,1]", "reset[B]"],
            "outputs": ["action[B,29]", "updated_mean[B,D]", "updated_count[B,1]"],
            "initial_state": "float32 zeros; boolean reset; one call per sensor control step",
            "audit": "12-step sequential carry plus synthetic prefix/EMA boundary and selective-reset states",
        } if memory else None,
        "notes": "Includes preprocessing, context encoder, base policy and residual. Runtime needs only TorchScript and must call torch.set_num_threads(1) before inference: rare cross-process multi-thread CPU discrepancies were observed in this environment. Sensor/FK history preparation and G1 action scaling remain the deployment adapter's responsibility. Not a hardware safety validation.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    run(args)


if __name__ == "__main__":
    main()
