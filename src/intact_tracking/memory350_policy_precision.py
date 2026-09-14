"""Checkpointed numerical precision for policy collection, PPO updates and evaluation."""

POLICY_PRECISIONS = ("tf32", "fp32")


def resolve_policy_precision(requested=None, checkpoint=None):
    saved = (checkpoint or {}).get("residual_policy", {}).get("policy_precision", "tf32")
    value = saved if requested is None else requested
    if value not in POLICY_PRECISIONS:
        raise ValueError(f"Unknown policy precision: {value!r}")
    return value


def configure_policy_precision(precision):
    import torch
    from mjlab.utils.torch import configure_torch_backends

    precision = resolve_policy_precision(precision)
    configure_torch_backends(allow_tf32=precision == "tf32")
    return {"policy_precision": precision, "policy_parameter_dtype": "float32",
            "policy_matmul_precision": "ieee" if precision == "fp32" else "tf32",
            "collection_update_and_evaluation_use_same_setting": True,
            "context_inference_autocast": "bfloat16 on CUDA; frozen context stored in float32",
            "torch_version": str(torch.__version__)}
