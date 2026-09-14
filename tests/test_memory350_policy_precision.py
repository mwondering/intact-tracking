import pytest

from intact_tracking.memory350_policy_precision import resolve_policy_precision, configure_policy_precision


def test_evaluation_and_resume_inherit_precision_but_legacy_keeps_its_original_mode():
    assert resolve_policy_precision() == "tf32"
    assert resolve_policy_precision(None, {"residual_policy": {}}) == "tf32"
    checkpoint = {"residual_policy": {"policy_precision": "fp32"}}
    assert resolve_policy_precision(None, checkpoint) == "fp32"
    assert resolve_policy_precision("tf32", checkpoint) == "tf32"
    assert resolve_policy_precision("fp32", {"residual_policy": {}}) == "fp32"
    with pytest.raises(ValueError, match="precision"):
        resolve_policy_precision(None, {"residual_policy": {"policy_precision": "bad"}})


def test_fp32_configuration_disables_tf32_without_changing_default_dtype():
    import torch
    previous = torch.backends.cuda.matmul.fp32_precision
    previous_cudnn = torch.backends.cudnn.fp32_precision
    dtype = torch.get_default_dtype()
    try:
        record = configure_policy_precision("fp32")
        assert torch.backends.cuda.matmul.fp32_precision == "ieee"
        assert torch.backends.cudnn.fp32_precision == "ieee"
        assert torch.get_default_dtype() == dtype
        assert record["policy_precision"] == "fp32"
        configure_policy_precision("tf32")
        assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous
        torch.backends.cudnn.fp32_precision = previous_cudnn


def test_comparison_excludes_updates_collected_under_different_precisions(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_memory350_compressed_ppo import update_comparison
    for fusion, precision in (("baseline", "tf32"), ("concat", "fp32")):
        directory = tmp_path / "ppo" / f"{fusion}_121"
        directory.mkdir(parents=True)
        rows = [{"completed_updates": 100, "protocol_sha256": "same", "metrics": {}, "policy_precision": precision},
                {"completed_updates": 200, "protocol_sha256": "same", "metrics": {}, "policy_precision": "fp32"}]
        (directory / "endpoint_eval_metrics.jsonl").write_text("\n".join(map(json.dumps, rows)))
    update_comparison(tmp_path)
    result = json.loads((tmp_path / "ppo_comparison.json").read_text())
    assert result["matched_updates"] == [200]
    assert result["excluded_different_precision_updates"] == [100]
