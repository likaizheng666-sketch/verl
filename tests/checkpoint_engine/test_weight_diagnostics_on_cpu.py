# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import hashlib
import json

import pytest
import torch

from tests.checkpoint_engine.weight_diagnostics import (
    diagnose_sender,
    diagnostic_env,
    fingerprint,
    mismatch_details,
    record,
    summarize,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_fingerprint_chunk_boundaries_and_layout(dtype):
    tensor = torch.arange(24).reshape(4, 6).to(dtype).T
    expected = hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    result = fingerprint(tensor, chunk_bytes=17)
    assert result == {"dtype": str(dtype), "shape": [6, 4], "sha256": expected}
    assert torch.equal(tensor, torch.arange(24).reshape(4, 6).to(dtype).T)


def test_reference_fingerprint_uses_comparison_dtype():
    reference = torch.tensor([0.5, -2.0], dtype=torch.bfloat16)
    assert fingerprint(reference, dtype=torch.float32) == fingerprint(reference.float())


def test_empty_and_scalar_fingerprints():
    assert fingerprint(torch.empty(0))["sha256"] == hashlib.sha256(b"").hexdigest()
    assert fingerprint(torch.tensor(1.0))["shape"] == []


def test_mismatch_details_handles_corruption_and_nonfinite():
    expected = torch.tensor([1.0, 2.0, float("nan"), float("inf")], dtype=torch.bfloat16)
    received = torch.tensor([1.0, 5.0, float("nan"), float("inf")])
    details = mismatch_details(expected, received, chunk_bytes=5)
    assert details["cpu_isclose_mismatches"] == 2
    assert details["nonfinite_mismatches"] == 1
    assert details["max_abs_finite"] == 3.0
    assert details["compared_dtype"] == "torch.float32"
    json.dumps(details, allow_nan=False)


def test_sender_is_lazy_and_preserves_tensor_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", str(tmp_path))
    tensor = torch.tensor([1.0, 2.0])
    iterator = diagnose_sender(iter([("weight", tensor)]), 2)
    assert not list(tmp_path.glob("*.jsonl"))
    name, observed = next(iterator)
    assert name == "weight" and observed is tensor
    entry = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert entry["stage"] == "sender" and entry["update_index"] == 2


def test_summarize_distinguishes_export_and_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTIC_CASE", "hccl-3072")
    original, changed = torch.tensor([1.0]), torch.tensor([2.0])
    for name, source, received, reference in (
        ("good", original, original, original),
        ("transport", original, changed, original),
        ("export_or_reference", changed, changed, original),
    ):
        for stage, tensor in (("sender", source), ("received", received), ("reference", reference)):
            record(stage, name, 0, **fingerprint(tensor))
    record("received", "missing_sender", 0, **fingerprint(original))
    record("mismatch", "transport", 0, **mismatch_details(original, changed))
    # Case/update boundaries must not join a different sender to this receiver.
    record("sender", "missing_sender", 1, **fingerprint(original))
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTIC_CASE", "hccl-128")
    record("sender", "missing_sender", 0, **fingerprint(original))
    result = summarize(tmp_path)
    assert result["compared"] == 3 and result["incomplete"] == 1
    differences = {entry["name"]: entry for entry in result["differences"]}
    assert not differences["transport"]["sender_equals_received"]
    assert differences["export_or_reference"]["sender_equals_received"]
    assert not differences["export_or_reference"]["received_equals_reference"]
    assert result["assertion_failures"][0]["name"] == "transport"
    assert result["assertion_failures"][0]["cpu_isclose_mismatches"] == 1
    assert json.loads((tmp_path / "comparison.json").read_text()) == result


def test_diagnostic_env_does_not_copy_unrelated_settings(monkeypatch):
    monkeypatch.setenv("VERL_CHECKPOINT_DIAGNOSTICS_DIR", "diagnostics")
    monkeypatch.setenv("VERL_CHECKPOINT_WEIGHT_FINGERPRINTS", "1")
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-copied")
    assert diagnostic_env("hccl-128") == {
        "VERL_CHECKPOINT_DIAGNOSTICS_DIR": "diagnostics",
        "VERL_CHECKPOINT_WEIGHT_FINGERPRINTS": "1",
        "VERL_CHECKPOINT_DIAGNOSTIC_CASE": "hccl-128",
    }


def test_incomplete_jsonl_is_reported(tmp_path):
    (tmp_path / "interrupted.jsonl").write_text('{"stage":', encoding="utf-8")
    result = summarize(tmp_path)
    assert result["unreadable"] == 1
    assert result["compared"] == 0
