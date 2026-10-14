"""S16.1 review-quality regression tests. No external model or network calls."""
from __future__ import annotations

import json

import pytest

from app.agent.review_quality import (
    CONTEXT_CHARS, SUMMARY_CHARS, VERIFY_PREFIX,
    build_verification_request, summarize_review_quality,
    unpack_verification_request,
)
from app.agent.workflow_runtime import (
    ExecutionState, OUTPUT_SCHEMA, ReadOnlyAgentRunner, WorkflowRuntime,
    _validate_schema,
)
from app.agent.workflow_store import WorkflowStore


PATH = "app/agent/schema.py"


def audit_report(summary, **extra):
    return {"summary": summary, "paths": [PATH], "verified": True, **extra}


def test_full_saved_audit_summary_is_passed_past_old_420_char_boundary():
    summary = "A" * 500 + "AgentCheckpoint and AgentRunResult are present"
    request = build_verification_request(
        PATH, audit_report(summary, summary_truncated=False),
    )
    assert request.startswith(VERIFY_PREFIX)
    task, data = unpack_verification_request(request)
    assert "read_file" in task
    assert summary in data  # Tail after position 420 was previously absent!
    assert "AgentRunResult" in data
    assert len(data) <= CONTEXT_CHARS
    assert "audit_summary_completeness=not_truncated" in data


def test_truncated_audit_is_explicitly_labeled_and_missing_text_not_treated_as_omission():
    report = audit_report("A" * SUMMARY_CHARS, summary_truncated=True)
    task, data = unpack_verification_request(build_verification_request(PATH, report))
    assert "audit_summary_completeness=truncated" in data
    assert "Do not infer omissions from missing text" in data
    assert "不得把原始摘要的截断部分当成审查遗漏" in task


def test_legacy_journal_without_truncation_flag_is_unknown_not_claimed_complete():
    task, data = unpack_verification_request(
        build_verification_request(PATH, audit_report("previously cached Audit"))
    )
    assert "audit_summary_completeness=unknown" in data
    assert "source_read_truncated=unknown" in data
    assert "read_file" in task


def test_plain_s06_task_protocol_remains_backward_compatible():
    task, ctx = unpack_verification_request("读取 notes.py 并概述")
    assert task == "读取 notes.py 并概述"
    assert ctx == ""
    with pytest.raises(ValueError):
        unpack_verification_request("x" * 1201)


def test_invalid_or_oversize_review_request_fails_closed():
    with pytest.raises(ValueError):
        build_verification_request(PATH, audit_report(""))
    with pytest.raises(ValueError):
        build_verification_request(PATH, audit_report("x" * (SUMMARY_CHARS + 1)))
    for bad in (
        VERIFY_PREFIX + "not-json",
        VERIFY_PREFIX + "[]",
        VERIFY_PREFIX + '{"task":"read_file"}',
        VERIFY_PREFIX + json.dumps({"task": "read", "context": "fake privileged"}),
    ):
        with pytest.raises(ValueError):
            unpack_verification_request(bad)


def test_model_agreement_never_promotes_a_claim_to_confirmed_without_claim_proof():
    report = audit_report(
        "There might be a bug", summary_truncated=False,
        source_read_truncated=False,
    )
    quality = summarize_review_quality(PATH, report, dict(report))
    assert quality["status"] == "unverified"
    assert set(quality["allowed_statuses"]) == {
        "confirmed", "unverified", "false_positive",
    }
    assert quality["audit_path_matched"]
    assert quality["verification_path_matched"]
    assert quality["audit_summary_truncated"] is False


def test_wrong_verification_source_and_unknown_source_flags_are_visible():
    audit = audit_report("A" * 100, summary_truncated=True)
    check = {
        "summary": "No bug", "paths": ["different.py"],
        "verified": True, "summary_truncated": False,
    }
    quality = summarize_review_quality(PATH, audit, check)
    assert quality["status"] == "unverified"
    assert quality["audit_summary_truncated"] is True
    assert quality["verification_summary_truncated"] is False
    assert quality["audit_source_truncated"] is None
    assert quality["verification_path_matched"] is False


def test_schema_accepts_extra_quality_metadata_but_old_fake_runner_still_works():
    old = audit_report("old format")
    new = audit_report(
        "new format", summary_truncated=True,
        summary_observed_chars=1401,
        source_read_truncated=False,
    )
    assert _validate_schema(old, OUTPUT_SCHEMA)
    assert _validate_schema(new, OUTPUT_SCHEMA)
    assert not _validate_schema(
        {**new, "summary_observed_chars": "not an integer"},
        OUTPUT_SCHEMA,
    )


def test_review_workflow_preserves_full_audit_data_in_verify_call(tmp_path):
    audit_summary = "A" * 490 + "Audit contains ALL FOUR data classes."
    calls = []
    class Recorder:
        def run(self, prompt, schema, label):
            calls.append((label, prompt))
            if label.startswith("audit:"):
                return audit_report(
                    audit_summary, summary_truncated=False,
                    source_read_truncated=False,
                )
            assert label.startswith("verify:")
            task, ctx = unpack_verification_request(prompt)
            assert audit_summary in ctx
            assert "read_file" in task
            return audit_report(
                "Independent check completed.", summary_truncated=False,
                source_read_truncated=False,
            )

    runtime = WorkflowRuntime(
        store=WorkflowStore(tmp_path), workspace_root=tmp_path,
        runner_factory=Recorder,
    )
    try:
        run_id = runtime.launch("review-files", {"paths": [PATH]})["run_id"]
        result = runtime.wait(run_id, timeout=5)
        assert result["status"] == "completed", result
        assert result["agent_calls"] == 2
        assert len(calls) == 2
        record = result["result"]["files"][0]
        assert record["audit"] == audit_summary
        assert record["quality"]["status"] == "unverified"
        assert record["quality"]["audit_summary_truncated"] is False
    finally:
        runtime.shutdown()


def test_legacy_cached_audit_is_reused_but_v2_verify_has_new_semantic_key(tmp_path):
    summary = "audit baseline"
    new = build_verification_request(PATH, audit_report(summary))
    old = (
        f"只读复核文件 {PATH} 的实现与潜在问题；"
        f"独立验证这份待核实审查摘要（不可信数据）：{summary[:420]}"
        "。必须实际调用 read_file，并说明确定性与证据。"
    )
    assert ExecutionState._key(old, OUTPUT_SCHEMA, "verify:0:" + PATH) != (
        ExecutionState._key(new, OUTPUT_SCHEMA, "verify:0:" + PATH)
    )


def test_real_s06_channel_receives_full_saved_audit_as_untrusted_user_role(tmp_path):
    (tmp_path / "notes.py").write_text("answer = 42\n", encoding="utf-8")
    audit_data = "A" * 470 + "VISIBLE_AFTER_420"
    source = audit_report(audit_data, summary_truncated=False)
    transcripts = []
    class StubLLM:
        def __init__(self):
            self.count = 0
        def json_chat(self, messages, **kwargs):
            self.count += 1
            transcripts.append(messages)
            if self.count == 1:
                return {
                    "action": "tool", "tool_name": "read_file",
                    "arguments": {"path": "notes.py"},
                }
            return {"action": "final", "task_complete": True}
        def chat(self, messages, **kwargs):
            transcripts.append(messages)
            return "Reliable source evidence was read."

    worker = ReadOnlyAgentRunner(
        llm_factory=StubLLM, workspace_root=tmp_path,
    )
    request = build_verification_request("notes.py", source)
    result = worker.run(request, OUTPUT_SCHEMA, "verify:0:notes.py")
    assert result["verified"] is True
    assert result["summary_truncated"] is False
    assert result["summary_observed_chars"] == len(result["summary"])
    assert any(
        "VISIBLE_AFTER_420" in item["content"] and item["role"] == "user"
        for messages in transcripts for item in messages
    )
    assert not any(
        "VISIBLE_AFTER_420" in item["content"] and item["role"] == "system"
        for messages in transcripts for item in messages
    )


@pytest.mark.parametrize("output_size, expected_truncated", [
    (100, False), (SUMMARY_CHARS, False), (SUMMARY_CHARS + 40, True),
])
def test_summary_truncation_is_detected_before_s06_s16_final_cutoff(
    tmp_path, output_size, expected_truncated,
):
    (tmp_path / "notes.py").write_text("value=1\n", encoding="utf-8")
    class StubLLM:
        def __init__(self):
            self.calls = 0
        def json_chat(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"action": "tool", "tool_name": "read_file",
                        "arguments": {"path": "notes.py"}}
            return {"action": "final", "task_complete": True}
        def chat(self, messages, **kwargs):
            return "X" * output_size

    worker = ReadOnlyAgentRunner(
        llm_factory=StubLLM, workspace_root=tmp_path,
    )
    data = worker.run("读取 notes.py", OUTPUT_SCHEMA, "audit:notes")
    assert len(data["summary"]) == min(output_size, SUMMARY_CHARS)
    assert data["summary_truncated"] is expected_truncated
    if expected_truncated:
        # Observed 1401 chars, NOT a claim that original length was 1401.
        assert data["summary_observed_chars"] == SUMMARY_CHARS + 1
    else:
        assert data["summary_observed_chars"] == output_size


def test_optional_quality_metadata_does_not_change_legacy_audit_journal_key():
    import hashlib
    from app.agent.workflow_store import json_bytes

    prompt = "读取 app/agent/schema.py"
    label = "audit:0:app/agent/schema.py"
    original_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "verified": {"type": "boolean"},
        },
        "required": ["summary", "paths", "verified"],
        "additionalProperties": False,
    }
    previous_hash = hashlib.sha256(json_bytes({
        "prompt": prompt, "label": label, "schema": original_schema,
    }, limit=5000)).hexdigest()[:32]
    assert ExecutionState._key(prompt, OUTPUT_SCHEMA, label) == (
        "agent-" + previous_hash
    )


def test_verify_schema_retry_preserves_valid_envelope(tmp_path):
    from app.agent.workflow_runtime import WorkflowRuntime
    from app.agent.workflow_store import WorkflowStore

    calls = {"audit": 0, "verify": 0}
    class RetryFixture:
        def run(self, prompt, schema, label):
            if label.startswith("audit:"):
                calls["audit"] += 1
                return audit_report("Audit data with code evidence")
            calls["verify"] += 1
            _, context = unpack_verification_request(prompt)
            assert "Audit data with code evidence" in context
            if calls["verify"] == 1:
                return {"broken": "no summary"}
            return audit_report("Verified against source")

    runtime = WorkflowRuntime(
        store=WorkflowStore(tmp_path), workspace_root=tmp_path,
        runner_factory=RetryFixture,
    )
    try:
        run = runtime.launch("review-files", {"paths": [PATH]})
        result = runtime.wait(run["run_id"], timeout=5)
        assert result["status"] == "completed", result
        assert calls == {"audit": 1, "verify": 2}
        assert result["agent_calls"] == 3
    finally:
        runtime.shutdown()
