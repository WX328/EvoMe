"""S16.2 regression: output JSON is not a workflow snapshot.

Do not execute Workflow or read the user's private .workflows directory.
The user runs pytest against these isolated tmp_path fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.workflow_store import WorkflowStore


def reserve_done(store: WorkflowStore) -> str:
    row = store.reserve("review-files", {"paths": ["app/agent/schema.py"]})
    run_id = row["id"]
    row["status"] = "completed"
    store.save(row)
    store.write_output(run_id, {"status": "reviewed", "count": 1})
    store.unlock(run_id)
    return run_id


def test_existing_completed_run_with_output_is_listed_once(tmp_path):
    store = WorkflowStore(tmp_path)
    old_id = reserve_done(store)
    names = {p.name for p in (tmp_path / ".workflows").iterdir()}
    assert f"{old_id}.json" in names
    assert f"{old_id}.output.json" in names

    rows = store.list()
    assert len(rows) == 1
    assert rows[0]["id"] == old_id
    assert rows[0]["status"] == "completed"
    assert store.read_output(old_id)["status"] == "reviewed"


def test_new_run_can_be_reserved_after_old_run_has_output(tmp_path):
    store = WorkflowStore(tmp_path)
    old_id = reserve_done(store)
    second = store.reserve("review-files", {"paths": ["app/background.py"]})
    try:
        assert second["id"] != old_id
        assert {row["id"] for row in store.list()} == {old_id, second["id"]}
    finally:
        store.unlock(second["id"])


def test_multiple_outputs_are_not_counted_as_additional_runs(tmp_path):
    store = WorkflowStore(tmp_path)
    one = reserve_done(store)
    two = reserve_done(store)
    assert len(store.list()) == 2
    assert {row["id"] for row in store.list()} == {one, two}


@pytest.mark.parametrize("bad_name", [
    "wf_not_a_valid_run_id.output.json",
    "wf_aaaaaaaaaaaa.extra.json",
    "wf_aaaaaaaaaaaa.output.output.json",
])
def test_unexpected_json_file_names_still_fail_closed(tmp_path, bad_name):
    store = WorkflowStore(tmp_path)
    reserve_done(store)
    (tmp_path / ".workflows" / bad_name).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        store.list()


def test_output_file_does_not_hide_invalid_snapshot_contents(tmp_path):
    store = WorkflowStore(tmp_path)
    run_id = reserve_done(store)
    (tmp_path / ".workflows" / f"{run_id}.json").write_text(
        '{"wrong_schema":true}', encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot schema"):
        store.list()


def test_only_valid_output_json_skipped_not_journal_deleted(tmp_path):
    store = WorkflowStore(tmp_path)
    run_id = reserve_done(store)
    journal = tmp_path / ".workflows" / f"{run_id}.journal.jsonl"
    journal.write_text('{"key":"fake","value":1}\n', encoding="utf-8")
    before = sorted(p.name for p in journal.parent.iterdir())
    assert len(store.list()) == 1
    after = sorted(p.name for p in journal.parent.iterdir())
    assert after == before
    assert journal.exists()
