"""S09: workspace Agent Memory storage, selection, approval and isolation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent import AgentLoop
from app.agent.memory import AgentMemoryStore
from app.agent.memory_pipeline import AgentMemoryPipeline
from app.cli_commands import normalize_cli_input, looks_like_operational_input
from app.skills.registry import SkillRegistry
from app.tools.builtin import build_default_tool_registry
from app.tools.permissions import PermissionPolicy


def make_store(tmp_path, **kwargs):
    return AgentMemoryStore(workspace_root=tmp_path, **kwargs)


def persist(
    store, name="python-tabs", memory_type="user",
    description="用户喜欢用制表符进行 Python 缩进",
    body="以后 Python 项目默认使用 tabs 缩进。",
):
    return store.add(
        name=name, memory_type=memory_type,
        description=description, body=body,
    )


def test_catalog_startup_is_read_only_and_default_empty(tmp_path):
    store = make_store(tmp_path)
    assert store.catalog() == []
    assert store.recall("Python")["selected"] == 0
    assert not (tmp_path / ".memory").exists()


def test_add_file_and_index_and_read_exact_body(tmp_path):
    store = make_store(tmp_path)
    record = persist(store)
    assert record.name == "python-tabs"
    assert record.memory_type == "user"
    assert record.relative_path == ".memory/python-tabs.md"
    saved = tmp_path / ".memory" / "python-tabs.md"
    index = tmp_path / ".memory" / "MEMORY.md"
    assert saved.is_file() and index.is_file()
    assert "name: \"python-tabs\"" in saved.read_text(encoding="utf-8")
    assert "[python-tabs](python-tabs.md)" in index.read_text(encoding="utf-8")
    assert store.read("python-tabs").body == "以后 Python 项目默认使用 tabs 缩进。"
    assert set(store.catalog()[0]) == {"name", "type", "description", "path"}


def test_catalog_does_not_load_memory_bodies(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    persist(store)
    original_read = Path.read_text

    def blocked_body(path, *args, **kwargs):
        if path.name == "python-tabs.md":
            raise AssertionError("catalog selection eagerly loaded a memory body")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", blocked_body)
    assert store.catalog()[0]["name"] == "python-tabs"
    assert store.select("Python 缩进")[0]["name"] == "python-tabs"


def test_cross_session_recall_and_unrelated_memory_filtered(tmp_path):
    first = make_store(tmp_path)
    persist(first)
    persist(
        first, name="ci-troubleshooting", memory_type="project",
        description="项目 CI 构建失败时先查看 Github Actions 日志",
        body="EvoMe CI 异常优先核对 Actions 配置。",
    )
    second = make_store(tmp_path)
    selected = second.recall("Python 缩进用什么")
    assert selected["selected"] >= 1
    assert selected["memories"][0]["name"] == "python-tabs"
    assert selected["total_chars"] <= 1800
    assert all(item["name"] != "ci-troubleshooting" for item in selected["memories"])


def test_recall_limits_total_loaded_text_and_count(tmp_path):
    store = make_store(tmp_path, max_recall_chars=400, max_recall=3)
    for n in range(6):
        persist(
            store, name=f"python-pref-{n}",
            description=f"Python 缩进习惯 {n}",
            body=f"Python code preference {n}: " + "a" * 250,
        )
    result = store.recall("Python 缩进习惯")
    assert result["selected"] <= 3
    assert result["total_chars"] <= 400
    assert all(item["body"] for item in result["memories"])


def test_existing_memory_never_silently_overwritten(tmp_path):
    store = make_store(tmp_path)
    persist(store)
    with pytest.raises(ValueError, match="already exists"):
        persist(store, body="不要修改我的历史记录")
    assert store.read("python-tabs").body == "以后 Python 项目默认使用 tabs 缩进。"


def test_duplicate_body_rejected_before_persistence(tmp_path):
    store = make_store(tmp_path)
    persist(store)
    with pytest.raises(ValueError, match="duplicate"):
        persist(store, name="duplicate-tabs", body="以后Python项目默认使用tabs缩进。")
    assert not (tmp_path / ".memory" / "duplicate-tabs.md").exists()


def test_invalid_names_types_secrets_and_multiline_metadata_rejected(tmp_path):
    store = make_store(tmp_path)
    for name in ("../.env", "a/b", "UPPER", ".", "", "escape\\nested"):
        with pytest.raises(ValueError):
            store.add(name=name, memory_type="project", description="safe", body="safe")
    with pytest.raises(ValueError):
        store.add(name="safe-name", memory_type="unknown", description="safe", body="safe")
    with pytest.raises(ValueError):
        store.add(name="safe-name", memory_type="project",
                  description="hello\nworld", body="safe")
    with pytest.raises(ValueError, match="protected"):
        store.add(name="safe-name", memory_type="project",
                  description="safe", body="api_key=THIS_MUST_NOT_LEAK")


def test_symlink_root_rejected(tmp_path):
    root = tmp_path / ".memory"
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        root.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Windows system cannot create directory symlinks")
    with pytest.raises(PermissionError):
        make_store(tmp_path)


def test_symlink_record_not_read_or_selected(tmp_path):
    store = make_store(tmp_path)
    persist(store)
    outside = tmp_path / "outside.md"
    outside.write_text(
        store._serialize("sneaky", "project", "Secret project", "secret"),
        encoding="utf-8",
    )
    try:
        (tmp_path / ".memory" / "sneaky.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation unavailable")
    assert "sneaky" not in [row["name"] for row in store.catalog()]
    with pytest.raises(LookupError):
        store.read("sneaky")


def test_partial_disk_write_rolls_back_candidate_and_index(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    persist(store)
    original = (tmp_path / ".memory" / "MEMORY.md").read_text(encoding="utf-8")
    write = store._atomic_write
    failed = False

    def unreliable(target, data):
        nonlocal failed
        if target.name == "MEMORY.md" and not failed:
            failed = True
            raise OSError("simulate index disk failure")
        return write(target, data)

    monkeypatch.setattr(store, "_atomic_write", unreliable)
    with pytest.raises(OSError):
        persist(store, name="new-fact", memory_type="project",
                description="another fact", body="something stable")
    assert not (tmp_path / ".memory" / "new-fact.md").exists()
    assert (tmp_path / ".memory" / "MEMORY.md").read_text(encoding="utf-8") == original


def test_consolidate_exact_same_type_duplicates_with_preview(tmp_path):
    store = make_store(tmp_path)
    persist(store)
    root = tmp_path / ".memory"
    (root / "copy-tabs.md").write_text(
        store._serialize(
            "copy-tabs", "user", "重复的缩进偏好",
            "以后 Python 项目默认使用 tabs 缩进。",
        ),
        encoding="utf-8",
    )
    preview = store.preview_consolidation()
    assert preview["exact_duplicate_groups"] == [
        {"keep": "copy-tabs", "remove": ["python-tabs"]}
    ]
    assert (root / "python-tabs.md").exists()
    applied = store.consolidate_duplicates()
    assert applied["removed"] == ["python-tabs"]
    assert not (root / "python-tabs.md").exists()
    assert "copy-tabs" in (root / "MEMORY.md").read_text(encoding="utf-8")
    assert store.consolidate_duplicates()["removed"] == []


def test_consolidation_does_not_auto_delete_conflicts(tmp_path):
    store = make_store(tmp_path)
    persist(store, name="prefer-tabs", body="以后默认 tabs 缩进")
    persist(store, name="prefer-spaces", body="以后默认 spaces 缩进")
    assert store.preview_consolidation()["exact_duplicate_groups"] == []
    assert store.consolidate_duplicates()["removed"] == []
    assert len(store.catalog()) == 2


class ProposerLLM:
    def __init__(self, result):
        self.result = result
        self.requests = []

    def json_chat(self, messages, **kwargs):
        self.requests.append(([dict(item) for item in messages], kwargs))
        return self.result


def offer(name, quote, *, description="Python 缩进习惯", typ="user", scope="persistent"):
    return {"name": name, "type": typ, "description": description,
            "scope": scope, "quote": quote}


def test_candidate_extract_is_staged_until_explicit_approval(tmp_path):
    utterance = "以后 Python 开发默认用 tabs 缩进，记住这个习惯。"
    quote = "以后 Python 开发默认用 tabs 缩进"
    model = ProposerLLM({"candidates": [offer("python-tabs", quote)]})
    store = make_store(tmp_path)
    pipeline = AgentMemoryPipeline(store=store, llm=model)
    staged = pipeline.propose(utterance)
    assert len(staged["pending"]) == 1
    assert not (tmp_path / ".memory").exists()
    assert len(model.requests) == 1
    assert model.requests[0][0][1]["content"] == utterance
    candidate_id = staged["pending"][0]["id"]
    stored = pipeline.confirm(candidate_id)
    assert stored.body == quote
    assert pipeline.list_pending() == []
    assert (tmp_path / ".memory" / "python-tabs.md").exists()
    with pytest.raises(KeyError):
        pipeline.confirm(candidate_id)


def test_current_task_hypothetical_and_non_durable_are_not_extracted(tmp_path):
    model = ProposerLLM({"candidates": [offer("fake", "just a test")]})
    pipeline = AgentMemoryPipeline(store=make_store(tmp_path), llm=model)
    for utterance in (
        "这次临时只读取文件，不修改文件。",
        "本次任务仅使用 Python 3.13。",
        "假设用户喜欢 tabs，要怎么保存？",
        "分析为什么代码会报错。",
    ):
        assert pipeline.propose(utterance)["pending"] == []
    assert model.requests == []
    assert not (tmp_path / ".memory").exists()


def test_hallucinated_quote_and_current_task_scope_rejected(tmp_path):
    text = "以后写 Python 时记住要使用 tabs 缩进。"
    model = ProposerLLM({"candidates": [
        offer("invented", "用户热爱量子计算"),
        offer("temporary", "以后写 Python 时记住要使用 tabs 缩进", scope="current_task"),
    ]})
    pipeline = AgentMemoryPipeline(store=make_store(tmp_path), llm=model)
    outcome = pipeline.propose(text)
    assert outcome["pending"] == []
    assert outcome["rejected"] == 2


def test_deny_pending_proposal_prevents_write(tmp_path):
    quote = "以后偏好 tabs"
    model = ProposerLLM({"candidates": [offer("python-tabs", quote)]})
    pipeline = AgentMemoryPipeline(store=make_store(tmp_path), llm=model)
    proposed = pipeline.propose(quote)
    candidate_id = proposed["pending"][0]["id"]
    assert pipeline.deny(candidate_id)
    assert not pipeline.deny(candidate_id)
    assert not (tmp_path / ".memory").exists()


class SimpleLLM:
    def __init__(self):
        self.snapshots = []

    def json_chat(self, messages, **kwargs):
        self.snapshots.append([dict(msg) for msg in messages])
        return {"action": "final", "task_complete": True}

    def chat(self, messages, **kwargs):
        return "已参考相关工作记忆"


def test_agent_context_uses_user_role_not_privileged_system(tmp_path):
    store = make_store(tmp_path)
    persist(store)
    llm = SimpleLLM()
    registry = build_default_tool_registry(
        skill_registry=SkillRegistry(), workspace_root=tmp_path,
    )
    agent = AgentLoop(
        llm=llm, tool_registry=registry,
        permission_policy=PermissionPolicy(),
        agent_memory_store=store,
    )
    result = agent.run(user_input="Python 缩进用什么")
    assert result.termination_reason == "final_answer"
    observed = llm.snapshots[0]
    memory_messages = [
        row for row in observed
        if row["content"].startswith("[AGENT_MEMORY_RECALL]")
    ]
    assert len(memory_messages) == 1
    assert memory_messages[0]["role"] == "user"
    assert "tabs" in memory_messages[0]["content"]
    assert "[AGENT_MEMORY_RECALL]" not in str([
        row for row in observed if row["role"] == "system"
    ])


def test_agent_new_store_has_no_recall_when_irrelevant(tmp_path):
    store = make_store(tmp_path)
    persist(store)
    llm = SimpleLLM()
    agent = AgentLoop(
        llm=llm,
        tool_registry=build_default_tool_registry(
            skill_registry=SkillRegistry(), workspace_root=tmp_path,
        ),
        agent_memory_store=store,
    )
    agent.run(user_input="请计算天气数据方差")
    assert "[AGENT_MEMORY_RECALL]" not in str(llm.snapshots[0])


def test_subagent_does_not_inherit_agent_memory_or_write_permissions(tmp_path):
    store = make_store(tmp_path)
    persist(store)
    llm = SimpleLLM()
    agent = AgentLoop(
        llm=llm,
        tool_registry=build_default_tool_registry(
            skill_registry=SkillRegistry(), workspace_root=tmp_path,
        ),
        agent_memory_store=store,
        subagent_enabled=True,
    )
    tools = {item.name for item in agent.subagent.registry.list_tools()}
    assert tools == {"read_file", "glob"}
    assert "recall_agent_memory" in {t.name for t in agent.tool_registry.list_tools()}
    assert "recall_agent_memory" not in tools
    assert "write_memory" not in tools


def test_memory_cli_commands_are_operational_not_learning_samples():
    assert normalize_cli_input("agent-memory-list") == "/agent-memory-list"
    assert normalize_cli_input("agent-memory-propose 以后默认 tabs") == (
        "/agent-memory-propose 以后默认 tabs"
    )
    assert looks_like_operational_input("/agent-memory-approve abcd123456")
    assert looks_like_operational_input("agent-memory-consolidate")


def test_s03_denies_direct_agent_writes_to_s09_memory_directory(tmp_path):
    store = make_store(tmp_path)
    policy = PermissionPolicy(
        workspace_root=tmp_path,
        protected_memory_directory=store.root,
    )
    for tool_name in ("write_file", "edit_file"):
        for path in (".memory/hijack.md", "skills/../.memory/hijack.md"):
            decision = policy.evaluate(tool_name, {"path": path, "content": "bad"})
            assert decision.action == "deny"
            assert decision.risk == "high"
    assert policy.evaluate(
        "write_file", {"path": "ordinary_source.py", "content": "ok"}
    ).action == "ask"
    assert policy.evaluate(
        "read_file", {"path": ".memory/python-tabs.md"}
    ).action == "allow"


def test_consolidation_rolls_back_records_after_index_failure(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    persist(store, name="first-tabs")
    root = tmp_path / ".memory"
    (root / "second-tabs.md").write_text(
        store._serialize("second-tabs", "user", "duplicate", "以后 Python 项目默认使用 tabs 缩进。"),
        encoding="utf-8",
    )
    names_before = sorted(path.name for path in root.glob("*.md"))
    index_before = (root / "MEMORY.md").read_text(encoding="utf-8")
    original = store._atomic_write

    def fail_index(target, data):
        if target.name == "MEMORY.md":
            raise OSError("simulated disk full")
        return original(target, data)

    monkeypatch.setattr(store, "_atomic_write", fail_index)
    with pytest.raises(OSError):
        store.consolidate_duplicates()
    assert sorted(path.name for path in root.glob("*.md")) == names_before
    assert (root / "MEMORY.md").read_text(encoding="utf-8") == index_before


def test_secret_quote_never_enters_pending_store(tmp_path):
    sentence = "以后记住 api_key=TEST_SECRET_VALUE 这个密钥。"
    model = ProposerLLM({"candidates": [
        offer("bad-secret", sentence, description="secret quote")
    ]})
    pipeline = AgentMemoryPipeline(store=make_store(tmp_path), llm=model)
    result = pipeline.propose(sentence)
    assert result["pending"] == []
    assert result["rejected"] == 1
    assert not (tmp_path / ".memory").exists()
