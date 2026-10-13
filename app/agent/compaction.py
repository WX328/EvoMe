"""S08 deterministic Context Compact for EvoMe's bounded Agent Loop.

Only the model-facing transcript is compacted. AgentStep.tool_output,
ToolCall, approval IDs, and seen_calls remain in the independent execution
checkpoint, so compression can never create permission/duplicate-call bypasses.

No LLM summarization, shell execution, file IO, or persistence occurs here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


MARKER = "[AGENT_CONTEXT_COMPACTED]"
TRUNCATED = "[CONTEXT_EXCERPT_TRUNCATED]"
_MAX_HISTORY_RECORDS = 18


class ContextBudgetError(ValueError):
    """Required system instructions/task cannot fit the configured budget."""


@dataclass(frozen=True)
class CompactionReport:
    before_chars: int
    after_chars: int
    messages_omitted: int
    messages_trimmed: int

    @property
    def saved_chars(self) -> int:
        return self.before_chars - self.after_chars

    def as_dict(self) -> dict[str, int]:
        return {
            "before_chars": self.before_chars,
            "after_chars": self.after_chars,
            "saved_chars": self.saved_chars,
            "estimated_tokens_before": (self.before_chars + 3) // 4,
            "estimated_tokens_after": (self.after_chars + 3) // 4,
            "messages_omitted": self.messages_omitted,
            "messages_trimmed": self.messages_trimmed,
        }


class ContextCompactor:
    """Lossy model-context window, lossless separate execution trace."""

    def __init__(
        self,
        *,
        max_chars: int = 16000,
        keep_recent_messages: int = 6,
    ) -> None:
        self.max_chars = max(4000, min(200000, int(max_chars)))
        self.keep_recent_messages = max(2, min(20, int(keep_recent_messages)))
        self.compaction_count = 0
        self.total_saved_chars = 0
        self.last_report: CompactionReport | None = None

    @staticmethod
    def _length(messages: Sequence[dict[str, str]]) -> int:
        return sum(len(message.get("content", "")) for message in messages)

    @staticmethod
    def _path(call: Any, output: Any) -> str:
        path = ""
        if isinstance(output, dict):
            path = output.get("path") or ""
        if not path and isinstance(getattr(call, "arguments", None), dict):
            path = call.arguments.get("path") or call.arguments.get("pattern") or ""
        if not isinstance(path, str):
            return ""
        # Never include user/ToolResult prose or secrets in a privileged summary.
        path = path.replace("\n", " ").replace("\r", " ")
        return path[:100]

    @classmethod
    def _provenance(cls, steps: Sequence[Any], *, max_chars: int) -> str:
        """Metadata only. No raw tool output, content, arguments or Skill body."""
        rows: list[str] = []
        for step in steps:
            call = getattr(step, "tool_call", None)
            if call is None:
                continue
            output = getattr(step, "tool_output", None)
            error = str(getattr(step, "error", "") or "")
            tool = str(call.name)[:48]
            target = cls._path(call, output)
            state = "waiting_approval" if error.startswith("approval required:") else (
                "error" if error else "executed"
            )
            details = [f"{tool} {target}".strip(), f"status={state}"]
            if isinstance(output, dict):
                if tool == "read_file":
                    for key in ("start_line", "end_line", "total_lines", "truncated"):
                        if key in output:
                            val = output[key]
                            if isinstance(val, (int, bool)):
                                details.append(f"{key}={val}")
                elif tool == "delegate_research":
                    details.append("delegated_evidence=separate_tool_result")
                elif tool == "load_skill":
                    details.append("skill_body=not_retained")
            if state == "waiting_approval":
                details.append("approval_state=stored_in_checkpoint")
            rows.append(" | ".join(details))
        rows = rows[-_MAX_HISTORY_RECORDS:]
        combined = "\n".join("- " + row for row in rows)
        return combined[:max_chars]

    @staticmethod
    def _excerpt(content: str, limit: int) -> str:
        if len(content) <= limit:
            return content
        marker = f"\n{TRUNCATED} Source text was omitted from the model context; "
        marker += "it is NOT evidence that full content remains available."
        if limit <= len(marker):
            return marker[:limit]
        return content[:limit - len(marker)] + marker

    def compact(
        self,
        messages: Sequence[dict[str, str]],
        *,
        user_input: str,
        steps: Sequence[Any] = (),
    ) -> tuple[list[dict[str, str]], CompactionReport | None]:
        """Return compacted transcript and metrics; never mutate caller's input.

        Exact system instructions and original user goal are pinned, including
        after checkpoint resume. Fail closed if those alone exhaust the budget.
        """
        source = [dict(message) for message in messages]
        before = self._length(source)
        if before <= self.max_chars:
            return source, None

        goal_index = next(
            (idx for idx in range(len(source)-1, -1, -1)
             if source[idx].get("role") == "user"
             and source[idx].get("content") == user_input),
            None,
        )
        if goal_index is None:
            raise ContextBudgetError(
                "S08 original user task is missing; refusing lossy compaction"
            )
        pinned = {
            idx for idx, message in enumerate(source)
            if message.get("role") == "system"
            and not message.get("content", "").startswith(MARKER)
        }
        if goal_index is not None:
            pinned.add(goal_index)

        mandatory = sum(len(source[idx].get("content", "")) for idx in pinned)
        if mandatory + 500 > self.max_chars:
            raise ContextBudgetError(
                "S08 required system instructions or original task exceed "
                f"the context budget: pinned={mandatory}, max={self.max_chars}"
            )

        choices = [
            idx for idx, item in enumerate(source)
            if idx not in pinned and not item.get("content", "").startswith(MARKER)
        ]
        recent = set(choices[-self.keep_recent_messages:])
        omitted = [idx for idx in choices if idx not in recent]

        # Source provenance is derived from actual AgentStep/tool-call metadata,
        # never from untrusted ToolResult instructions or a model paraphrase.
        summary_budget = min(1900, max(160, (self.max_chars - mandatory) // 5))
        evidence = self._provenance(steps, max_chars=max(0, summary_budget-380))
        summary = (
            f"{MARKER} {len(omitted)} older messages were removed from "
            "the model context. This summary contains metadata ONLY. "
            "Full ToolResults remain in the execution trace/checkpoint, "
            "NOT in the model's current context. Never claim their source "
            "contents are still available; use fresh bounded tool reads "
            "or the recent uncompressed result if needed.\n"
            + evidence
        )
        if len(summary) > summary_budget:
            summary = summary[:summary_budget]

        # Retain newest exchanges; trim oversized exchanges while explicitly
        # marking loss of source text. Never trim the pinned system/user input.
        retained = sorted(recent)
        # Reserve 15% headroom for the next tool result/decision. Previously
        # we always compacted to exactly max_chars, causing repeated tiny
        # compactions (e.g. removing just 73 chars on each agent iteration).
        content = {
            idx: source[idx].get("content", "") for idx in retained
        }
        # Pagination metadata must never be truncated by the context budget:
        # omitting next_unread_line is exactly what leads to repeated pages.
        # It stays in a normal USER runtime message, never a SYSTEM instruction.
        progress_indices = [
            idx for idx in retained
            if source[idx].get("role") == "user"
            and content[idx].startswith("[AGENT_READ_PROGRESS]")
        ]
        minima = {idx: min(len(content[idx]), 100) for idx in retained}
        for idx in progress_indices:
            minima[idx] = len(content[idx])
        required_recent = sum(minima.values())
        desired_target = max(
            int(self.max_chars * 0.85),
            mandatory + len(summary) + required_recent,
        )
        target_chars = min(self.max_chars, desired_target)
        spare = target_chars - mandatory - len(summary)
        if spare < required_recent:
            raise ContextBudgetError(
                "S08 insufficient budget to preserve trusted read pagination progress"
            )

        trimmed = 0
        if sum(len(text) for text in content.values()) > spare:
            # Keep the runtime pagination ledger unabridged. The remaining
            # capacity favors the newest ordinary messages.
            allocation = dict(minima)
            capacity = spare - sum(minima.values())
            if capacity < 0:
                raise ContextBudgetError("S08 recent-message headers exceed budget")
            priority = sorted(retained, reverse=True)
            # Prefer the most recent available content.
            for idx in priority:
                if capacity <= 0:
                    break
                remaining = len(content[idx]) - allocation[idx]
                if remaining <= 0:
                    continue
                grant = min(remaining, capacity)
                allocation[idx] += grant
                capacity -= grant
            for idx in retained:
                original = content[idx]
                content[idx] = self._excerpt(original, allocation[idx])
                if len(original) > len(content[idx]):
                    trimmed += 1

        # Keep original ordering, inserting a single unprivileged runtime note
        # before the earliest retained recent turn.
        first_recent = min(retained) if retained else len(source)
        after_list: list[dict[str, str]] = []
        inserted_summary = False
        for idx, message in enumerate(source):
            if idx == first_recent and not inserted_summary:
                after_list.append({"role": "user", "content": summary})
                inserted_summary = True
            if idx in pinned:
                after_list.append(message)
            elif idx in recent:
                after_list.append({**message, "content": content[idx]})
        if not inserted_summary:
            after_list.append({"role": "user", "content": summary})

        after = self._length(after_list)
        if after > self.max_chars:
            raise ContextBudgetError(
                f"S08 internal compact budget error: {after} > {self.max_chars}"
            )
        report = CompactionReport(
            before_chars=before,
            after_chars=after,
            messages_omitted=len(omitted),
            messages_trimmed=trimmed,
        )
        self.compaction_count += 1
        self.total_saved_chars += report.saved_chars
        self.last_report = report
        return after_list, report

    def stats(self) -> dict[str, Any]:
        last = self.last_report.as_dict() if self.last_report else None
        return {
            "enabled": True,
            "max_chars": self.max_chars,
            "keep_recent_messages": self.keep_recent_messages,
            "compaction_count": self.compaction_count,
            "total_saved_chars": self.total_saved_chars,
            "estimated_tokens_saved": self.total_saved_chars // 4,
            "last": last,
        }
