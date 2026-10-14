from __future__ import annotations


# Central command vocabulary used by both the CLI and Learning Eligibility Gate.
# Keeping this in one place prevents an operational command typo from becoming a
# learning sample simply because the slash was omitted.
PARAM_COMMAND_NAMES = frozenset({
    "candidate-approve",
    "candidate-reject",
    "candidate-rebase",
    "proposal-normalize",
    "response-policy",
    "revision-rollback",
    "learning-invalidate",
    "tool-approve",
    "tool-deny",
    "plan",
    "plan-show",
    "plan-run",
    "plan-replan",
    "plan-cancel",
    "subagent",
    "skill-load",
    "agent-memory-find",
    "agent-memory-propose",
    "agent-memory-approve",
    "agent-memory-deny",
    "task-create",
    "task-depend",
    "task-show",
    "task-start",
    "task-done",
    "task-release",
    "task-bg-run",
    "task-bg-show",
    "task-bg-wait",
    "task-bg-recover",
    "cron-at", "cron-add", "cron-after",
    "cron-show", "cron-pause", "cron-resume", "cron-cancel",
    "team-propose", "team-approve", "team-reject", "team-resume",
    "team-bind", "team-send", "team-plan-request",
    "team-plan-approve", "team-plan-reject", "team-stop",
    "team-recover",
    "mcp-connect", "mcp-disconnect", "mcp-call",
})

COMMAND_NAMES = frozenset(
    {
        "memory",
        "experience",
        "reflection",
        "hypothesis",
        "revision",
        "revision-signals",
        "revision-pressure",
        "revision-groups",
        "ancestor-revalidate",
        "signal-revalidations",
        "evidence-summary",
        "revision-analyze",
        "revision-analyses",
        "revision-proposals",
        "proposal-normalize",
        "proposal-normalizations",
        "response-policy",
        "candidate-evaluate",
        "candidates",
        "candidate-rebase",
        "candidate-rebases",
        "revision-apply-auto",
        "candidate-approve",
        "candidate-reject",
        "revision-events",
        "revision-rollback",
        "learning-invalidate",
        "learning-invalidations",
        "status",
        "token-usage",
        "audit",
        "skills",
        "skill-docs",
        "skill-load",
        "context-stats",
        "agent-memory-list",
        "agent-memory-find",
        "agent-memory-propose",
        "agent-memory-pending",
        "agent-memory-approve",
        "agent-memory-deny",
        "agent-memory-preview",
        "agent-memory-consolidate",
        "task-list",
        "task-ready",
        "task-show",
        "task-create",
        "task-depend",
        "task-start",
        "task-done",
        "task-release",
        "task-bg-run",
        "task-bg-list",
        "task-bg-show",
        "task-bg-wait",
        "task-bg-recover",
        "cron-at", "cron-add", "cron-after", "cron-list",
        "cron-show", "cron-pause", "cron-resume", "cron-cancel",
        "cron-events", "cron-tick",
        "team-propose", "team-approve", "team-reject", "team-resume",
        "team-status", "team-bind", "team-send", "team-inbox",
        "team-plan-request", "team-plan-approve", "team-plan-reject",
        "team-stop", "team-shutdown", "team-recover",
        "mcp-servers", "mcp-connect", "mcp-disconnect",
        "mcp-tools", "mcp-call",
        "hooks",
        "plans",
        "plan",
        "plan-show",
        "plan-run",
        "plan-replan",
        "plan-cancel",
        "subagent",
        "tool-approvals",
        "tool-approve",
        "tool-deny",
        "exit",
    }
)


def normalize_cli_input(text: str) -> str:
    """Accept known CLI commands with or without a leading slash.

    No-argument commands are normalized only when the entire input is exactly
    the command name. Parameterized commands may include arguments. This keeps
    natural language such as `hypothesis 是什么意思？` out of the command path.
    """

    stripped = str(text or "").strip()
    if not stripped or stripped.startswith("/"):
        return stripped

    parts = stripped.split(maxsplit=1)
    head = parts[0].lower()
    if head not in COMMAND_NAMES:
        return stripped
    if len(parts) == 1 or head in PARAM_COMMAND_NAMES:
        return "/" + stripped
    return stripped


def looks_like_operational_input(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if stripped.startswith("/"):
        return True
    parts = stripped.split(maxsplit=1)
    head = parts[0].lower()
    if head not in COMMAND_NAMES:
        return False
    return len(parts) == 1 or head in PARAM_COMMAND_NAMES
