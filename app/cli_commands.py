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
        "audit",
        "skills",
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
