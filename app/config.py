import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
EVOME_DB_URL = os.getenv("EVOME_DB_URL", "sqlite:///./evome.db")

# S1 Agent Loop. Disabled by default for migration safety; enable with
# EVOME_AGENT_LOOP_ENABLED=1 after applying the patch.
EVOME_AGENT_LOOP_ENABLED = os.getenv(
    "EVOME_AGENT_LOOP_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_AGENT_MAX_STEPS = max(1, int(os.getenv("EVOME_AGENT_MAX_STEPS", "6")))

# S2 Tool Use. File tools are always workspace-scoped. Bash stays off until
# explicitly enabled; S3 adds approval/permission policy.
EVOME_TOOL_WORKSPACE = os.getenv("EVOME_TOOL_WORKSPACE", ".")
EVOME_BASH_TOOL_ENABLED = os.getenv(
    "EVOME_BASH_TOOL_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_BASH_TIMEOUT_SECONDS = max(1, int(os.getenv("EVOME_BASH_TIMEOUT_SECONDS", "20")))
EVOME_TOOL_MAX_OUTPUT_CHARS = max(1000, int(os.getenv("EVOME_TOOL_MAX_OUTPUT_CHARS", "12000")))
EVOME_BASH_PERMISSION = os.getenv("EVOME_BASH_PERMISSION", "ask").strip().lower()

# S04 Hooks. These intercept lifecycle events without bypassing S03 approvals.
EVOME_HOOKS_ENABLED = os.getenv(
    "EVOME_HOOKS_ENABLED", "1"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_HOOK_HISTORY_LIMIT = max(1, int(os.getenv("EVOME_HOOK_HISTORY_LIMIT", "100")))
EVOME_HOOK_PROTECT_SECRETS = os.getenv(
    "EVOME_HOOK_PROTECT_SECRETS", "1"
).strip().lower() in {"1", "true", "yes", "on"}

# S05 Todo/Planning is opt-in to preserve the S01-S04 conversation behavior.
EVOME_PLANNING_ENABLED = os.getenv(
    "EVOME_PLANNING_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_PLAN_MAX_TASKS = max(1, min(10, int(os.getenv("EVOME_PLAN_MAX_TASKS", "5"))))
EVOME_PLAN_STEP_MAX_STEPS = max(1, min(12, int(os.getenv("EVOME_PLAN_STEP_MAX_STEPS", "4"))))
EVOME_PLAN_STEP_MAX_CONTINUATIONS = max(0, min(3, int(os.getenv("EVOME_PLAN_STEP_MAX_CONTINUATIONS", "2"))))
EVOME_PLAN_MAX_REPLANS = max(0, min(3, int(os.getenv("EVOME_PLAN_MAX_REPLANS", "1"))))
EVOME_PLAN_AUTO_REPLAN = os.getenv(
    "EVOME_PLAN_AUTO_REPLAN", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# Require actual file-tool evidence before marking code-reading plan steps done.
EVOME_PLAN_REQUIRE_FILE_EVIDENCE = os.getenv(
    "EVOME_PLAN_REQUIRE_FILE_EVIDENCE", "1"
).strip().lower() in {"1", "true", "yes", "on"}

# S06 Research Subagent: independent read-only child Agent with bounded steps.
EVOME_SUBAGENT_ENABLED = os.getenv(
    "EVOME_SUBAGENT_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_SUBAGENT_MAX_STEPS = max(1, min(10, int(os.getenv("EVOME_SUBAGENT_MAX_STEPS", "4"))))
EVOME_SUBAGENT_MAX_CALLS = max(1, min(5, int(os.getenv("EVOME_SUBAGENT_MAX_CALLS", "2"))))

# S07 on-demand SKILL.md instructions, separate from executable SkillRegistry.
EVOME_SKILL_LOADING_ENABLED = os.getenv(
    "EVOME_SKILL_LOADING_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_SKILL_DIRECTORY = os.getenv("EVOME_SKILL_DIRECTORY", "skills")
EVOME_SKILL_MAX_CONTENT_CHARS = max(
    1000, min(32000, int(os.getenv("EVOME_SKILL_MAX_CONTENT_CHARS", "16000")))
)

# S08 Context Compact. Opt-in; bounded model-facing history only.
EVOME_CONTEXT_COMPACT_ENABLED = os.getenv(
    "EVOME_CONTEXT_COMPACT_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_CONTEXT_MAX_CHARS = max(
    4000, min(200000, int(os.getenv("EVOME_CONTEXT_MAX_CHARS", "16000")))
)
EVOME_CONTEXT_KEEP_RECENT_MESSAGES = max(
    2, min(20, int(os.getenv("EVOME_CONTEXT_KEEP_RECENT_MESSAGES", "6")))
)

# S09 Agent working memory is separate from existing SQLite Personal Memory.
# Persistent writes require explicit candidate approval; auto-stage never saves.
EVOME_AGENT_MEMORY_ENABLED = os.getenv(
    "EVOME_AGENT_MEMORY_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_AGENT_MEMORY_DIR = os.getenv("EVOME_AGENT_MEMORY_DIR", ".memory")
EVOME_AGENT_MEMORY_AUTO_STAGE = os.getenv(
    "EVOME_AGENT_MEMORY_AUTO_STAGE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_AGENT_MEMORY_RECALL_CHARS = max(
    400, min(6000, int(os.getenv("EVOME_AGENT_MEMORY_RECALL_CHARS", "1800")))
)

# S10 durable Task System, separate from S05 session plans and S11 workers.
EVOME_TASK_SYSTEM_ENABLED = os.getenv(
    "EVOME_TASK_SYSTEM_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_TASK_DIRECTORY = os.getenv("EVOME_TASK_DIRECTORY", ".tasks")

# S11 background read-only research tasks (opt-in, single worker).
# Requires S10 TaskStore and EVOME_AGENT_LOOP_ENABLED.
EVOME_S11_BACKGROUND_ENABLED = os.getenv(
    "EVOME_S11_BACKGROUND_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_S11_MAX_STEPS = max(
    2, min(10, int(os.getenv("EVOME_S11_MAX_STEPS", "6")))
)

# S12 opt-in Cron Scheduler: persisted triggers require S10 + S11.
EVOME_S12_CRON_ENABLED = os.getenv(
    "EVOME_S12_CRON_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_S12_CRON_DIR = os.getenv("EVOME_S12_CRON_DIR", ".schedules")
EVOME_S12_POLL_SECONDS = max(
    0.2, min(60.0, float(os.getenv("EVOME_S12_POLL_SECONDS", "1.0")))
)

# S13 team runtime is off by default. Only the interactive CLI opts in.
EVOME_S13_TEAMS_ENABLED = os.getenv(
    "EVOME_S13_TEAMS_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_S13_TEAM_DIR = os.getenv("EVOME_S13_TEAM_DIR", ".team")
EVOME_S13_MAX_STEPS = max(
    2, min(10, int(os.getenv("EVOME_S13_MAX_STEPS", "5")))
)

# S14 MCP Plugin. Explicit operator-connect only; no startup process spawn.
EVOME_S14_MCP_ENABLED = os.getenv(
    "EVOME_S14_MCP_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
EVOME_S14_MCP_CONFIG = os.getenv("EVOME_S14_MCP_CONFIG", ".mcp_servers.json")
EVOME_S14_MCP_TIMEOUT = max(
    0.25, min(20.0, float(os.getenv("EVOME_S14_MCP_TIMEOUT", "6.0")))
)

# Phase 5.7A: hard output caps for each LLM workload. These protect JSON
# helpers from accidentally generating long responses while leaving the
# user-facing stream comparatively generous.
EVOME_MAX_TOKENS_CHAT = int(os.getenv("EVOME_MAX_TOKENS_CHAT", "4096"))
# Agent JSON includes the user-facing answer, unlike internal JSON helpers.
EVOME_MAX_TOKENS_AGENT = int(os.getenv("EVOME_MAX_TOKENS_AGENT", "4096"))
EVOME_MAX_TOKENS_JSON_DEFAULT = int(os.getenv("EVOME_MAX_TOKENS_JSON_DEFAULT", "768"))
EVOME_MAX_TOKENS_EVALUATION = int(os.getenv("EVOME_MAX_TOKENS_EVALUATION", "512"))
EVOME_MAX_TOKENS_GATE = int(os.getenv("EVOME_MAX_TOKENS_GATE", "384"))
EVOME_MAX_TOKENS_CONTEXT = int(os.getenv("EVOME_MAX_TOKENS_CONTEXT", "768"))
EVOME_MAX_TOKENS_REFLECTION = int(os.getenv("EVOME_MAX_TOKENS_REFLECTION", "1024"))
EVOME_MAX_TOKENS_POST_TURN = int(os.getenv("EVOME_MAX_TOKENS_POST_TURN", "1200"))
EVOME_MAX_TOKENS_RELATION = int(os.getenv("EVOME_MAX_TOKENS_RELATION", "1024"))
EVOME_MAX_TOKENS_REVISION = int(os.getenv("EVOME_MAX_TOKENS_REVISION", "1200"))

# Phase 5.7C: bound how much of an assistant response is embedded into internal
# JSON-helper prompts. The post-turn assessor and the experience gate only need
# enough of the response to judge intent and outcome.
EVOME_PROMPT_RESPONSE_CHAR_LIMIT = int(
    os.getenv("EVOME_PROMPT_RESPONSE_CHAR_LIMIT", "800")
)

# Phase 5.7D: per-memory character cap for memories embedded into internal
# prompts. Without this, recalled memories are embedded unbounded.
EVOME_PROMPT_MEMORY_CHAR_LIMIT = int(
    os.getenv("EVOME_PROMPT_MEMORY_CHAR_LIMIT", "200")
)

# Phase 5.7C: slow-loop cadence. The slow evolution path is submitted on every
# user turn by default. Setting this above 1 runs it only every Nth turn,
# trading learning frequency for token cost. 1 = current behavior.
EVOME_SLOW_LOOP_EVERY_N_TURNS = int(os.getenv("EVOME_SLOW_LOOP_EVERY_N_TURNS", "1"))

EVOME_QDRANT_PATH = os.getenv("EVOME_QDRANT_PATH", "./evome_qdrant")
HYPOTHESIS_TOP_K = int(os.getenv("HYPOTHESIS_TOP_K", "5"))
HYPOTHESIS_SCORE_THRESHOLD = float(os.getenv("HYPOTHESIS_SCORE_THRESHOLD", "0.60"))
HYPOTHESIS_RELATION_CONFIDENCE = float(os.getenv("HYPOTHESIS_RELATION_CONFIDENCE", "0.60"))

REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS = int(os.getenv("REVISION_ANALYSIS_MIN_INDEPENDENT_SIGNALS", "2"))

REVISION_ANALYSIS_MAX_PER_RUN = int(os.getenv("REVISION_ANALYSIS_MAX_PER_RUN", "2"))
REVISION_CANDIDATE_MAX_PER_RUN = int(os.getenv("REVISION_CANDIDATE_MAX_PER_RUN", "2"))
REVISION_CANDIDATE_MAX_EVIDENCE = int(os.getenv("REVISION_CANDIDATE_MAX_EVIDENCE", "8"))
REVISION_AUTO_APPLY_MIN_UTILITY = float(os.getenv("REVISION_AUTO_APPLY_MIN_UTILITY", "0.20"))
REVISION_AUTO_APPLY_BUDGET_HOURS = int(os.getenv("REVISION_AUTO_APPLY_BUDGET_HOURS", "24"))
REVISION_AUTO_APPLY_PER_HYPOTHESIS = int(os.getenv("REVISION_AUTO_APPLY_PER_HYPOTHESIS", "1"))
REVISION_AUTO_APPLY_MAX_PER_RUN = int(os.getenv("REVISION_AUTO_APPLY_MAX_PER_RUN", "2"))

REVISION_REVALIDATION_MAX_SIGNALS_PER_BATCH = int(os.getenv("REVISION_REVALIDATION_MAX_SIGNALS_PER_BATCH", "16"))
REVISION_REVALIDATION_MAX_HYPOTHESES_PER_RUN = int(os.getenv("REVISION_REVALIDATION_MAX_HYPOTHESES_PER_RUN", "2"))
