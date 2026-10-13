# EvoMe

> An evidence-driven Personal AI that learns from interaction through traceable memory, hypotheses, and revisions.

EvoMe 是一个实验性的 Personal AI 核心系统。它探索 AI 如何从长期交互中形成关于用户的记忆与关于自身行为的经验，同时避免因为一次模型判断就随意改变策略。

当前开发版本：`v0.3.0-dev`（S1–S4 Agent Runtime）。项目仍处于研究原型阶段，不建议直接用于生产环境。

## 核心设计

EvoMe 将不同用途的信息分开保存：

| 数据 | 含义 |
| --- | --- |
| Personal Memory | 关于用户目标、偏好、项目、决定与长期背景的信息 |
| Interaction Record | 用户输入与 AI 回复的原始审计记录 |
| Experience | EvoMe 某次任务执行的过程、环境和结果 |
| Reflection | 从单次 Experience 中提取的局部认识 |
| Strategy Hypothesis | 多次 Reflection 聚合形成的策略假设 |
| Hypothesis Revision | 策略假设某个版本的不可变快照 |
| Revision Signal | 新证据对当前 Revision 提出的局部修改信号 |
| Candidate Revision | 聚合、分析并验证后的候选修订 |
| Revision Event | 修订采用、拒绝或回滚的审计事件 |

## 工作流

```mermaid
flowchart TD
    A[User Input] --> B[Fast Path]
    B --> C[Retrieve Personal Memory]
    C --> D[Stream Response]
    D --> E[Audit Interaction]
    E --> F[Background Slow Path]
    F --> G[Learning Eligibility Gate]
    G -->|Personal| H[Personal Memory Gate]
    G -->|Evolution| I[Experience Gate]
    I --> J[Context Builder]
    J --> K[Experience]
    K --> L[Reflection]
    L --> M[Hypothesis Aggregator]
    M -->|SUPPORTS| N[Supporting Evidence]
    M -->|REFINES| O[Revision Signals]
    M -->|CONFLICTS| P[Counter Evidence]
    O --> Q[Revision Aggregation]
    Q --> R[Revision Analysis]
    R --> S[Candidate Revision]
    S --> T[Approve / Reject / Controlled Auto-apply]
    T --> U[New Immutable Revision]
```

用户可见回答走 Fast Path；记忆提取、经验反思和假设演化走后台 Slow Path。一次普通用户交互最多形成 Experience、Reflection、Hypothesis Evidence 与 Revision Signal，不会顺带清空全局修订积压。较重的 Revision 分析、候选验证和采用通过显式后台任务执行。

## 关键边界

- 当前用户明确输入优先于历史 Personal Memory。
- Interaction 审计范围较宽，Learning 写入范围更严格。
- 命令、噪声、模拟测试和纯状态查询默认不会污染长期学习数据。
- Reflection 是局部认识，不等于已验证规律。
- `SUPPORTS`、`REFINES`、`CONFLICTS` 和 `UNRELATED` 分开处理。
- Revision Signal 绑定具体 Hypothesis 与 Base Revision。
- Hypothesis Revision 是不可变快照；新认知通过新 Revision 表达。
- Candidate Revision 需要经过规范化和验证；支持人工批准、拒绝、受限自动采用与回滚。
- 每次采用、拒绝、回滚与学习失效都有可追溯记录。

## 当前能力

- Personal Memory 与 Evolution Memory 分层
- 原始交互审计与学习失效处理
- Learning Eligibility Gate
- Experience Gate 与结构化 Context Builder
- Reflection 与 Strategy Hypothesis 聚合
- Revision-aware Evidence 与不可变 Revision 历史
- Revision Signal 质量检查、聚合与证据摘要
- Revision Proposal、Policy Normalization 与 Candidate Validation
- Candidate rebase、ancestor signal revalidation、批准、拒绝与回滚
- 前台流式回答与后台学习队列
- SQLite 持久化
- Qdrant 本地检索及 SQLite 回退
- 按 LLM 工作负载限制输出 token
- 进程内 Token Usage 统计
- Fake LLM 离线测试，不需要 API Key
- S1–S4 Agent Runtime：有界 Agent Loop、自主工具调用、工作区文件工具、人工审批与生命周期 Hooks

## 尚在实验

- Personal Memory 的冲突合并、更新与失效策略
- 基于真实用户反馈和工具结果的 Outcome grounding
- Strategy Signal 的完整原子化
- Applicability Context 的长期验证
- Candidate 自动采用策略的真实环境评估
- 多轮短期对话上下文
- Web/API 界面与多用户隔离

## 技术栈

- Python 3.10+
- SQLAlchemy + SQLite
- LangGraph
- DeepSeek API（OpenAI-compatible client）
- Qdrant local mode
- pytest

## 安装

```powershell
git clone <your-repository-url>
cd EvoMe

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写：

```env
DEEPSEEK_API_KEY=your_api_key
```

完整的可选配置和默认值见 [`.env.example`](.env.example)。

## 运行

```powershell
python main.py
```

常用命令：

| 命令 | 用途 |
| --- | --- |
| `/memory` | 查看 Personal Memory |
| `/experience` | 查看 Experience |
| `/reflection` | 查看 Reflection |
| `/hypothesis` | 查看 Strategy Hypothesis |
| `/revision` | 查看 Revision 历史 |
| `/revision-signals` | 查看有效 Revision Signal |
| `/revision-pressure` | 查看某个假设的修订压力 |
| `/revision-groups` | 查看 Revision Signal 聚合结果 |
| `/evidence-summary` | 查看证据摘要 |
| `/revision-analyze` | 提交 Revision Analysis 后台任务 |
| `/revision-proposals` | 查看 Revision Proposal |
| `/candidate-evaluate` | 提交 Candidate Validation 后台任务 |
| `/candidates` | 查看 Candidate Revision |
| `/revision-apply-auto` | 显式运行受限自动采用任务 |
| `/candidate-approve <id>` | 人工批准候选 Revision |
| `/candidate-reject <id>` | 人工拒绝候选 Revision |
| `/revision-rollback <hypothesis_id> <revision>` | 回滚到历史 Revision |
| `/learning-invalidate <interaction_id> [reason]` | 使某次交互产生的学习失效 |
| `/status` | 查看后台任务 |
| `/token-usage` | 查看当前进程的 Token 使用量 |
| `/audit` | 查看原始交互审计 |
| `/skills` | 查看 Skill Registry |
| `/exit` | 等待已提交任务结束并安全退出 |

CLI 允许省略已知命令前面的 `/`。未知命令会被拦截，不会发送给 LLM，也不会进入学习系统。

## 测试

测试默认使用内存数据库和 Fake LLM，不调用外部模型接口：

```powershell
python -m pytest -q
```

## 项目结构

```text
app/
├── audit/          # 原始交互审计
├── context/        # Context 构建与分区
├── evaluation/     # 回答评估
├── evolution/      # Experience 与 Reflection 持久化
├── gates/          # Learning / Memory / Experience Gate
├── hypothesis/     # Hypothesis、Revision、Candidate 与演化流程
├── learning/       # 学习失效与级联处理
├── memory/         # Personal Memory
├── reflection/     # Reflection 提取
├── skills/         # Skill 定义与注册
├── strategy/       # LangGraph 工作流
├── telemetry/      # Token Usage 统计
└── vector/         # 本地向量表示与检索支持
```

## 版本历史

- `v0.1`：建立最初的 Memory、Reflection 和 Strategy Workflow 骨架。
- `v0.1.1`：拆分 Personal Memory、Interaction、Experience 与 Reflection。
- `v0.2`：加入结构化 Context 与 Strategy Hypothesis 聚合。
- `v0.2.1-dev`：加入完整的 Revision 演化管线、后台学习边界与 LLM 成本控制。
- `v0.3.0-dev`：加入 S1 Agent Loop、S2 Tool Use、S3 Permission/Approval 与 S04 Hooks 生命周期钩子（开发版）。

## 设计理念

```text
Observation
→ Reflection
→ Evidence
→ Hypothesis
→ Revision Signal
→ Candidate Revision
→ Validation
→ Adoption or Rejection
→ Rollback when needed
```

EvoMe 的目标不是让模型频繁改写自身，而是建立一条可观察、可验证、可修订、可回滚的认知演化路径。

## License

当前尚未选择开源许可证。正式公开前请根据用途添加合适的 `LICENSE` 文件。

## S04 Hooks

With `EVOME_AGENT_LOOP_ENABLED=1`, the S04 hook manager supports
`before_agent`, `after_agent`, `before_tool`, and `after_tool` lifecycle
events. Register trusted Python handlers via
`runtime.hook_manager.register(event, handler, name="...", priority=10)`.
A pre-hook may return `HookDecision(False, "reason")` to block execution.
Exceptions in pre-hooks fail closed; exceptions in post-hooks do not change the
Tool result. S03 PermissionPolicy remains authoritative: a hook cannot approve
a pending write or shell command.

The default `before_tool` guard blocks direct file-tool access to `.env`,
private keys, `.git`, `.venv`, the local database and the Qdrant storage
directory (including paths inside the configured workspace).
For operator visibility, enter `/hooks` in EvoMe CLI. Only event names,
outcomes and minimal metadata are retained in a bounded in-memory history;
no arguments, command text, file contents or prompts are logged there.
Hooks are local trusted Python callbacks, not executable user prompts. The
file guard does *not* sandbox arbitrary shell commands.

## S05 Todo / Planning

设置 `EVOME_AGENT_LOOP_ENABLED=1` 与 `EVOME_PLANNING_ENABLED=1`
可启用基于多步骤 Todo 的 Agent 执行计划。S05 默认关闭以保证 S01-S04 行为不变。
规划器负责生成至多 `EVOME_PLAN_MAX_TASKS` 个任务，逐个使用现有 Agent Loop
及其 ToolExecutor 执行，步骤有独立状态、执行摘要和有限重规划预算。

- `/plan <目标>`：生成计划并立即执行可执行的步骤
- `/plans`：查看计划列表；`/plan-show <id>`：查看每步完整结果和 Todo 状态
- `/plan-run <id>`：继续待处理步骤，或从暂停步骤的断点续跑
- `/plan-replan <id>`：失败或暂停后按剩余预算重新规划
- `/plan-cancel <id>`：取消计划；等待中的 S03 审批同时作废

若写文件或运行 Shell 需要审批，计划进入 `waiting_approval`。
执行 `/tool-approve <id>` 后再用 `/plan-run <计划ID>` 继续；
拒绝审批会令步骤失败，不会将未执行操作记为已完成。
可选 `EVOME_PLAN_AUTO_REPLAN=1`：只对失败过程中使用过只读工具的任务
尝试自动重新规划（最多 `EVOME_PLAN_MAX_REPLANS` 次）。
S03 Permission 和 S04 Hooks 仍是每次真实 Tool 调用的硬边界。

每批的工具/决策轮数上限取 `EVOME_AGENT_MAX_STEPS` 和
`EVOME_PLAN_STEP_MAX_STEPS` 的较小值。最后一次成功的工具调用后，额外保留一次
只能回答、不能执行工具的收尾机会；模型必须明确确认任务完成，否则仍显示未完成。
若当前步骤仍需读取资料，且此前只有成功的只读工具调用，会保留原始工具结果、
消息和重复调用记录，自动续跑最多 `EVOME_PLAN_STEP_MAX_CONTINUATIONS` 批（默认 2，
设为 0 可关闭自动续跑）。每次 `/plan` 或 `/plan-run` 对每步分别执行这一有限预算。
预算用尽后状态为 `paused`，可用 `/plan-run <id>` 继续原步骤，不必重新规划。
有工具错误、需要审批或涉及写入的步骤不会自动续跑；新调用仍经过权限和 Hooks。
重新规划会替换所有旧的未完成步骤，
已完成和被替换的步骤保留为历史记录。

长文件按行分页读取，返回实际 `end_line`、`truncated` 和 `next_start_line`。
单行超过输出上限时会明确标记 `line_truncated`，不会把部分内容当作全文。
`/plan-show <id>` 还会显示工具调用及实际读取范围，便于定位缺少的证据。

每步结果完整保存在内存中；普通列表的省略摘要会提示 `/plan-show <id>`，
计划完成时直接展示最后一步的完整回答。Agent 回答使用独立的
`EVOME_MAX_TOKENS_AGENT`（默认 4096），内部 JSON 辅助任务继续使用原有预算。
模型报告输出长度耗尽时，截断的 Agent JSON 不会作为工具指令或完成结果执行。

Agent 的决策 JSON 只包含工具参数或 `task_complete`，长篇回答在独立的文本阶段生成，
避免 Markdown、引号和换行破坏控制消息。格式异常只额外重试一次；连续异常显示
`protocol_error`，不会误报步数耗尽，也不会信任自动修复的 JSON。文本回答达到输出
上限时仍显示未完成。独立文本阶段会增加一次模型调用，可用 `/token-usage` 查看用量。

计划执行时，每次只处理当前步骤，其他任务由调度器继续执行。
若目标包含查看执行记录，模型可调用只读 `plan_show` 工具，实际计划 ID 由运行时提供；
无需模型猜测 ID 或执行 CLI/Shell 命令。综合步骤可以使用已完成步骤的文件证据，
但查看计划记录不能代替读取源码。连续没有新工具调用且未确认完成时，会明确提示
没有进展并暂停，避免自动重复。手动 `/plan-show <id>` 始终可查看当前完整记录。

S05 的计划和 Todo 记录仅保存在当前进程内，重启后清空；
不是持久化任务队列，也不会自动写入 Personal Memory。

## S06 Research Subagent

开启 `EVOME_AGENT_LOOP_ENABLED=1` 和 `EVOME_SUBAGENT_ENABLED=1` 后，
主 Agent 增加 `delegate_research` 工具，可委派独立、只读的研究任务。
命令行也支持 `/subagent <任务>`，用于定向手工验收。

- 子 Agent 使用独立的 AgentLoop 和上下文；不传入主 Agent 的对话和 Personal Memory。
- 子 Agent 只拥有 read_file / glob，没有写文件、Shell、创建计划或再次委派的能力。
- 每个子任务最多 EVOME_SUBAGENT_MAX_STEPS 次 Agent 决策，默认 4。
  主 Agent 每次运行最多 EVOME_SUBAGENT_MAX_CALLS 次委派，默认 2。
- 子 Agent 的只读操作仍经过 S03 PermissionPolicy 与 S04 Hooks，
  并额外限制 .env、密钥等敏感文件。
- 只有完成且有实际文件检索证据，才将子任务视作成功；
  返回结构化摘要、证据路径、行范围与工具调用数。
- 本阶段仅同步、单层；不涉及并行 Agent Team、子任务持久化或自动重试。

示例：`/subagent 读取 app/agent/schema.py，解释数据结构，不修改文件`
