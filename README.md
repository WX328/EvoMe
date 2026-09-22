# EvoMe V0.1 Core

第一版只验证一个核心闭环：

`Memory -> Reflection -> Strategy Workflow -> Skill -> Evaluation -> New Memory`

## 1. 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

然后在 `.env` 中填入 DeepSeek API Key。

## 2. 先跑不需要 API 的测试

```powershell
pytest -q
```

## 3. 运行

```powershell
python main.py
```

命令：

- `/memory`：查看 Memory
- `/skills`：查看 Skill Registry
- `/exit`：退出

## 4. V0.1 暂时不做

- Qdrant
- Neo4j
- Goal System
- Strategy 自动变异
- Skill 自动生成
- MCP
- 自动修改代码

它们会在当前核心闭环稳定后逐层加入。

## V0.2.1 Phase 5.4.1：Learning Eligibility Boundary

Slow Loop 在写入 PersonalMemory / Experience 之前增加 `LearningEligibilityGate`：

- `operational`：命令/状态操作，只审计，不学习。
- `noise`：乱码/误按，只审计，不学习。
- `testing`：模拟/测试输入默认不学习；显式要求记录测试结果时只允许 Evolution，禁止模拟事实进入 PersonalMemory。
- `meta`：仅观察 EvoMe 当前 Memory/Hypothesis/状态时只审计；真实设计/修复任务仍可进入 Evolution。
- `natural`：再按 `allow_personal_memory` 与 `allow_evolution` 两条独立通道进入后续 Gate。

已知 CLI 命令可省略 `/`，例如 `revision-signals` 会自动规范化为 `/revision-signals`。未知 `/xxx` 会被 CLI 拦截，不发送给 LLM，也不进入学习系统。

原则：**Audit is broad; learning is selective.**
