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
