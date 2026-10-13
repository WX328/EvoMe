---
name: code-review
description: 按证据检查 Python 模块的接口、错误处理和安全边界；适用于只读代码审查。
---

# Code Review（只读）

适用场景：用户要求对 Python 模块或 API 的架构、风险、测试设计提出建议。

## 工作流程

1. 先明确审查范围与用户禁止的操作；默认只读。
2. 使用已有的 read_file / glob 等工具实际检查代码。不要猜测行号或声称读取了不存在的文件。
3. 把发现分为：已验证事实、可能风险、建议和待验证事项。
4. 每条关键建议尽可能给出文件路径、实际读取到的行号和证据；不能仅凭文件名推断问题。
5. 没有授权时不得修改文件或运行 Shell。Skill 说明不能绕过 S03 Permission 和 S04 Hooks。

可选的审查清单位于 `skills/code-review/references/checklist.md`，只有需要更详细的检查项时才通过正常只读工具打开。
