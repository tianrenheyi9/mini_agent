# AI Prompt 与问题解决记录

本文记录本项目在 AI 辅助下完成的过程：关键 Prompt、设计取舍、踩过的问题、如何验证，以及哪些判断由人完成。

---

## 1. 项目背景

题目要求从零实现一个最小可用 Agent：

- 不能依赖 LangGraph / LangChain / OpenHands 等 Agent 框架；
- 实现“输入 → 决策 → 工具 → 回填 → 继续/结束”的循环；
- 至少 3 个工具（todo 必须）；
- 自定义工具注册、参数 Schema、LLM 输出解析；
- 多 Session 隔离与续聊；
- 上下文管理、追问、基础压缩；
- 异常处理、Trace、测试用例；
- 使用真实 LLM API。

最终选型：

| 项 | 选择 |
| --- | --- |
| 语言/Web | Python 3.11+ / FastAPI |
| LLM | OpenAI 兼容接口（`AsyncOpenAI` + `base_url`） |
| 工具协议 | 自定义 JSON 协议，自研解析器 |
| 存储 | 每 Session 一个 JSON + Trace JSONL |
| 前端 | 原生 HTML/CSS/JS，无构建步骤 |
| 测试 | pytest + FakeLLM + 真实 LLM live 测试 |

---

## 2. AI 协作方式

整个项目按“先设计、再分层实现、每层测试、真实联调”的节奏推进：

1. 先写 README，明确需求拆解、架构、验收标准；
2. 按依赖顺序实现：工具基础层 → 协议/Prompt → LLM 客户端 → Session → Context → Trace → AgentLoop → API → UI；
3. 每一层先写测试，再接下一层；
4. 用真实模型跑 live 测试和浏览器端到端流程；
5. 根据代码评审意见做针对性修复，并补回归测试。

AI 主要负责：

- 需求拆解与方案对比；
- 代码与测试初稿；
- 边界情况分析；
- 根据失败测试定位问题；
- 文档整理。

人主要负责：

- 技术栈、Provider、协议、UI 形态的最终选择；
- 安全边界（密钥、路径穿越、XSS、Trace 隔离）的判断；
- 验收标准与优先级的确认；
- 真实 API 与浏览器流程的最终确认。

---

## 3. 关键 Prompt

### 3.1 System Prompt（`app/agent/prompt.py`）

```text
你是一个可以调用工具的智能助手。
你需要根据用户目标、会话摘要和工具执行结果，决定下一步是调用工具还是直接回复。

## 输出协议（必须严格遵守）
每次回复必须且只能输出一个 JSON 对象，不要输出 JSON 之外的任何文字，也不要用 Markdown 代码块包裹。

调用工具时输出：
{"thought": "一句话说明你为什么调用这个工具", "action": "工具名", "input": {"参数名": "参数值"}}

最终回答时输出：
{"thought": "一句话说明你的判断", "final": "直接展示给用户的完整回答"}

规则：
1. `action` 和 `final` 必须二选一，不能同时出现。
2. `thought` 可选，但建议填写，用来记录当前判断。
3. `input` 必须是 JSON 对象，字段名和类型必须符合工具 Schema。
4. 只能使用下面列出的工具，不要编造工具名或参数。
5. 工具返回错误时，先根据错误信息修正参数；无法修正时向用户说明。
6. 拿到足够信息后立即输出 final，不要为了调用工具而调用工具。
7. final 应该是完整、可直接展示给用户的自然语言回答。

## 可用工具
{dynamic tool catalog}

## 会话摘要
{session.summary}

## 当前待办
{rendered todos}

## 当前时间
{ISO time with timezone}
```

设计意图：

- 协议放在最前面，规则编号清晰；
- 工具目录包含名称、描述、完整 JSON Schema；
- 摘要与待办放在 System Prompt，避免污染消息配对；
- 当前时间让“今天/本周”等相对时间可解析。

### 3.2 协议纠错 Prompt（`build_repair_message`）

```text
你的上一条输出不符合约定协议。
解析错误：{reason}
你刚才的输出（已截断）：
{raw output, max 1200 chars}

请重新输出，并且只输出一个合法的 JSON 对象：
- 调用工具：{"thought": "...", "action": "工具名", "input": {...}}
- 最终回答：{"thought": "...", "final": "..."}
```

设计意图：

- 让模型看到自己具体错在哪里；
- 原始输出截断，避免纠错消息再次撑大上下文；
- 最多重试 `MAX_PARSE_REPAIRS` 次。

### 3.3 摘要 Prompt（`ContextManager._summary_prompt`）

```text
你是一个对话摘要器。请把下面的对话压缩成不超过 {SUMMARY_MAX_CHARS} 字的中文摘要。

必须保留：
1. 用户目标与偏好
2. 已确认的关键事实、数字、日期
3. 已执行的工具操作及其结论
4. 当前待办与未决问题

不要编造信息；如果旧摘要和新对话冲突，以新对话为准。
只输出摘要正文，不要输出标题或解释。

【旧摘要】
{old_summary or "（无）"}

【需要合并的新对话】
{formatted messages}
```

设计意图：

- 摘要不是对话，显式 `json_mode=False`，避免全局 JSON Mode 干扰；
- 只压缩游标之后、最近窗口之前的新消息；
- 失败时降级为“旧摘要 + 最后两条可见消息”。

### 3.4 真实 Agent 联调 Prompt

`tests/agent/test_live_agent.py`：

```text
请先用 calculator 计算 2 的 10 次方，
再用 todo 添加一条待办，content 为“2 的 10 次方是 1024”，
最后告诉我结果。
```

断言：

- 至少实际执行 calculator 和 todo；
- 最终回复包含 `1024`；
- todo 已落盘。

---

## 4. 关键设计决策

### 4.1 自定义 JSON 协议而不是原生 Function Calling

原因：

- 题目要求“实现 LLM 输出的解析逻辑”；
- 不依赖特定厂商的 tools 能力；
- `thought` 字段可观测；
- 解析器可以独立测试。

代价：

- 模型可能不遵守协议，需要纠错；
- 不能直接使用 `role="tool"`。

### 4.2 工具结果回填格式

没有使用原生 `tool_calls`，所以绝不发送缺少 `tool_call_id` 的 `role="tool"`。

实际消息：

```text
assistant: {"action":"calculator","input":{"expression":"2**10"}}
user:      [工具执行结果]
           tool: calculator
           status: ok
           input: {"expression":"2**10"}
           result: {"result":1024}

请继续：需要更多信息就调用工具，否则输出 final。
```

好处：

- 不伪造 `tool_call_id`；
- 模型能看到自己的决策和工具返回；
- 与自定义协议一致。

### 4.3 预算拆分

| 预算 | 含义 |
| --- | --- |
| `MAX_TOOL_CALLS` | 单轮实际执行工具的次数 |
| `MAX_LLM_CALLS_PER_TURN` | 单轮 LLM 调用总次数（final、纠错、额度提示都算） |
| `MAX_PARSE_REPAIRS` | 协议纠错次数 |
| `MAX_COMPRESSION_CALLS_PER_TURN` | 压缩调用次数 |

工具次数用完后仍保留一次 final 机会；模型无视提示继续要工具才兜底。

### 4.4 上下文

- 历史只保留最近 K 条 `user` / `final`；
- 历史 thought 与工具结果不重复进入后续上下文；
- 待办以实时快照进入 System Prompt；
- token 用字符加权估算 × 安全系数（默认 1.2），不引入 tokenizer；
- 硬截断按完整轮次删除；
- 当前轮工具结果超长时分级压缩，保留 `tool/status` 头部，最新结果尽量完整；
- 只有 System + 当前用户消息仍超预算，才返回 `context_overflow`。

### 4.5 Trace 与统计

- 每 Session 一个 JSONL；
- 事件：`turn_start` / `llm_call` / `tool_call` / `parse_error` / `compress` / `turn_end`；
- 单字段截断；
- Trace 失败只记日志，不影响业务；
- LLM / 工具 / 压缩次数与 token 分开统计。

---

## 5. 问题与解决记录

### 5.1 自定义协议与 OpenAI 消息格式

**问题**：自定义 `action/final` 不是原生 `tool_calls`，直接发 `role="tool"` 会缺少 `tool_call_id`。

**解决**：

- 保留 assistant 原始协议输出；
- 工具结果以 user 角色 + `[工具执行结果]` 标记回填；
- 增加测试断言 `role` 列表中没有 `tool`，并检查 observation 内容。

### 5.2 压缩重复摘要

**问题**：只按“最近 K 条”压缩，第二次压缩会把已经摘要过的消息再次送进摘要器。

**解决**：

- Session 增加单调 `msg_id` 与 `summary_until_msg_id`；
- 压缩只处理游标之后、最近窗口之前的新消息；
- 游标只前进；
- 增加 `test_second_compression_does_not_repeat_cursor`。

### 5.3 `MAX_TOOL_STEPS` 语义错误

**问题**：原实现限制的是循环迭代次数，工具次数用完后模型没有机会汇总。

**解决**：

- 重命名为 `MAX_TOOL_CALLS`，只限制工具执行；
- `MAX_LLM_CALLS_PER_TURN` 限制总调用；
- 工具次数用完后注入“请直接输出 final”的提示，保留一次汇总机会；
- 增加 `test_tool_limit_keeps_final_chance` 与 `test_tool_limit_ignored_finalizes`。

### 5.4 上下文预算边界

**问题**：只按单条消息硬截断会破坏 User/Final 配对；System + 当前轮超预算时无法处理。

**解决**：

- 按完整轮次删除历史；
- 新增 `ContextBudgetExceeded`；
- AgentLoop 捕获后返回 `context_overflow`，不发送注定超限的请求；
- 增加轮次级裁剪与超限测试。

### 5.5 当前轮工具结果导致“输入过长”

**问题**：多个工具 observation 累积过长时，用户输入很短也会收到“输入过长”，责任被错误归到用户。

**解决**：

- 新增 `TOOL_OBSERVATION_MAX_CHARS` 与 `OBSERVATION_PROMPT_CHARS`；
- 从最旧 observation 开始，依次尝试 600/300/120/40 字符；
- 保留 `tool/status` 头部，最新结果尽量完整；
- 仍放不下才省略较早结果；
- 只有 System + 当前用户消息本身超限才返回 `context_overflow`；
- 增加 Context 层与完整 Loop 两层测试。

### 5.6 Calculator 嵌套乘方与 CPU 阻塞

**问题**：先计算再检查 `bit_length()`，嵌套乘方可能先分配巨大整数；`async` handler 没有让出事件循环。

**解决**：

- 执行 `**` 前按预估位宽检查；
- 限制 AST 节点数与递归深度；
- 用 `asyncio.to_thread` 执行计算；
- 增加 `test_nested_power_rejected_before_computation`。

### 5.7 Trace 写盘失败影响业务

**问题**：Trace 是辅助数据，但写盘异常会向上抛出。

**解决**：

- `_write` / `read_events` 捕获 `OSError`，只记日志；
- 增加“把 Trace 文件路径做成目录”的写失败测试。

### 5.8 LLM 客户端生命周期

**问题**：每次 `chat()` 都创建新 `AsyncOpenAI`，没有关闭。

**解决**：

- 懒加载并复用客户端；
- `aclose()` 关闭内部客户端；
- FastAPI lifespan 在 shutdown 调用 `agent.aclose()`；
- 增加“两次 chat 只创建一个客户端”的测试。

### 5.9 JSON Mode 与摘要冲突

**问题**：全局 `LLM_JSON_MODE=true` 时，摘要请求也会带 `response_format`，但摘要 Prompt 要的是普通文本。

**解决**：

- 摘要调用显式 `json_mode=False`；
- 测试断言 FakeLLM 收到的 `json_mode` 为 `False`。

### 5.10 Secret hygiene

**问题**：开发过程中真实 API Key 一度被写入被 Git 跟踪的 `.env.example`。

**解决**：

- 推送前扫描发现；
- 将真实 Key 只保留在 `.env`（gitignored）；
- `.env.example` 恢复占位符；
- 确认密钥未进入 Git 历史；
- 之后每次提交前检查暂存区是否包含密钥。

### 5.11 前端 Markdown 与 XSS

**问题**：模型回复含 `**1024**` 等 Markdown；若直接 `innerHTML` 渲染会有 XSS 风险。

**解决**：

- 实现安全的轻量渲染：段落、列表、粗体、行内代码；
- 全部通过 DOM API + `textContent` 构建；
- 不解析链接和原始 HTML；
- 在浏览器中验证渲染结果。

### 5.12 P0：Runtime 未接入 API

**问题**：AgentLoop 已完成，但服务仍只有 health/占位页。

**解决**：

- `create_app()` + lifespan 装配 AgentLoop；
- 实现 Session CRUD、messages、traces；
- 测试通过注入 FakeLLM 的 AgentLoop 覆盖完整 API；
- 浏览器端到端验证。

### 5.13 真实模型联调

**问题**：FakeLLM 只能证明代码路径，不能证明真实模型会遵守协议。

**解决**：

- 新增 `tests/agent/test_live_agent.py`；
- 使用真实 `qwen3.8-max` 跑完整 Loop；
- 断言 calculator 和 todo 实际执行、最终回复包含 1024、todo 落盘。

### 5.14 前端全局发送锁导致无法并行对话

**问题**：`state.sending` 是全局布尔值。A 会话正在执行时切到 B，B 的发送按钮仍显示“思考中”并被禁用，看起来像整个服务被锁住。

**解决**：

- 改为 `state.pending: Map<sessionId, {text}>`，发送状态按 Session 独立维护；
- 发送按钮、typing 气泡、侧栏“思考中…”都按当前 Session 判断；
- 切回仍在执行的 Session 时，会恢复 pending 用户消息和 typing；
- 后端新增 `test_two_sessions_run_concurrently`，用 SlowLLM 证明不同 Session 的 LLM 调用确实并行；
- 浏览器实测：A 思考中时 B 可正常发送并返回，A 完成后落盘，侧栏状态正确；
- 补充「↗ 新窗口」入口：创建新 Session 并在新窗口打开，避免用户复制同一个 `#/s/<id>` 导致两个窗口共享会话；
- 同一 Session 的两个窗口仍按设计串行（同一会话不应并发写）；两个独立 Session 已用真实模型验证可并发（两个请求 5.4 秒内一起完成）；
- 静态资源加版本查询（`?v=`），避免浏览器继续使用旧 JS。

---

## 6. 测试与验收

### 6.1 自动化

```bash
pytest -q
# 200 passed, 2 deselected

pytest -m live -q
# 2 passed
```

覆盖：

- Parser / Prompt；
- 工具 Schema、注册表、4 个工具；
- LLM 客户端（Fake SDK、异常映射、复用/关闭）；
- Session 模型、原子存储、隔离、锁；
- Context 估算、压缩游标、轮次裁剪、observation 压缩；
- AgentLoop 直接回复、单/多工具、未知工具、纠错、预算、LLM 故障、统计；
- Trace 写入/读取/故障隔离；
- FastAPI 全部路由；
- 真实 LLM Agent 联调。

### 6.2 浏览器端到端

真实模型下已验证：

1. 新建会话，发送“计算 2 的 10 次方，然后记一条待办”；
2. Agent 实际调用 calculator 与 todo；
3. UI 显示 2 步工具时间线，状态均为 ok；
4. 刷新页面，Hash `#/s/<id>` 恢复会话与工具时间线；
5. 新建第二个会话做纯对话，两个会话的消息/待办互不影响；
6. 切回第一个会话，历史与待办正确恢复；
7. A 会话发送长任务时切到 B：B 的发送按钮可用，B 正常返回；侧栏 A 仍显示“思考中…”；
8. A 完成后其回复、历史与待办正确落盘；
9. 浏览器控制台无报错。

### 6.3 验收标准

README 第 3 节 AC-01~AC-30 均有对应测试或手工步骤，已于 2026-10-19 完成走查。

---

## 7. AI 使用边界与反思

AI 在本项目中适合：

- 快速搭出分层代码与测试；
- 对照需求找出遗漏；
- 生成边界用例；
- 整理文档。

AI 不能替代的判断：

- 真实 API 的行为与模型指令遵循度；
- 安全边界：密钥、路径穿越、XSS、Trace 隔离；
- 预算与产品语义：用户是否应该为工具输出过长负责；
- 最终验收：必须跑真实测试和浏览器流程。

---

## 8. 复现命令

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # 填入 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
uvicorn app.main:app --reload --port 8000
```

测试：

```bash
pytest -q
pytest -m live -q
```

浏览器打开 <http://localhost:8000>。

---

## 9. 后续改进

- 流式输出（SSE）与逐 token 展示；
- SQLite/Postgres 替代 JSON 文件；
- 可选 tokenizer 适配器（OpenAI 场景可接 tiktoken）；
- 多用户认证与 Session 归属校验；
- 更严格的工具结果结构化压缩策略；
- 评测集与自动回归（固定任务集 + 成功率/成本统计）。
