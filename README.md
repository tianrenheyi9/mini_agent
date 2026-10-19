# 省心购 Vibe Coding · 最小可用 Agent

一个**不依赖任何 Agent 框架**（无 LangGraph / LangChain / OpenHands 等）的最小可用 Agent Runtime。
核心循环、工具注册与调度、LLM 输出解析、Session 管理、上下文压缩、异常处理与 Trace 全部自研，底层使用 OpenAI 官方 SDK 与 Web 框架。

- 技术栈：Python 3.11+ / FastAPI / OpenAI SDK（底层 httpx）/ 原生 HTML+CSS+JS（无前端构建步骤）
- LLM：任意 OpenAI 兼容接口（`BASE_URL + API_KEY + MODEL` 配置）
- 工具协议：自定义 JSON 协议，由自研解析器提取 `thought / action / final`
- 会话：Web 端左侧 Session 列表，多个浏览器窗口可分别绑定不同 Session，互不影响
- 存储：文件系统 JSON（会话）+ JSONL（Trace），无需数据库
- 测试：pytest（单元 / 集成 / 真实 LLM 冒烟）

> 代码仓库：`<GitHub 链接待填>`

---

## 目录

1. [快速开始](#1-快速开始)
2. [需求分析与实现对照](#2-需求分析与实现对照)
3. [验收标准](#3-验收标准)
4. [系统设计](#4-系统设计)
5. [Memory 的召回时机与放置方式](#5-memory-的召回时机与放置方式)
6. [上下文压缩策略](#6-上下文压缩策略)
7. [异常处理](#7-异常处理)
8. [Trace 与日志](#8-trace-与日志)
9. [API 一览](#9-api-一览)
10. [测试方案](#10-测试方案)
11. [设计取舍与已知限制](#11-设计取舍与已知限制)
12. [AI Prompt 与问题解决记录](#12-ai-prompt-与问题解决记录)

---

## 1. 快速开始

### 1.1 环境要求

- Python 3.11+
- 一个可用的 OpenAI 兼容 LLM API Key（见 1.3）

### 1.2 安装、配置与启动

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env    # macOS/Linux: cp .env.example .env
```

编辑 `.env`：

```ini
# 任意 OpenAI 兼容接口
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
LLM_MODEL=deepseek-chat

# 可选：访问境外 API 时设置标准代理环境变量
# OpenAI SDK 底层的 httpx 会自动读取 HTTPS_PROXY
# HTTPS_PROXY=http://127.0.0.1:7897

# Agent 运行参数
LLM_TIMEOUT_SECONDS=60
# 温度与输出上限；留空表示不发送该参数（兼容不支持这些参数的模型）
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=1024
# 开启后发送 response_format={"type": "json_object"}；部分兼容服务不支持
LLM_JSON_MODE=false

TOOL_TIMEOUT_SECONDS=10       # 单个工具执行超时
MAX_TOOL_STEPS=8              # 单轮最多几次“决策 → 工具”循环
MAX_LLM_CALLS_PER_TURN=10     # 单轮 LLM 调用上限（含协议纠错）
MAX_COMPRESSION_CALLS_PER_TURN=1  # 单轮压缩调用上限
MAX_PARSE_REPAIRS=2           # 单轮协议纠错最多重试几次
CONTEXT_TOKEN_BUDGET=6000     # 估算 token 超过该值时触发压缩
MAX_HISTORY_MESSAGES=40       # 未压缩历史的消息条数硬上限
KEEP_RECENT_MESSAGES=8        # 压缩后保留的最近消息条数
SUMMARY_MAX_CHARS=800         # 会话摘要最大字符数
DATA_DIR=./data               # 会话与 Trace 落盘目录
```

启动：

```bash
uvicorn app.main:app --reload --port 8000
```

打开 <http://localhost:8000>。

### 1.3 Provider 配置示例

| Provider | `LLM_BASE_URL` | `LLM_MODEL` 示例 |
| --- | --- | --- |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问（兼容模式） | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Moonshot | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |

> 只需兼容 `POST /chat/completions`（`messages` 入参、`choices[0].message.content` 出参）即可接入。
> 项目**不使用**原生 function calling，工具决策完全由 Prompt + 自研解析器完成，因此对模型能力要求低、可移植性强。

### 1.4 运行测试

```bash
pytest -q                 # 单元 + 集成（全部使用 FakeLLM，无需 API Key）
pytest -q -m live         # 真实 LLM 冒烟测试（需要 .env 配置）
```

### 1.5 两个窗口的演示步骤

1. 打开浏览器窗口 A，新建会话，输入：
   > 北京今天天气怎么样？另外帮我把“给妈妈买生日礼物”记到待办里。
   >
   > 预期：调用 `weather` → 调用 `todo(add)` → 汇总回复。
2. 打开浏览器窗口 B，新建会话，输入：
   > 帮我写一份本周周报，3 条要点，并把“周五前提交周报”记到待办里。
   >
   > 预期：直接生成周报文本 → 调用 `todo(add)`。B 中看不到 A 的待办。
3. 回到窗口 A，追问：
   > 刚才记的待办是什么？
   >
   > 预期：调用 `todo(list)`，只返回 A 会话的待办。
4. 在 A 继续追问：
   > 那上海呢？
   >
   > 预期：带上下文工具追问，调用 `weather(上海)`。
5. 刷新两个窗口：URL 中的 `#/s/<session_id>` 会恢复对应会话，历史消息、待办、摘要均保留。

---

## 2. 需求分析与实现对照

### 2.1 题目拆解

| 题目要求 | 理解与落点 |
| --- | --- |
| 从零实现 Agent Runtime | 手写 `AgentLoop`：构建上下文 → 调 LLM → 解析 → 执行工具 → 回填 → 再决策。不引入任何 Agent/编排框架 |
| Step 1 接收用户输入 | `POST /api/sessions/{id}/messages` 写入用户消息，开启一个 turn |
| Step 2 判断直接回复还是调用工具 | 解析 LLM 输出：含 `final` 则直接回复；含 `action` 则进入工具调用 |
| Step 3 调用工具 | `ToolRegistry` 按名称查找工具 → Schema 校验参数 → 注入 `session_id` → 执行 → 结果作为 observation 回填 |
| Step 4 判断继续 loop 还是返回 | 一轮工具执行后重新调 LLM，直到输出 `final` 或触达 `MAX_TOOL_STEPS` |
| 至少 3 个工具，todo 必须 | 实现 4 个：`todo`、`calculator`、`search`（mock）、`weather`（mock） |
| 工具注册机制（名称、描述、参数 Schema） | `ToolRegistry.register(ToolSpec)`，`ToolSpec` 含 `name / description / parameters(JSON Schema) / handler` |
| LLM 基于 Schema 自主决策 | 每轮把工具目录（含完整 Schema）渲染进 System Prompt |
| 实现 LLM 输出解析逻辑 | `parser.py` 自研：去代码围栏、扫描平衡大括号、容错前后缀文本、校验协议、区分 `thought/action/final` |
| Session 隔离与续聊 | 每个 Session 独立 JSON 文件 + 独立内存锁；列表/切换/刷新恢复；A/B 窗口互不影响 |
| 最大轮次限制 | 分开限制：`MAX_TOOL_STEPS`（决策循环）、`MAX_LLM_CALLS_PER_TURN`（含纠错）、`MAX_COMPRESSION_CALLS_PER_TURN`（压缩） |
| 记住之前状态 / 支持追问 | 最近消息窗口 + 会话摘要 + 实时待办快照 + 当前轮工具链 |
| 纯对话追问 | 历史 user/assistant 消息在 context 中 |
| 带工具追问 | 历史 + 工具状态快照 + 工具目录持续可见 |
| 哪些信息塞入 context | 见 [第 5 节](#5-memory-的召回时机与放置方式)，逐项给出放置位置与理由 |
| context 过长基础压缩 | token 估算超阈值时，对较旧消息做 LLM 增量摘要；失败时硬截断兜底 |
| 基本异常处理 | LLM 重试、解析纠错、未知工具/参数错误自纠正、工具异常回填、超步数兜底 |
| 工具调用 trace / 执行日志 | 每次 LLM 调用、工具调用、压缩、异常都写 JSONL + 控制台日志，并在 UI 可展开查看 |
| 测试用例 | 单元 + 集成 + 真实 LLM 冒烟 + 手工双窗口验收，见 [第 10 节](#10-测试方案) |
| 真实 LLM API | 通过 OpenAI 兼容 HTTP 接口调用，无任何 mock LLM 参与线上路径 |
| README 三件套 | 运行方式、系统设计、Memory 召回时机与放置（本文）；AI 记录见 `AI_LOG.md` |

### 2.2 技术选型

| 选择 | 方案 | 理由 |
| --- | --- | --- |
| 语言 | Python 3.11 | Agent 逻辑短、可读性高；pytest 测试生态成熟 |
| Web | FastAPI + uvicorn | 自带 Pydantic 校验与 OpenAPI；静态文件托管足够本项目 UI |
| LLM 客户端 | `openai` SDK（`AsyncOpenAI`） | 官方 SDK 负责请求构造、响应类型解析、超时与重试；`base_url` 兼容任意 Provider，错误映射自行实现 |
| 前端 | 原生 HTML/CSS/JS | 免构建、免 Node 环境，面试官 `pip install` 即可跑 |
| 存储 | 每会话一个 JSON + Trace JSONL | 零依赖、可读、易调试；单机 MVP 足够 |
| Schema 校验 | 自研 JSON Schema 子集校验 | 只支持 `type/required/enum/properties/items`，足以覆盖工具参数，避免过度设计 |
| token 估算 | 中英文加权字符估算 | 避免引入 tiktoken；误差由预算留白吸收 |
| 测试 | pytest + FakeLLM | Agent 循环与异常分支可确定性测试；真实 API 单独标记 |

**明确不使用**：LangGraph、LangChain、OpenHands、AutoGen、CrewAI、官方 Agent SDK 等任何 Agent 编排/运行时框架。

### 2.3 项目结构（规划）

```text
.
├─ app/
│  ├─ main.py                 # FastAPI 入口、静态资源、异常兜底
│  ├─ config.py               # 环境变量与运行参数
│  ├─ api/
│  │  └─ routes.py            # Session / Message / Trace 接口
│  ├─ agent/
│  │  ├─ loop.py              # AgentLoop：核心循环与终止条件
│  │  ├─ parser.py            # 输出协议解析器
│  │  ├─ prompt.py            # System Prompt / 协议说明 / 工具目录渲染
│  │  ├─ context.py           # 上下文构建、token 估算、压缩
│  │  └─ llm.py               # OpenAI SDK 客户端、异常映射与超时
│  ├─ tools/
│  │  ├─ base.py              # ToolSpec / 参数校验 / 执行结果
│  │  ├─ registry.py          # 工具注册表
│  │  ├─ todo.py              # 待办（会话级状态）
│  │  ├─ calculator.py        # 安全四则运算（AST 白名单）
│  │  ├─ search.py            # Mock 搜索（固定语料）
│  │  └─ weather.py           # Mock 天气（固定规则）
│  ├─ session/
│  │  ├─ models.py            # Session / Message / 统计 / 压缩游标
│  │  ├─ manager.py           # 创建/读取/删除/会话锁
│  │  └─ store.py             # JSON 原子读写
│  ├─ trace.py                # JSONL Trace + 日志
│  └─ web/
│     ├─ index.html
│     ├─ app.js
│     └─ style.css
├─ tests/
│  ├─ agent/
│  │  ├─ test_parser.py
│  │  ├─ test_prompt.py
│  │  ├─ test_loop.py
│  │  ├─ test_context.py
│  │  └─ test_live_agent.py
│  ├─ tools/
│  │  ├─ test_schema.py
│  │  ├─ test_registry.py
│  │  ├─ test_calculator.py
│  │  ├─ test_todo.py
│  │  ├─ test_search.py
│  │  └─ test_weather.py
│  ├─ session/
│  │  ├─ test_models.py
│  │  ├─ test_store.py
│  │  └─ test_manager.py
│  ├─ llm/
│  │  └─ test_llm.py
│  ├─ test_api.py
│  └─ test_trace.py
├─ data/                      # 运行时生成（gitignore）
├─ .env.example
├─ requirements.txt
├─ AI_LOG.md                  # AI Prompt 与问题解决记录
└─ README.md
```

---

## 3. 验收标准

> 每条 AC 都能通过自动化测试或明确的手工步骤复验。测试文件名为规划命名，实现时保持一致。

| ID | 验收项 | 预期结果 | 验证方式 |
| --- | --- | --- | --- |
| AC-01 | 直接回复 | 无需工具的问题直接返回 `final` | `test_loop.py::test_direct_answer` |
| AC-02 | 单工具调用 | 调用 `calculator` 后基于结果回复 | `test_loop.py::test_single_tool` |
| AC-03 | 多工具串联 | 一轮内 `weather` → `todo(add)` → 汇总 | `test_loop.py::test_multi_step_chain` |
| AC-04 | 工具注册与 Schema | 注册表含 4 个工具；缺失必填/类型错误被拒绝 | `test_tools.py` |
| AC-05 | todo 持久化 | 新增/完成/删除/列表跨轮、跨请求生效 | `test_tools.py::test_todo_lifecycle` |
| AC-06 | 会话隔离 | A/B 交错发消息，待办与历史互不污染 | `test_sessions.py::test_session_isolation` |
| AC-07 | 会话恢复 | 重新从磁盘加载后历史完整，可继续对话 | `test_sessions.py::test_reload_from_disk` |
| AC-08 | 纯对话追问 | 多轮指代（“那上海呢”）能利用历史 | `test_loop.py::test_pure_chat_follow_up` |
| AC-09 | 带工具追问 | 追问触发正确工具且参数来自上下文 | `test_loop.py::test_tool_follow_up` |
| AC-10 | 最大轮次 | 超过 `MAX_TOOL_STEPS` 停止并给出兜底回复 | `test_loop.py::test_max_steps_exceeded` |
| AC-11 | 上下文压缩 | 超预算触发摘要；最近窗口保留；关键事实不丢 | `test_context.py::test_compression_triggered` |
| AC-12 | 压缩兜底 | 摘要 LLM 失败时硬截断且不中断对话 | `test_context.py::test_summary_failure_fallback` |
| AC-13 | 解析容错 | 代码围栏、前后缀文本、嵌套大括号、非法 JSON | `test_parser.py` |
| AC-14 | 未知工具自纠正 | 回填可用工具列表，模型下一轮改正 | `test_loop.py::test_unknown_tool_self_correct` |
| AC-15 | 工具异常 | 异常转为 observation，不出 500，trace 留存堆栈 | `test_loop.py::test_tool_exception` |
| AC-16 | LLM 异常 | 超时/5xx 重试后返回可读错误 | `test_loop.py::test_llm_retry_then_fail` |
| AC-17 | Trace | LLM 调用与工具调用均落 JSONL，含耗时与状态 | `test_trace.py` |
| AC-18 | API 契约 | 创建/列表/详情/发消息/404 | `test_api.py` |
| AC-19 | 真实 LLM | 通过 AgentLoop 完成一次工具调用并最终回答 | `test_live_agent.py`（`-m live`） |
| AC-20 | 双窗口演示 | 按 1.5 步骤手动复验 | 手工 |
| AC-21 | 工具结果回填格式 | 不使用缺少 `tool_call_id` 的 `role="tool"`；assistant 原始协议 + user observation | `test_loop.py::test_observation_message_format` |
| AC-22 | 压缩游标 | 第二次压缩只处理游标之后的新消息，不重复摘要 | `test_context.py::test_compression_cursor_advances` |
| AC-23 | 分类统计 | LLM 调用、工具调用、压缩次数与 token 分别统计 | `test_loop.py::test_turn_stats` |
| AC-24 | Calculator 资源上限 | 嵌套乘方在执行前按预估位宽拒绝 | `test_calculator.py::test_nested_power_rejected` |

---

## 4. 系统设计

### 4.1 总体架构

```text
┌─────────────────────────────────────────────────────────────┐
│  浏览器（窗口 A / 窗口 B）                                    │
│  左侧 Session 列表 · 聊天区 · 思考/工具调用可展开 · URL 绑定会话 │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST / JSON
┌──────────────────────────▼──────────────────────────────────┐
│  FastAPI                                                     │
│  routes: sessions / messages / traces / health               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Agent Runtime（自研）                                        │
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ContextBuilder│──▶│   AgentLoop  │◀─▶│   LLMClient      │  │
│  │ 摘要/历史/快照│   │ 多步循环/终止 │   │ OpenAI 兼容 HTTP │  │
│  └─────────────┘   └──────┬───────┘   └──────────────────┘  │
│         ▲                 │ parse                            │
│         │                 ▼                                  │
│  ┌──────┴──────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │SessionManager│   │    Parser    │   │  ToolRegistry    │  │
│  │ 隔离/锁/落盘 │   │ thought/action│──▶│ Schema 校验/执行 │  │
│  └─────────────┘   │ /final       │   └────────┬─────────┘  │
│         │          └──────────────┘            │            │
│         │                                      ▼            │
│         │                             todo/calculator/       │
│         │                             search/weather         │
│         ▼                                                     │
│  JSON 会话文件 + JSONL Trace                                  │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 一次用户请求的时序

```text
用户 ──▶ POST /messages
          │
          ├─ 1. SessionManager 加锁，加载会话
          ├─ 2. 追加 user 消息（分配 msg_id，记录 turn_id）
          ├─ 3. ContextBuilder 估算 token；超预算 → 按游标压缩旧消息
          ├─ 4. while step < MAX_TOOL_STEPS 且 llm_calls < MAX_LLM_CALLS_PER_TURN:
          │      ├─ 构建 messages（System + 历史窗口 + 当前轮）
          │      ├─ LLMClient.chat() ──▶ 原始文本
          │      ├─ Parser.parse() ──▶ ToolCall | FinalAnswer | ParseError
          │      ├─ FinalAnswer ──▶ 落盘、返回
          │      ├─ ToolCall ──▶ Registry.execute()
          │      │                  ├─ 追加 assistant 原始协议消息
          │      │                  └─ 以 user 角色追加 observation（见 4.4.1）
          │      └─ ParseError → 协议纠错消息（计入 MAX_PARSE_REPAIRS / MAX_LLM_CALLS_PER_TURN）
          ├─ 5. 触达任一上限 → 汇总已有 observation，返回兜底回复
          └─ 6. 释放锁，更新统计，写 Trace，返回 reply + steps
```

### 4.3 Agent Loop（伪代码）

```python
async def run_turn(session, user_text):
    turn = Turn(turn_id=session.new_turn_id())
    session.append_user(user_text, turn.id)

    # 压缩按游标增量进行，并受 MAX_COMPRESSION_CALLS_PER_TURN 限制
    if context.over_budget(session):
        await context.compress(
            session,
            turn,
            budget=config.MAX_COMPRESSION_CALLS_PER_TURN,
        )

    for step in range(config.MAX_TOOL_STEPS):
        # 这是独立于工具循环的第二道预算：协议纠错也会消耗 LLM 调用
        if turn.llm_calls >= config.MAX_LLM_CALLS_PER_TURN:
            return graceful("本轮 LLM 调用次数已达上限，先给出阶段性结果。")

        messages = context.build(session, turn.id)
        response = await llm.chat(messages)      # SDK 内部处理超时/重试

        turn.llm_calls += 1
        turn.prompt_tokens += response.prompt_tokens or 0
        turn.completion_tokens += response.completion_tokens or 0
        trace.llm_call(turn, step, response)

        parsed = parser.parse(response.content)

        if isinstance(parsed, FinalAnswer):
            session.append_assistant_final(parsed, turn.id)
            return TurnResult(reply=parsed.final, steps=turn.steps, stats=turn.stats)
            # 工具结果的“结论”已由模型写入 final，后续轮次不再重复注入原始工具输出

        if isinstance(parsed, ParseError):
            if turn.repairs >= config.MAX_PARSE_REPAIRS:
                return graceful("模型输出格式异常，请重试。")
            turn.repairs += 1
            # 保留模型的原始输出，再追加明确的纠错消息
            session.append_assistant_raw(response.content, turn.id)
            session.append_user(
                build_repair_message(response.content, parsed.reason),
                turn.id,
                kind="repair",
            )
            continue

        # ToolCall：先记录模型这次决策，再执行工具
        session.append_assistant_tool_call(parsed, response.content, turn.id)
        result = await registry.execute(
            name=parsed.action,
            raw_input=parsed.input,
            context=ToolContext(session_id=session.id, todos=session.todos),
        )

        turn.tool_calls += 1
        # 自定义协议不使用 role="tool"；
        # 工具结果以 user 角色的显式 observation 回填（见 4.4.1）
        session.append_tool_observation(parsed, result, turn.id)
        trace.tool_call(turn, step, parsed, result)

    return graceful(
        "已达到单轮最大工具循环次数（%d），先给你阶段性结果。"
        % config.MAX_TOOL_STEPS
    )
```

**终止条件**（任一满足）：

1. 解析出 `final` → 正常返回；
2. 达到 `MAX_TOOL_STEPS`（决策循环上限）→ 兜底返回；
3. 达到 `MAX_LLM_CALLS_PER_TURN`（含协议纠错）→ 兜底返回；
4. 达到 `MAX_PARSE_REPAIRS` → 格式异常提示；
5. LLM 重试耗尽 → 错误提示；
6. 客户端断开 → 取消协程（取消时也写 Trace）。

**统计口径**：`TurnStats` 分开记录 `llm_calls`、`tool_calls`、`compression_calls`、`prompt_tokens`、`completion_tokens`；Session 再累计一份，供 Trace 和 UI 展示。

### 4.4 LLM 输出协议与解析器

**协议**：模型每轮只能输出一个 JSON 对象。

工具调用：

```json
{
  "thought": "用户要查北京天气，我先调用天气工具。",
  "action": "weather",
  "input": { "city": "北京" }
}
```

最终回答：

```json
{
  "thought": "已经拿到天气和待办结果，可以汇总回复。",
  "final": "北京今天 18~26℃，多云。已帮你把“给妈妈买生日礼物”记入待办。"
}
```

**解析器职责**（`agent/parser.py`）：

1. 去除 Markdown 代码围栏（```json / ```）；
2. 若模型输出前后夹带说明文字，**扫描字符串中第一个平衡的 `{...}`**（正确处理字符串内的花括号与转义）；
3. `json.loads` 失败 → 返回 `ParseError`（不抛异常到接口层）；
4. 结构校验：
   - `action` 与 `final` 必须**恰好出现一个**；
   - `action` 必须是非空字符串，`input` 必须是对象（缺省为 `{}`）；
   - `thought` 可选，必须为字符串；
5. 返回判别联合：`ToolCall(thought, action, input)` / `FinalAnswer(thought, final)` / `ParseError(reason)`；
6. 解析失败时，把“原始输出 + 具体错误”作为协议纠错消息回填，最多 2 次。

**为什么不用原生 Function Calling**：

- 题目要求“实现 LLM 输出的解析逻辑”，自研协议能完整覆盖决策、解析、纠错、trace；
- 不依赖特定厂商的 tools 能力，任何 OpenAI 兼容模型都能跑；
- `thought` 字段让思考过程显式可观测，便于 UI 展示与调试。

#### 4.4.1 工具结果如何回填（重要）

自定义 JSON 协议**不是** OpenAI 原生 `tool_calls`。因此不能直接发送：

```python
{"role": "tool", "content": "..."}  # 缺少 tool_call_id，API 会拒绝或行为异常
```

本项目采用“**assistant 原始协议 + user observation**”：

```python
# 1. 原样保留模型刚才的决策
{
    "role": "assistant",
    "content": '{"action":"calculator","input":{"expression":"2**10"}}',
}

# 2. 用 user 角色回填工具结果
{
    "role": "user",
    "content": """[工具执行结果]
tool: calculator
status: ok
input: {"expression": "2**10"}
result: {"result": 1024}

请继续：需要更多信息就调用工具，否则输出 final。""",
}
```

失败时同样回填，但标记为 error：

```text
[工具执行结果]
tool: weather
status: error
input: {"city": ""}
error: 城市不能为空
请修正参数后重试，或向用户说明原因。
```

这样做的原因：

- 不需要伪造 `tool_call_id`；
- 模型能看到“自己刚才做了什么”和“工具返回了什么”，支持多步串联；
- 与自定义协议完全一致，不混用两套调用机制；
- `[工具执行结果]` 是显式标记，避免模型把 observation 误认为新的用户需求。

**不要同时使用 `role="tool"` 和自定义协议**：那会形成一套不完整的原生工具调用消息，容易导致 API 报错或模型混淆。

### 4.5 工具系统

**注册机制**：

```python
@dataclass(frozen=True)
class ToolSpec:
    name: str                 # 唯一名称，LLM 调用时使用
    description: str          # 何时使用、返回什么
    parameters: dict          # JSON Schema 子集
    handler: Callable[..., Awaitable[Any]]
```

- `ToolRegistry.register(spec)`：名称重复直接报错（启动期暴露问题）；
- `ToolRegistry.tools()`：返回全部 ToolSpec，交给 `prompt.py` 渲染工具目录；
- `ToolRegistry.execute(name, raw_input, context)`：
  1. 查找工具（未知 → 返回可用工具列表）；
  2. 用自研 `validate_parameters(schema, raw_input)` 校验 `required/type/enum/properties/items`；
  3. Runtime 注入 `ToolContext`（含 `session_id` 与 `todos`，模型看不到这些内部字段）；
  4. 执行并统一包装为 `ToolResult(tool, ok, data, error, duration_ms)`；
  5. 异常捕获，返回可读错误，完整堆栈只进日志/Trace。

**四个工具**：

| 工具 | 作用 | 参数 Schema（摘要） | 说明 |
| --- | --- | --- | --- |
| `todo` | 会话级待办管理 | `action: add/list/complete/delete/clear`；`content?`、`id?` | **必需**。状态存 Session，不靠模型记忆 |
| `calculator` | 安全算术计算 | `expression: string` | `ast` 白名单解析，禁止 `eval`、变量、函数调用、属性访问 |
| `search` | 文档/网页检索（Mock） | `query: string`；`limit?: int(1~5)` | 固定语料关键词打分，结果确定，便于测试 |
| `weather` | 天气查询（Mock） | `city: string`；`date?: string` | 固定城市表 + 日期哈希生成稳定天气 |

**待办数据流**：

```text
todo(add) ──▶ 写入 session.todos（落盘）
                     │
下一次构建 context ──▶ 渲染为 System Prompt 的“当前待办快照”
                     │
UI 会话侧栏/详情 ◀── 直接读取 session.todos
```

即：待办既通过工具修改，又通过**每轮实时快照**进入上下文。模型不需要“记住”待办，避免状态漂移。

### 4.6 Session 管理

**数据模型**（`data/sessions/<session_id>.json`）：

```json
{
  "id": "s_01J...",
  "title": "北京天气与待办",
  "created_at": "2026-10-18T12:00:00+08:00",
  "updated_at": "2026-10-18T12:03:00+08:00",
  "summary": "用户询问北京天气并把买礼物加入待办……",
  "summary_until_msg_id": "m_6",
  "next_msg_seq": 7,
  "stats": {
    "llm_calls": 3,
    "tool_calls": 2,
    "compression_calls": 1,
    "prompt_tokens": 1832,
    "completion_tokens": 210
  },
  "todos": [
    { "id": "t_1", "content": "给妈妈买生日礼物", "done": false, "created_at": "..." }
  ],
  "messages": [
    { "msg_id": "m_1", "role": "user", "content": "...", "turn_id": "u_1", "created_at": "..." },
    {
      "msg_id": "m_2",
      "role": "assistant",
      "kind": "tool_call",
      "content": "{\"action\":\"weather\",\"input\":{\"city\":\"北京\"}}",
      "thought": "先查天气",
      "turn_id": "u_1",
      "created_at": "..."
    },
    {
      "msg_id": "m_3",
      "role": "user",
      "kind": "tool_observation",
      "tool": "weather",
      "status": "ok",
      "content": "[工具执行结果] tool=weather status=ok ...",
      "turn_id": "u_1",
      "created_at": "..."
    },
    {
      "msg_id": "m_4",
      "role": "assistant",
      "kind": "final",
      "content": "北京今天 18~26℃，多云。",
      "thought": "信息已足够",
      "turn_id": "u_1",
      "created_at": "..."
    }
  ]
}
```

**三个关键字段**：

| 字段 | 作用 |
| --- | --- |
| `msg_id` | 会话内单调递增（`m_1`、`m_2`…），永不复用；压缩游标和 UI 定位都依赖它 |
| `next_msg_seq` | 下一条消息的序号；落盘保存，避免重启后 id 回退 |
| `summary_until_msg_id` | 摘要已经覆盖到的最后一条消息；压缩时只处理它之后的新消息，避免重复摘要 |
| `stats` | 累计统计 LLM 调用、工具调用、压缩次数与 token，和 `MAX_TOOL_STEPS` 等限制分开 |

**消息 `kind`**：`user` / `final` / `tool_call` / `tool_observation` / `repair`。
历史轮次进入上下文时只取 `user` 原文与 `final`；`tool_call`、`tool_observation`、`repair` 供当前轮推理、UI 和 Trace 使用。

**隔离与并发**：

- 会话 ID 使用 `s_` + 随机 ULID/UUID，文件名与 ID 一致；
- 每个会话一把 `asyncio.Lock`，同一会话请求串行，不同会话可并行；
- 写入采用“临时文件 + `os.replace`”原子替换，避免半写文件；
- 列表接口只读取元数据（启动时可用轻量索引），按 `updated_at` 倒序；
- URL Hash `#/s/<id>` 绑定会话：两个浏览器窗口天然是两条独立会话，刷新可恢复。

**消息不删除**：压缩只推进 `summary_until_msg_id`，不改写历史消息；UI 与 Trace 仍能看到完整过程。

**首条消息生成标题**：取首条用户消息前 20 个字符作为标题（失败/为空则“新会话”），无需额外 LLM 调用。

### 4.7 Web 交互

- 左侧：Session 列表（标题、更新时间、新建按钮、删除）；
- 右侧：消息流；assistant 消息下可展开「思考过程」与「工具调用」时间线（来自本轮 `steps`）；
- 输入框：Enter 发送、Shift+Enter 换行；发送中禁用并显示进度；
- 错误以条幅展示（模型不可用、网络错误、会话不存在）；
- 无构建步骤：FastAPI `StaticFiles` 直接托管 `app/web/`。

---

## 5. Memory 的召回时机与放置方式

本项目的 “memory” 指**会话内记忆**。Session 是记忆的隔离边界：跨 Session 不共享任何上下文或状态（符合题目“窗口 1 / 窗口 2 互不影响”）。

### 5.1 信息放置总表

| 信息 | 存储位置 | 进入上下文的位置 | 召回时机 | 为什么放这里 |
| --- | --- | --- | --- | --- |
| System 角色与输出协议 | `prompt.py` 常量 | System Prompt 开头 | 每次 LLM 调用 | 协议必须最高优先级、每轮可见 |
| 工具目录（名称/描述/Schema） | `ToolRegistry` | System Prompt | 每次 LLM 调用 | 让模型基于真实 Schema 决策，避免臆造参数 |
| 会话摘要 | `session.summary` | System Prompt | 每轮构建上下文时 | 压缩后的长期事实，放在 system 中不影响消息配对 |
| 摘要游标 | `session.summary_until_msg_id` | 不进入 Prompt | 每次压缩 | 只摘要游标之后的新消息，防止重复摘要 |
| 运行统计 | `session.stats` / `TurnStats` | Trace / TurnResult | 每轮结束 | LLM、工具、压缩次数与 token 分开统计 |
| 待办快照 | `session.todos` | System Prompt | 每轮**实时生成** | 状态类事实以存储为准，避免模型“记忆漂移” |
| 当前时间 | 运行时 | System Prompt | 每轮 | 支持“今天/本周”等相对时间 |
| 最近 N 条已完成对话 | `session.messages` | messages 数组尾部 | 每轮 | 保证追问连贯，控制 token |
| 当前轮的 user 输入 | `session.messages` | messages 数组 | 当前轮 | 直接任务输入 |
| 当前轮的 assistant 工具调用 | `session.messages` | messages 数组 | 同一轮内后续步骤 | 多步工具串联必需（模型要知道上一步调了什么） |
| 当前轮的工具 observation | `session.messages` | messages 数组（user 角色 + `[工具执行结果]` 标记） | 同一轮内后续步骤 | 工具结果是本轮推理依据；不使用缺少 `tool_call_id` 的 `role="tool"` |
| **历史轮的中间工具调用/结果** | `session.messages`（供 UI/Trace） | **默认不进入上下文** | 不召回 | final 已沉淀结论；原始结果重复注入会快速膨胀 |
| **历史轮的 thought** | `session.messages`（供 UI/Trace） | **不进入上下文** | 不召回 | 思考过程只对当轮决策有效，历史 thought 会干扰后续 |
| 完整 Trace | `data/traces/*.jsonl` | 不进入上下文 | 仅排障 | 可观测性数据，不属于对话记忆 |

### 5.2 每轮上下文的最终形态

```text
messages = [
  { role: "system", content:
      [角色与行为约束]
    + [输出协议：只输出一个 JSON，action/final 二选一]
    + [工具目录：4 个工具的 name/description/JSON Schema]
    + [会话摘要：≤ SUMMARY_MAX_CHARS]
    + [当前待办快照：未完成在前，含 id]
    + [当前时间]
  },
  ...最近 K 条已完成轮次（仅 user 原文 + assistant final）,
  { role: "user", content: 当前输入 },
  ...当前轮中间步骤（assistant 原始协议 + user observation）,
]
```

工具 observation 的放置方式很关键：它使用 **user 角色**并带 `[工具执行结果]` 标记，而不是 `role="tool"`。原因见 [4.4.1](#441-工具结果如何回填重要)。

### 5.3 追问为什么能工作

- **纯对话追问**（“那上海呢？”）：
  最近消息窗口里保留上一轮的 `user: 北京天气…` 与 `assistant: 北京今天…`，模型可解析指代，直接输出 `final` 或再次调用工具。
- **带工具追问**（“刚才记的待办是什么？”）：
  即使历史消息被压缩，`session.todos` 仍会以快照形式进入 System Prompt；模型看到待办列表后调用 `todo(list)` 或直接回答。
- **跨轮状态**（“把刚才那条标记完成”）：
  待办 id 在快照中持续可见，模型可用 `todo(complete, id=...)` 精确操作。

### 5.4 召回时机一句话总结

> **每轮请求开始时召回一次**：从磁盘加载会话 → 估算上下文 → 必要时压缩 → 将「摘要 + 待办快照 + 工具目录 + 最近对话 + 当前轮」组装为 prompt。
> **轮内每次 LLM 调用重新组装一次**：追加当前轮新增的 assistant 工具调用与 observation。
> **不做事后召回**：没有向量库、没有跨会话检索；Session 内状态由结构化存储保证，而不是让模型“回忆”。

---

## 6. 上下文压缩策略

### 6.1 触发条件

满足任一即触发（在追加当前用户消息之后、进入循环之前）：

- 估算 token 超过 `CONTEXT_TOKEN_BUDGET`（默认 6000）；
- 已完成消息条数超过安全上限（默认 40 条）。

token 估算采用中英文加权：

```text
estimate_tokens(text) ≈ 中文字符数 × 1.0 + 其他字符数 ÷ 4 + 每条消息固定开销 4
```

估算只用于触发阈值，实际精度不影响正确性；README 明确标注其为近似值。

### 6.2 压缩算法（基础版）

```text
保留：最近 KEEP_RECENT_MESSAGES（默认 8）条已完成消息
摘要范围：summary_until_msg_id 之后、最近窗口之前的消息
摘要输入：旧摘要 + 这批新消息（user 原文、assistant final、工具调用摘要）
摘要输出：≤ SUMMARY_MAX_CHARS 的结构化文本：
  1. 用户目标与偏好
  2. 已确认的关键事实/数字
  3. 已完成的工具操作及其结论
  4. 当前待办与未决问题
写回：session.summary
      session.summary_until_msg_id = 最后一条被摘要消息的 msg_id
      session.stats.compression_calls += 1
消息不删除（供 UI/Trace），但游标之前的消息不再直接进入上下文
```

**增量摘要有两个前提**：

1. `new_summary = summarize(old_summary + new_dropped_messages)`，而不是从零重算；
2. **游标只前进不后退**。第二次压缩时，`m_1..m_12` 已经被摘要覆盖，不会再被送进摘要器，避免重复消耗 token 和摘要失真。

**游标之后没有新消息、但仍然超预算**怎么办？
说明最近窗口本身太大（例如单条工具结果极长）。此时降级为对最近窗口做字符截断，或临时缩小 `KEEP_RECENT_MESSAGES`，并记录 `event=compress, fallback=truncate`。

**分类统计**：

- `compression_calls`：压缩 LLM 调用次数（受 `MAX_COMPRESSION_CALLS_PER_TURN` 限制）；
- `compression_prompt_tokens` / `compression_completion_tokens`：压缩自身消耗；
- 与主循环的 `llm_calls`、`tool_calls` 分开记录，不混为一个 `MAX_TOOL_STEPS`。

**兜底**：

- 摘要 LLM 调用失败/超时 → 不重试超过 1 次，直接使用“上一版摘要 + 被淘汰消息中的最后 2 条 user/assistant 原文”作为新摘要；
- 摘要仍超长 → 按字符硬截断并加 `…`，保证 prompt 不超预算；
- 压缩过程写 Trace（`event=compress`），记录压缩前后估算 token、淘汰条数。

### 6.3 不做的复杂压缩

按题目要求，不实现：向量检索记忆、分层摘要树、重要度打分、工具结果语义去重。README 中将其列为后续可扩展方向。

---

## 7. 异常处理

| 场景 | 处理 | 用户感知 |
| --- | --- | --- |
| LLM 网络错误 / 超时 / 5xx / 429 | 由 OpenAI SDK 自动指数退避重试（`max_retries=2`），仍失败则中止本轮 | “模型服务暂时不可用，请稍后重试。” |
| LLM 4xx（鉴权/参数） | 不重试，记录错误码 | “模型配置有误：<原因>” |
| 输出不是合法 JSON | 回填协议纠错消息，最多 2 次 | 正常回复或格式异常提示 |
| `action` 与 `final` 同时/都缺失 | 视为协议错误，同上 | 同上 |
| 未知工具 | 把“可用工具：…”作为 observation 回填 | 模型自我纠正 |
| 参数 Schema 校验失败 | 把具体字段错误回填 | 模型自我纠正 |
| 工具执行抛异常 | 捕获并回填“工具执行失败：<安全信息>”，堆栈只进 Trace | 模型解释或改用其他方式 |
| 工具超时 | 每个工具 10s 超时，视同执行失败 | 同上 |
| 超过 `MAX_TOOL_STEPS` 或 `MAX_LLM_CALLS_PER_TURN` | 终止循环，返回阶段性结果说明 | 明确告知步数/调用上限 |
| 协议纠错达到 `MAX_PARSE_REPAIRS` | 停止重试，返回格式异常提示 | “模型输出格式异常，请重试。” |
| 压缩达到 `MAX_COMPRESSION_CALLS_PER_TURN` | 跳过后续压缩，降级为硬截断 | 对话继续 |
| calculator 嵌套乘方 | 执行 `**` 前按预估位宽拒绝，不先算出大整数 | 模型看到“计算结果过大”并调整 |
| 会话不存在 / ID 非法 | API 返回 404/400 | UI 提示并回到会话列表 |
| 同一会话并发请求 | 会话锁串行化 | 稍候自动完成 |
| 磁盘写失败 | 返回 500，日志记录；内存状态不提交 | 提示重试 |
| 压缩失败 | 降级为硬截断 | 对话继续 |
| 客户端断开 | 取消异步任务，释放锁 | — |

所有异常都不向模型泄漏堆栈、文件路径或密钥；错误信息经过白名单化处理。

---

## 8. Trace 与日志

### 8.1 Trace 文件

`data/traces/<session_id>.jsonl`，每次请求的每个事件一行：

```json
{"ts":"2026-10-18T12:00:01.120+08:00","session_id":"s_01J...","turn_id":"u_1","step":0,"event":"llm_call","latency_ms":812,"prompt_tokens_est":932,"raw_output":"{\"thought\":...}","parsed":{"type":"tool_call","action":"weather"}}
{"ts":"2026-10-18T12:00:01.130+08:00","session_id":"s_01J...","turn_id":"u_1","step":0,"event":"tool_call","tool":"weather","input":{"city":"北京"},"status":"ok","latency_ms":2,"result_preview":"{\"city\":\"北京\",\"temp\":\"18~26℃\"}"}
{"ts":"2026-10-18T12:00:01.950+08:00","session_id":"s_01J...","turn_id":"u_1","step":1,"event":"llm_call","latency_ms":790,"parsed":{"type":"tool_call","action":"todo"}}
{"ts":"2026-10-18T12:00:02.001+08:00","session_id":"s_01J...","turn_id":"u_1","step":1,"event":"tool_call","tool":"todo","input":{"action":"add","content":"给妈妈买生日礼物"},"status":"ok"}
{"ts":"2026-10-18T12:00:02.300+08:00","session_id":"s_01J...","turn_id":"u_1","event":"compress","before_tokens_est":7200,"after_tokens_est":4100,"summarized_messages":6,"summary_until_msg_id":"m_12","fallback":null}
{"ts":"2026-10-18T12:00:02.900+08:00","session_id":"s_01J...","turn_id":"u_1","step":2,"event":"llm_call","parsed":{"type":"final"}}
{"ts":"2026-10-18T12:00:02.901+08:00","session_id":"s_01J...","turn_id":"u_1","event":"turn_end","status":"ok","steps":2,"llm_calls":3,"tool_calls":2,"compression_calls":1,"prompt_tokens":1832,"completion_tokens":210,"latency_ms":1801}
```

### 8.2 日志输出

- `logging` 标准库，INFO 级别输出：请求开始/结束、工具调用、压缩触发、异常摘要；
- Trace 与日志共用同一 `turn_id`，便于串联排障；
- 敏感信息（API Key）不落日志；原始 LLM 输出最多保留 2000 字符。

### 8.3 UI 展示

`POST /messages` 的响应包含本轮 `steps`，前端在 assistant 消息下渲染可折叠时间线：
「思考 → 调用 weather(北京) → 结果 → 调用 todo(add) → 结果 → 最终回答」。
用户能直接看到 Agent 的决策过程，满足“提取思考过程”的可观测要求。

---

## 9. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查与当前模型配置（不含 Key） |
| `POST` | `/api/sessions` | 新建会话，返回 `session_id` |
| `GET` | `/api/sessions` | 会话列表（id、title、updated_at、todo 未完成数） |
| `GET` | `/api/sessions/{sid}` | 会话详情（messages、summary、todos） |
| `DELETE` | `/api/sessions/{sid}` | 删除会话及其 Trace |
| `POST` | `/api/sessions/{sid}/messages` | 发送消息，执行完整 Agent 轮次 |
| `GET` | `/api/sessions/{sid}/traces?limit=50` | 最近 Trace |

发送消息请求/响应：

```json
// POST /api/sessions/s_01J.../messages
{ "content": "北京天气怎么样？顺便记个待办" }

// 200
{
  "session_id": "s_01J...",
  "reply": "北京今天 18~26℃，多云。已记入待办。",
  "thought": "先查天气，再记录待办，最后汇总。",
  "steps": [
    { "type": "tool", "tool": "weather", "input": { "city": "北京" },
      "status": "ok", "duration_ms": 2, "result": { "city": "北京", "temp": "18~26℃" } },
    { "type": "tool", "tool": "todo", "input": { "action": "add", "content": "买牛奶" },
      "status": "ok", "duration_ms": 1, "result": { "id": "t_3" } }
  ],
  "usage": { "prompt_tokens_est": 1200, "steps_used": 3 },
  "compressed": false
}
```

错误响应统一：

```json
{ "error": { "code": "SESSION_NOT_FOUND", "message": "会话不存在" } }
```

---

## 10. 测试方案

### 10.1 分层

| 层级 | 对象 | 方式 |
| --- | --- | --- |
| 单元 | Parser、Schema 校验、Calculator 安全、Search/Weather 确定性 | 纯函数测试 |
| 组件 | AgentLoop + ToolRegistry | 注入 `FakeLLM`，按脚本返回预设 JSON，断言调用序列 |
| 集成 | SessionManager + 存储 + ContextBuilder | 临时目录，落盘/重载/隔离/压缩 |
| API | FastAPI 路由 | `TestClient` |
| 真实 | 真实 OpenAI 兼容接口 | `pytest -m live`，无 Key 自动跳过 |
| 手工 | 双窗口交互 | 按 1.5 节步骤 |

### 10.2 核心测试用例

**解析器**
- 纯 JSON / ```json 围栏 / 前后有解释文字；
- 字符串内含 `{}`、转义引号、嵌套对象；
- 缺少 `action` 与 `final`、两者同时存在、`input` 非对象、非 JSON；
- 超长输出中定位第一个完整对象。

**工具**
- 重复注册报错；目录渲染包含全部 4 个工具；
- `calculator`：优先级、括号、负数、除零、`__import__`/函数调用/属性访问被拒绝；
- `calculator` 嵌套乘方：`(2**100)**100` 在计算前被预估位宽拦截；
- `todo`：add → list → complete → delete → clear；未知 id 报错；跨请求持久化；
- `search`：命中固定语料、无结果、limit 边界；
- `weather`：同城同日结果稳定、未知城市降级。

**循环**
- 直接回复；单工具；多工具串联；
- **observation 消息格式**：assistant 原始协议 + user 角色 `[工具执行结果]`，不含 `role="tool"`；
- 未知工具 → 自我纠正；参数错误 → 自我纠正；
- 非法 JSON → 纠错重试；达到 `MAX_PARSE_REPAIRS` → 兜底；
- 工具抛异常 → observation 回填后继续循环；
- 达到 `MAX_TOOL_STEPS` 或 `MAX_LLM_CALLS_PER_TURN` → 终止且回复完整；
- `TurnStats` 分别记录 llm_calls / tool_calls / compression_calls / tokens；
- LLM 超时/5xx → SDK 重试 → 最终失败提示。

**Session**
- 两个 Session 交错执行，消息与待办完全隔离；
- `msg_id` 单调递增、不复用；
- `summary_until_msg_id` 持久化并在重载后保留；
- `stats` 累计值与 turn 增量一致；
- 落盘后新建 Manager 重载，历史/摘要/待办一致；
- 删除会话同时删除 Trace；
- 并发请求同一 Session 串行（锁生效）。

**Context**
- 未超预算不压缩；
- 超预算触发摘要，摘要进入 System Prompt，最近 K 条原文保留；
- **游标只前进**：第二次压缩不会重复摘要 `summary_until_msg_id` 之前的消息；
- 摘要合并旧摘要（增量）；
- 摘要失败降级截断；
- 历史工具结果不进入后续上下文（断言 prompt 中不含旧 observation）。

**API**
- 创建/列表/详情/删除；
- 发消息返回 reply + steps + stats；
- 不存在的 session → 404；空消息 → 422。

**真实 LLM**
- 通过 `AgentLoop` 发送“2 的 10 次方是多少，然后记一条待办”；
- 断言至少一次工具调用、最终回复包含 `1024`、todo 落盘；
- **仅调用 `LLMClient.chat()` 回复“你好”不算 Agent 工具调用验收**；live 测试必须走完整循环。

### 10.3 FakeLLM 设计

- 实现与 `LLMClient` 相同的 `chat(messages) -> str` 接口；
- 按顺序返回预设响应，并记录收到的 messages，供断言上下文内容；
- 支持注入异常（超时、5xx）与延迟；
- 使 Agent 循环、压缩、异常路径全部可在无网络、无 API Key 的情况下确定性测试。

---

## 11. 设计取舍与已知限制

| 取舍 | 决定 | 原因 / 代价 |
| --- | --- | --- |
| 工具协议 | 自研 JSON 协议 | 满足“自行解析”要求、可移植；代价是比原生 tool calling 多一层纠错 |
| 工具结果回填 | assistant 原始协议 + user observation | 不伪造 `tool_call_id`；代价是模型需要通过 `[工具执行结果]` 文本标记理解 observation |
| 循环预算 | 步数、LLM 调用、压缩、纠错分别限制 | 避免一个 `MAX_TOOL_STEPS` 混淆不同成本；代价是配置项更多 |
| 压缩游标 | `summary_until_msg_id` 只前进 | 防止重复摘要和 token 浪费；代价是 Session 模型多一个字段 |
| 流式输出 | 暂不实现 | MVP 以完整轮次返回；UI 显示步骤时间线。后续可改 SSE 推送 step 事件 |
| 存储 | JSON 文件 | 零依赖、可读；不适合大规模并发/多实例，后续可换 SQLite/Postgres |
| 记忆 | 仅会话内 | 题目要求窗口隔离；跨会话检索会引入隐私与串扰问题，列为扩展 |
| 压缩 | 单层增量摘要 | 满足“基础压缩”；不做向量召回与摘要树 |
| token 估算 | 字符加权 | 免依赖 tiktoken；预算留白吸收误差 |
| 认证 | 无登录 | 单机演示；生产需加用户体系与 Session 归属校验 |
| 工具 mock | search / weather 固定数据 | 保证测试确定性；真实接入只需替换 handler |

**明确不做的**：多 Agent 协作、MCP、长期向量记忆、流式 token、多用户权限、分布式部署。

---

## 12. AI Prompt 与问题解决记录

完整记录见 [`AI_LOG.md`](./AI_LOG.md)，内容包括：

1. 需求拆解与选型讨论（技术栈、Provider、协议、UI 形态的对比与结论）；
2. 关键 Prompt：System Prompt 全文、协议纠错 Prompt、压缩摘要 Prompt；
3. 开发过程中的问题与解决：解析器容错、上下文膨胀、压缩后指代丢失、Session 并发写入、mock 工具确定性等；
4. 真实 LLM 联调记录：模型不遵守 JSON 协议时的纠错策略与迭代。

---

## 附：提交物清单

- [x] 源码（自研 Agent Runtime，无 Agent 框架依赖）
- [x] `README.md`：运行方式、系统设计、Memory 召回时机与放置方式
- [ ] `AI_LOG.md`：AI Prompt 与问题解决记录
- [ ] 测试用例与运行结果
- [ ] GitHub 仓库链接
