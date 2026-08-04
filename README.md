# win-ai-agent

Windows 本地 AI Coding Agent（MVP）。浏览器聊天，能读写文件、跑命令、联网搜索、改代码，
数据全存本地 SQLite。核心智能靠千问，本项目负责外壳 + 工具循环 + 安全阀 + 存储。

## 快速开始

需要 **Python 3.12+**。

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy config.example.json config.json
```

打开 `config.json` 填两个东西：

- `api_key` —— 阿里云百炼的千问 key
- `work_dir` —— agent 的工作目录，**它只能操作这个目录里的东西**

然后双击 `启动.bat`（或 `python main.py`）。浏览器会自动打开 `http://127.0.0.1:8000`。

> `启动.bat` 会优先用 `.venv` 里的 Python，装不装到系统环境都不影响。
> 8000 端口被占时自动往后找，控制台会打印实际用的端口。

## 它能做什么

输入一个目标，agent 会先列计划，然后按 ReAct 循环逐步执行：读目录、读文件、搜代码、
改文件、跑命令、联网查资料，完成后自检一遍再回复。

**危险操作会停下来等你点确认**：

| 操作 | 是否需要确认 |
|---|---|
| `list_dir` `read_file` `search_in_files` `web_search` `update_plan` `save_memory` | 自动放行 |
| `run_command`，且命令在只读白名单内（`dir` `type` `git status` `python --version` 等） | 自动放行 |
| `run_command`，其它任何命令 | **必须确认** |
| `write_file` | **必须确认**（先给你看 diff） |
| 其它任何工具，**包括以后新加的** | **必须确认** |

自动放行是一份**显式白名单**（`safety/commands.py` 的 `AUTO_APPROVED`），没列进去的一律要
人工确认。反过来写（"认识的危险工具才拦"）是 fail-open：以后加一个会写盘的工具、忘了
同步改白名单，它就静默地自动放行。两条测试盯着这份名单不跟 registry 走散。

**一轮里的多个危险操作合成一张卡**，一并确认或一并拒绝（模型常常一次就发好几个
`write_file`）。同一轮里自动放行的工具不进这张卡，照常执行。

**等待确认时没有任何线程挂着**：图撞上闸就存 checkpoint 并返回，线程随之结束；
你点了确认之后再另起一个线程从 checkpoint 接着跑。所以关掉页面也不会挂住东西，
隔多久回来那道闸都还在，答完照样往下走。

## 安全边界

- **目录沙箱**：所有路径先解析成真实绝对路径再判断是否落在 `work_dir` 下，`..`、符号链接、
  短文件名、跨盘符都拦得住。命令的 cwd 也锁死在 `work_dir`。
- **只读白名单而非危险名单**：危险名单拦不住组合命令（`dir && del /f /s /q *`），所以反着做——
  白名单外的一律人工确认，不做命令语义解析。换行符也算命令分隔符，`type ..\..\x` 这类参数穿越也拦。
- **提示注入**：system prompt 要求忽略工具返回内容里的指令。这不是 100% 可靠，真正的兜底
  仍然是"危险操作必须人工确认"这道闸。
- **系统级操作**（改 `C:\Windows`、写注册表）需要管理员身份启动，默认不给。

## 数据

- `agent.db` —— 会话、消息、长期记忆（SQLite 单文件）。默认在程序目录，可用 `db_path` 改。
  **不能放进 `work_dir`**（agent 对那里有写权限，会改到自己的记忆）——放了直接拒绝启动。
- **长期记忆**：`save_memory` 写进去，右栏面板显示。三条约束：
  重复的不再入库（折叠空白、忽略大小写；极性不同算两条不同的事实）；
  拼进 system prompt 的**只取最近 `max_memories`（默认 50）条**，库里一条不删——
  限的是注入预算，不是数据；
  **禁令单独打标**，提示词里是 `【用户明确禁止】…`，面板上标红。
  「要用 tab」和「不要用 tab」结构一样的话，那个「不」字全靠模型自己读到，读漏一次
  就会做出正好相反的事。
  **记错了能删**：面板上每条记忆 hover 出 `×`，点一下变「删除？」再确认才真删
  （硬删、没有撤销，所以做成两步）。agent 没有删除工具，这是唯一的删除入口——
  没有它，模型记错一次就永久错下去。
- `agent.checkpoints.sqlite` —— LangGraph 的 checkpoint，跟着 `db_path` 走、和 `agent.db`
  分开放（两者的锁和生命周期各归各管）。它只用来跨越确认闸，**任务一结束就删掉那一轮**，
  所以不会无界增长。
- 写文件前自动逐代备份为 `<原名>.<时间戳>.bak`。备份不设上限，长跑后需要人工清理；
  `search_in_files` 会跳过它们，否则搜索结果很快全是历代旧副本。

## 配置项

| 键 | 默认 | 说明 |
|---|---|---|
| `base_url` | 百炼兼容接口 | 换 Claude/GPT 只改这里和 `model` |
| `api_key` | 空 | 留空时发消息会直接报错提示 |
| `model` | `qwen-plus` | 也可用 `qwen-max`。**别用 `qwen-turbo`**，工具调用不稳 |
| `work_dir` | `./workspace` | 沙箱根目录，不存在会自动创建 |
| `db_path` | `./agent.db` | 数据库位置 |
| `search_provider` | `dashscope` | 见下 |
| `port` | `8000` | 被占用时自动 +1 |
| `verify_cmd` | 空 | 验收命令，如 `pytest -q`。**留空就整个关掉**，见下 |
| `model_direct` / `model_slow` | 空 | 模型分层，留空就都用 `model`。见下 |

### 分诊与模型分层

目标进来先过一道**纯关键词**的分诊（`app/agent/router.py`，零 LLM 成本），分成三条路：

| 路径 | 什么样的目标 | 模型 |
|---|---|---|
| `direct` | 元问题与闲聊（"你能做什么"） | `model_direct` |
| `fast` | 目标具体的活儿（"修好 calc.py 的 bug"） | `model` |
| `slow` | 跨文件、要先想清楚（"分析所有会写盘的地方"） | `model_slow` |

判错的代价**不对称**，所以规则往上偏：把闲聊判成 fast 只是多花点 token，把真任务判成
direct 则是让模型没有工具、只能凭空编。因此 `direct` 收得很窄，认不出来的一律走 `fast`。

规则不只看关键词，还看**范围**——「恰好点名一个文件」是具体活儿，「所有 / 整个项目」
或点了两个文件就是跨文件的活。关键词表是先写标注语料、再设计规则定出来的，那份语料
就是 `tests/test_router.py`。

> **当前状态**：分诊和模型分层已经生效，但 `direct` 与 `slow` 目前**仍走 fast 那套
> ReAct**，区别只在用哪档模型。不配 `model_direct` / `model_slow` 的话三档同一个模型，
> 行为与改造前完全一致。

### 联网搜索选哪个

| 值 | 需要额外 key | 有 URL | 说明 |
|---|---|---|---|
| `dashscope` | 否，复用 `api_key` | **否** | 用千问自带的联网搜索。结果是真实抓取的，但兼容模式不返回结构化来源，只有标题和摘要 |
| `duckduckgo` | 否 | **是** | 免 key 且**有链接**，走 `ddgs`（它自己聚合多个搜索源）。旧的 `duckduckgo-search` 对网络出口很敏感、一搜就 `202 Ratelimit`，换成 `ddgs` 后实测连搜 3 次都正常返回 |

搜索源挂掉时 `web_search` 不会抛异常，而是返回一句明确的"不可用、不要重试"给模型，
避免它换着措辞反复试同一个坏工具。要接 Tavily / 博查之类，实现一个 `SearchProvider`
子类再在 `build_provider` 里加一个分支即可。

### 会话历史

每轮的用户目标、思考、计划、工具结果、最终回答都落进 `agent.db`。重新打开页面时
自动回放**最近一次**会话，下面用一条分隔线标出"以上为上次会话"。正在跑任务时不回放，
避免和实时推送的事件重复。多会话切换是 Phase 2。

还有几个上限，键名与 `app/config.py` 的字段名一致，写进 `config.json` 即可覆盖：
`max_file_bytes` 单文件 >1MB 拒读、`read_output_limit` 返回给模型的内容 ≤8000 字符、
`cmd_output_limit` 命令输出 ≤2000 字符、`cmd_timeout` 命令 30 秒超时、
`max_write_chars` 单次写入 ≤64000 字符、`max_steps` 最多 20 步、
`max_memories` 注入提示词的记忆条数 ≤50、`max_verify_rounds` 验收最多重试 2 轮。

## 开发

```bat
.venv\Scripts\python.exe -m pytest -v
```

258 个测试，不联网、不需要 api_key（模型层用假的 chat model，agent 循环用脚本化的 FakeLLM）。

其中 32 个是**页面逻辑**的检查（`tests/frontend_checks.js`）：把 `index.html` 里的
`<script>` 抠出来，配一套最小 DOM 假件直接跑，不装任何 JS 依赖。装了 node 就跟着
`pytest` 一起跑，每条检查是一个独立用例；**没装 node 就整组跳过**，不会让 pytest 变红。

分层：`app/config.py` 配置 · `app/safety/` 沙箱与白名单 · `app/tools/` 八个工具与注册表 ·
`app/store/` SQLite · `app/llm/` 模型客户端 · `app/agent/` ReAct 图与上下文裁剪 ·
`app/web/` FastAPI 与静态页。各层接口明确，可独立测试。

**验收闸**（`verify_cmd`，默认留空 = 关闭）：模型说完成时，先跑一遍你配的验收命令
（比如 `pytest -q`）。不过就把失败输出喂回去继续修，最多 `max_verify_rounds` 轮；
还不过就如实说"改完了但没通过"，而不是硬说完成。只在**真动过东西**之后才跑
（判据复用安全阀的 `needs_confirmation`——需要确认的操作正是会改东西的那些），
纯问答和只读命令不会触发。

这道闸和确认闸位置类似、方向相反：确认闸拦危险操作，它拦"没验证就说完成"。
`run_command` 本来就在工具箱里、模型随时能跑测试，区别在于那是**可选的**，全凭自觉。

**错误分三类**（`app/agent/errors.py`）：可重试（限流 / 5xx / 网络）、可修正（参数不对 /
路径越界，喂回去让模型自己修）、终止（key 无效 / 模型名不对，重试没有意义）。给用户的是
一句能照着做的话，不是 `Error code: 401 - {'error': ...}`。认不出来的默认按"可修正"处理
——把任务打死是更坏的默认。

ReAct 循环由 **LangGraph 的 `StateGraph`** 驱动（`agent ⇄ tools` 一个环），模型层走
`langchain-openai` 的 `ChatOpenAI`。工具是 LangChain 工具对象，模型看到的函数定义由
pydantic 参数模型自动生成。**没有用预制的 `ToolNode`**——工具执行要过确认闸、走熔断、
按类型发不同事件，包一层比自己写更长。`max_steps` 仍然是「最多几轮模型调用」，
换算成 `recursion_limit` 时要乘 2（它数的是节点执行次数，一个回合两个节点）。

确认闸走 **`interrupt()` + `SqliteSaver`**：图停住、存 checkpoint、返回，页面回答后
`Command(resume=...)` 接着跑。因为恢复时**整个节点会从头重跑**，闸必须在任何副作用
之前、且一轮只有一个——这就是「一轮里的危险操作合成一张卡」的由来，不只是 UX。

`app/agent/loop.py` 的 `AgentRunner` 是 web 层用的接口（`start` / `resume`，各跑到下一个
停点）；`run_agent()` 是给测试和命令行的包装，用一个 `confirm` 回调把整轮跑完。

八个工具：`list_dir` `read_file` `search_in_files` `web_search` `update_plan` `save_memory`
（自动放行）、`write_file` `run_command`（过安全阀）。`fs.preview_write` 只给确认卡片生成
diff，没有注册成工具，模型看不到它。

**流程图**：[`docs/流程图.html`](docs/流程图.html) —— 分层依赖、任务时序（确认闸为什么不占线程）、
图的三个节点（agent / tools / verify）、安全阀判定树、事件类型对照。用浏览器打开，
图表需要联网加载一次 mermaid；加载不出来时会原样显示图表源码。

改代码后记得同步这里。**同步完用 mermaid 自己的解析器过一遍**——语法错在浏览器里
是渲染成一片空白而不是报错，肉眼看不出来。

## 已知限制

- 单会话、单任务串行。上一个任务没跑完（**含等着你确认**）时发新目标会被拒，
  否则两个 agent 会互相覆盖文件。
- 页面 1 秒轮询，没有流式打字机效果。
- 页面逻辑的检查跑在最小 DOM 假件上，不是真浏览器——CSS、布局、真实事件顺序测不到。
- checkpoint 只跨越确认闸，**不支持"关掉程序明天接着批"**：`AgentRunner` 和 `ToolSet`
  都在内存里，进程一退就没了，重启后那一轮无法恢复。要做的话得把这两样也持久化，
  并且不要在任务结束时删 checkpoint。
- `run_command` 用 `shell=True`。在"白名单外全部人工确认 + 用户看到完整命令 + cwd 锁死沙箱"
  三重前提下 MVP 可接受，Phase 2 收紧为 `shell=False` + 参数数组。
- 程序运行期间 `agent.db` 会被 sqlite 占用（删不掉也移不走），正常退出时释放。
  连接跨线程共享，所有读写都过同一把锁——没有锁的话退出时关连接会撞上 agent
  线程正在写，直接访问违规而不是抛异常。
- 默认的 `dashscope` 搜索**拿不到 URL**（兼容模式不返回结构化来源）。要链接就把
  `search_provider` 换成 `duckduckgo`。

## Phase 2

RAG、`fetch_url`、流式输出、打包 .exe、多会话管理 UI、撤销/回滚。
