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

确认时 agent 线程是真的停在那里等。等待期间页面超过 30 秒没有轮询（多半是被关掉了），
按「拒绝」处理并继续往下跑——否则线程会永久挂着，运行权也就永远释放不出来。

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

### 联网搜索选哪个

| 值 | 需要额外 key | 有 URL | 说明 |
|---|---|---|---|
| `dashscope` | 否，复用 `api_key` | **否** | 用千问自带的联网搜索。结果是真实抓取的，但兼容模式不返回结构化来源，只有标题和摘要 |
| `duckduckgo` | 否 | 是 | 免 key 且有链接，但对网络出口敏感——被风控时所有 backend 都返回 `202 Ratelimit`，换 UA、换版本、等一等都没用 |

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
`max_write_chars` 单次写入 ≤64000 字符、`max_steps` 最多 20 步。

## 开发

```bat
.venv\Scripts\python.exe -m pytest -v
```

120 个测试，不联网、不需要 api_key（模型层用 monkeypatch，agent 循环用脚本化的 FakeLLM）。

分层：`app/config.py` 配置 · `app/safety/` 沙箱与白名单 · `app/tools/` 八个工具与注册表 ·
`app/store/` SQLite · `app/llm/` 模型客户端 · `app/agent/` ReAct 循环与上下文裁剪 ·
`app/web/` FastAPI 与静态页。各层接口明确，可独立测试。

八个工具：`list_dir` `read_file` `search_in_files` `web_search` `update_plan` `save_memory`
（自动放行）、`write_file` `run_command`（过安全阀）。`fs.preview_write` 只给确认卡片生成
diff，没有注册成工具，模型看不到它。

**流程图**：[`docs/流程图.html`](docs/流程图.html) —— 分层依赖、任务时序（确认闸的线程阻塞）、
`run_agent` 单步控制流、安全阀判定树。用浏览器打开，图表需要联网加载一次 mermaid；
加载不出来时会原样显示图表源码。改代码后记得同步。

## 已知限制

- 单会话、单任务串行。上一个任务没跑完时发新目标会被拒（否则两个 agent 会互相覆盖文件）。
- 页面 1 秒轮询，没有流式打字机效果。
- `run_command` 用 `shell=True`。在"白名单外全部人工确认 + 用户看到完整命令 + cwd 锁死沙箱"
  三重前提下 MVP 可接受，Phase 2 收紧为 `shell=False` + 参数数组。
- 程序运行期间 `agent.db` 会被 sqlite 占用（删不掉也移不走），正常退出时释放。
  连接跨线程共享，所有读写都过同一把锁——没有锁的话退出时关连接会撞上 agent
  线程正在写，直接访问违规而不是抛异常。
- `duckduckgo-search` 还锁在 5.3.1。当初锁它是因为 6.x 的 Rust 扩展没有 py38 wheel，
  现在跑在 3.12 上，这个理由已经不成立，可以升，只是还没升。

## Phase 2

RAG、`fetch_url`、流式输出、打包 .exe、多会话管理 UI、撤销/回滚。
