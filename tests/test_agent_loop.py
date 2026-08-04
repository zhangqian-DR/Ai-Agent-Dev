import copy
import json

from langchain_core.messages import AIMessage, ToolMessage, convert_to_openai_messages

from app.agent.loop import run_agent
from app.config import Config
from app.store.db import Store
from app.tools.registry import build_tools
from app.tools.web import SearchProvider


class FakeProvider(SearchProvider):
    def search(self, q, max_results=5):
        return []


def _tools(tmp_path):
    cfg = Config("", "", "qwen-plus", tmp_path)
    return cfg, build_tools(cfg, Store(str(tmp_path / "t.db")), FakeProvider())


class VaryingOutputTools:
    """每次返回都不一样的工具——模拟 run_command 那种带耗时/时间戳的输出。"""

    def __init__(self):
        self.calls = 0
        self.plan, self.plan_current = [], 0

    def tools(self):
        return []

    def preview(self, name, args):
        return ""

    def execute(self, name, args):
        self.calls += 1
        return f"[exit=1]\n1 failed in {self.calls}.0{self.calls}s"


def _run(tmp_path, llm, confirm=lambda a: True, tools=None):
    cfg, built = _tools(tmp_path)
    tools = built if tools is None else tools
    events = []
    ans = run_agent("目标", llm=llm, tools=tools, cfg=cfg,
                    emit=events.append, confirm=confirm, memories=[])
    return ans, events


class ScriptLLM:
    """按脚本依次返回；脚本用完后一直返回最后一条。记录每次收到的 messages。"""

    def __init__(self, *script):
        self.script = list(script)
        self.seen = []
        self.n = 0

    def chat(self, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        item = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return item


def _call(name, args, cid="1", content=""):
    return AIMessage(content=content,
                     tool_calls=[{"name": name, "args": args, "id": cid}])


def _done(text="完成"):
    return AIMessage(content=text)


# ---------- 原有行为不能破坏 ----------

def test_agent_runs_tool_then_answers(tmp_path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    ans, events = _run(tmp_path, ScriptLLM(_call("read_file", {"path": "a.txt"}), _done("文件内容是 hi，完成。")))
    assert "完成" in ans
    assert any(e["type"] == "tool" for e in events)


def test_agent_requests_confirmation_for_write(tmp_path):
    asked = []
    _run(tmp_path, ScriptLLM(_call("write_file", {"path": "x.txt", "content": "hi"}), _done("已写入，完成")),
         confirm=lambda a: asked.append(a) or True)
    assert asked and asked[0]["name"] == "write_file"
    assert (tmp_path / "x.txt").exists()


def test_confirm_rejected_skips_write(tmp_path):
    _run(tmp_path, ScriptLLM(_call("write_file", {"path": "x.txt", "content": "hi"}), _done("已取消")),
         confirm=lambda a: False)
    assert not (tmp_path / "x.txt").exists()


def test_out_of_sandbox_write_is_fed_back_not_fatal(tmp_path):
    """越界的 write_file 只该被挡下来喂回模型，不该打死整个任务。

    生成确认卡片的 preview 也要过沙箱，它抛的 SandboxError 原来在 try 之外，
    一路冒到 web 层变成"出错：…"，会话就此终止——而同样越界的 read_file
    只是返回一句错误让模型反思。同一类错误，两种下场。
    """
    ans, events = _run(tmp_path, ScriptLLM(
        _call("write_file", {"path": "..\\..\\evil.txt", "content": "x"}),
        _done("那我换个路径，完成")))
    assert "完成" in ans
    tool_events = [e for e in events if e["type"] == "tool"]
    assert tool_events and "SandboxError" in tool_events[0]["result"]
    assert not (tmp_path.parent.parent / "evil.txt").exists()


def test_missing_tool_argument_is_fed_back_not_fatal(tmp_path):
    """模型漏传 content 时是 KeyError，同样不能让整个会话陪葬。"""
    ans, events = _run(tmp_path, ScriptLLM(
        _call("write_file", {"path": "x.txt"}),
        _done("补上参数重来，完成")))
    assert "完成" in ans
    assert any(e["type"] == "tool" and "KeyError" in e["result"] for e in events)


# ---------- 缺陷一：熔断只数异常，工具返回错误字符串时不触发 ----------

def test_circuit_breaker_trips_on_repeated_failure(tmp_path):
    """read_file 读不存在的文件时是【返回错误字符串】而不是抛异常。
    只按异常计数的话熔断永远不触发，agent 会一直重复到 max_steps。"""
    llm = ScriptLLM(_call("read_file", {"path": "nope.txt"}))   # 永远重复同一个调用

    ans, events = _run(tmp_path, llm)

    assert "已达最大步数" not in ans, "熔断没生效，一路跑到了步数上限"
    assert llm.n < 8, f"模型被调用了 {llm.n} 次，熔断来得太晚"
    assert any(k in ans for k in ("多次", "循环", "终止")), f"终止原因不明确：{ans}"


def test_circuit_breaker_nudges_before_terminating(tmp_path):
    """第 2 次拿到完全相同的结果时，要把这件事明确告诉模型（强制反思），
    而不是等到第 3 次直接掐断——原计划写了这一档，代码里没有对应分支。"""
    llm = ScriptLLM(_call("read_file", {"path": "nope.txt"}))

    _run(tmp_path, llm)

    tool_msgs = [m.content for msgs in llm.seen for m in msgs if isinstance(m, ToolMessage)]
    assert any("重复" in t or "已经" in t for t in tool_msgs), \
        f"第二次重复时没有给模型任何提示：{tool_msgs}"


def test_rejected_tool_event_carries_a_flag(tmp_path):
    """页面原来靠 result 以「用户拒绝」开头判断这步被跳过了——工具真返回一句
    以那三个字开头的正常结果就会被误标。改成事件里带明确标志。"""
    _, events = _run(tmp_path, ScriptLLM(
        _call("write_file", {"path": "x.txt", "content": "hi"}), _done("已取消")),
        confirm=lambda a: False)

    tool = [e for e in events if e["type"] == "tool"][0]
    assert tool["ok"] is False


def test_normal_tool_event_is_marked_ok(tmp_path):
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    _, events = _run(tmp_path, ScriptLLM(_call("read_file", {"path": "a.txt"}), _done()))

    tool = [e for e in events if e["type"] == "tool"][0]
    assert tool["ok"] is True


def test_final_event_is_marked_ok(tmp_path):
    """正常回答要带 ok=True——页面据此决定画绿框还是红框，
    而不是看这段话是不是以「出错：」开头。"""
    _, events = _run(tmp_path, ScriptLLM(_done("出错：这三个字开头的正常回答")))

    assert [e for e in events if e["type"] == "final"][0]["ok"] is True


def test_breaker_trips_when_only_the_output_varies(tmp_path):
    """熔断按「工具+参数+结果」计数，而 run_command 的输出常带耗时/时间戳，
    逐字比对永远不相等——熔断对它形同虚设，只能一路空跑到步数上限。
    连着跑同一条命令本身就说明没在推进，不该指望输出一模一样。"""
    tools = VaryingOutputTools()
    llm = ScriptLLM(_call("run_command", {"cmd": "pytest"}))    # 永远重复同一条命令

    ans, _ = _run(tmp_path, llm, tools=tools)

    assert "已达最大步数" not in ans, "熔断没生效，一路跑到了步数上限"
    assert tools.calls < 8, f"命令被跑了 {tools.calls} 次，熔断来得太晚"


def test_different_results_do_not_trip_breaker(tmp_path):
    """结果每次都不同 = agent 在推进，不该被熔断误杀。"""
    (tmp_path / "log.txt").write_text("start\n", encoding="utf-8")
    seq = [_call("run_command", {"cmd": f"echo {i}"}, cid=str(i)) for i in range(5)] + [_done()]
    ans, _ = _run(tmp_path, ScriptLLM(*seq), confirm=lambda a: True)
    assert "完成" in ans


def test_final_answer_is_not_emitted_twice(tmp_path):
    """没有 tool_calls 时那段 content 就是最终回答，只该以 final 出现一次。
    原来 assistant 和 final 都发，页面上同一段话显示两遍，数据库里也存两条。"""
    _, events = _run(tmp_path, ScriptLLM(_done("这是最终回答")))

    kinds = [e["type"] for e in events if e["type"] in ("assistant", "final")]
    assert kinds == ["final"], f"最终回答被发了 {len(kinds)} 次：{kinds}"


def test_thinking_text_still_emitted_when_calling_tools(tmp_path):
    """但有工具调用时，那段思考文字仍然要显示——它不是最终回答。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    llm = ScriptLLM(_call("read_file", {"path": "a.txt"}, content="我先读文件"), _done())
    _, events = _run(tmp_path, llm)
    assert any(e["type"] == "assistant" and e["content"] == "我先读文件" for e in events)


def test_plan_event_carries_current(tmp_path):
    """面板要靠模型自己报的步号，而不是数工具调用次数——两者粒度对不上。"""
    llm = ScriptLLM(
        _call("update_plan", {"steps": ["查资料", "写代码", "跑起来"], "current": 1}),
        _call("update_plan", {"steps": ["查资料", "写代码", "跑起来"], "current": 3}, cid="2"),
        _done())
    _, events = _run(tmp_path, llm)

    plans = [e for e in events if e["type"] == "plan"]
    assert [p["current"] for p in plans] == [1, 3]
    assert plans[0]["steps"] == ["查资料", "写代码", "跑起来"]


def test_step_events_report_progress_and_context(tmp_path):
    """页面要能显示「第几步 / 上下文占用」，这两个数只有循环里知道。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    _, events = _run(tmp_path, ScriptLLM(_call("read_file", {"path": "a.txt"}), _done()))

    steps = [e for e in events if e["type"] == "step"]
    assert [s["n"] for s in steps] == [1, 2]
    assert all(s["max"] == 20 for s in steps)
    assert steps[1]["chars"] > steps[0]["chars"]   # 历史在增长


# ---------- 缺陷二：执行完的超大 arguments 一直占着历史 ----------

def test_large_tool_arguments_shrunk_after_execution(tmp_path):
    """write_file 的文件内容在 tool_calls 的参数里，是整段历史最大的负载。
    执行完之后模型不需要再看一遍，留着只会挤掉真正有用的上下文。

    断言落在**真正发出去的 wire 形态**上而不是内存里的 dict：真实模型回复会在
    additional_kwargs 里另存一份原始 JSON，只压 tool_calls 的话那份还在，
    压缩完全白做——这条只有按 wire 量才测得出来。
    """
    big = "x" * 50_000
    reply = _call("write_file", {"path": "a.txt", "content": big})
    # 补上真实回复才有的那份原始副本，假件默认不带
    reply.additional_kwargs = {"tool_calls": [
        {"id": "1", "type": "function",
         "function": {"name": "write_file",
                      "arguments": json.dumps({"path": "a.txt", "content": big})}}]}
    llm = ScriptLLM(reply, _done())

    _run(tmp_path, llm)

    second = llm.seen[1]          # 第二次请求带的历史
    asst = [m for m in second if isinstance(m, AIMessage) and m.tool_calls]
    assert asst, "历史里应有带 tool_calls 的 assistant 消息"

    wire = json.dumps(convert_to_openai_messages(asst), ensure_ascii=False)
    assert len(wire) < 2000, f"执行完的参数仍占 {len(wire)} 字符"
    assert "a.txt" in wire, "路径这种短字段要留着，模型还需要知道自己改了哪个文件"
