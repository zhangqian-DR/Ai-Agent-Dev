import copy
import json

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


def _run(tmp_path, llm, confirm=lambda a: True):
    cfg, tools = _tools(tmp_path)
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


def _call(name, args, cid="1"):
    return {"content": None, "tool_calls": [{"id": cid, "name": name, "args": args}]}


def _done(text="完成"):
    return {"content": text, "tool_calls": []}


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

    tool_msgs = [m["content"] for msgs in llm.seen for m in msgs if m.get("role") == "tool"]
    assert any("重复" in t or "已经" in t for t in tool_msgs), \
        f"第二次重复时没有给模型任何提示：{tool_msgs}"


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
    llm = ScriptLLM({"content": "我先读文件", "tool_calls":
                     [{"id": "1", "name": "read_file", "args": {"path": "a.txt"}}]}, _done())
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
    """write_file 的文件内容在 tool_calls.arguments 里，是整段历史最大的负载。
    执行完之后模型不需要再看一遍，留着只会挤掉真正有用的上下文。"""
    big = "x" * 50_000
    llm = ScriptLLM(_call("write_file", {"path": "a.txt", "content": big}), _done())

    _run(tmp_path, llm)

    second = llm.seen[1]          # 第二次请求带的历史
    asst = [m for m in second if m.get("role") == "assistant" and m.get("tool_calls")]
    assert asst, "历史里应有带 tool_calls 的 assistant 消息"
    raw = asst[0]["tool_calls"][0]["function"]["arguments"]
    assert len(raw) < 2000, f"执行完的 arguments 仍有 {len(raw)} 字符"
    assert "a.txt" in raw, "路径这种短字段要留着，模型还需要知道自己改了哪个文件"
    json.loads(raw)               # 必须仍是合法 JSON，否则协议校验会失败
