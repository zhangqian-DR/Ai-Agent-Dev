import copy
import json

from langchain_core.messages import (AIMessage, HumanMessage, ToolMessage,
                                     convert_to_openai_messages)

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
    """按脚本依次返回；脚本用完后一直返回最后一条。记录每次收到的 messages。

    每次返回**一份新的拷贝**：真实模型每次都给新消息，而 add_messages 是按 id
    合并的——同一个对象返回两次，第二次会被当成「替换第一次」而不是追加，
    历史直接被搅乱。脚本重复播放最后一条时必然踩到。
    """

    def __init__(self, *script):
        self.script = list(script)
        self.seen = []
        self.n = 0

    def chat(self, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        item = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return copy.deepcopy(item)


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
    assert asked and asked[0]["actions"][0]["name"] == "write_file"
    assert (tmp_path / "x.txt").exists()


def test_confirm_rejected_skips_write(tmp_path):
    _run(tmp_path, ScriptLLM(_call("write_file", {"path": "x.txt", "content": "hi"}), _done("已取消")),
         confirm=lambda a: False)
    assert not (tmp_path / "x.txt").exists()


def _writes(*paths):
    return AIMessage(content="", tool_calls=[
        {"name": "write_file", "args": {"path": p, "content": p[0].upper()}, "id": str(i)}
        for i, p in enumerate(paths, 1)])


def test_one_gate_for_all_dangerous_calls_in_a_turn(tmp_path):
    """一轮里的危险操作一次问完，不逐条弹。

    这不只是 UX：阶段 4b 用 interrupt 恢复时，节点会**从头重跑**，所以闸必须
    在任何副作用之前、且一轮只有一个。逐条弹的话，恢复会把前面已经执行过的
    工具再执行一遍。
    """
    asked = []
    llm = ScriptLLM(_writes("a.txt", "b.txt"), _done())

    _run(tmp_path, llm, confirm=lambda a: asked.append(a) or True)

    assert len(asked) == 1, f"闸弹了 {len(asked)} 次"
    assert [x["name"] for x in asked[0]["actions"]] == ["write_file", "write_file"]
    assert "+A" in asked[0]["actions"][0]["preview"]
    assert (tmp_path / "a.txt").exists() and (tmp_path / "b.txt").exists()


def test_safe_calls_in_the_same_turn_are_not_gated(tmp_path):
    """自动放行的工具不该混进确认卡片，但仍要照常执行。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    asked = []
    llm = ScriptLLM(AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.txt"}, "id": "1"},
        {"name": "write_file", "args": {"path": "b.txt", "content": "B"}, "id": "2"}]),
        _done())

    _, events = _run(tmp_path, llm, confirm=lambda a: asked.append(a) or True)

    assert [x["name"] for x in asked[0]["actions"]] == ["write_file"]
    assert [e["name"] for e in events if e["type"] == "tool"] == ["read_file", "write_file"]


def test_rejecting_the_gate_skips_only_the_gated_calls(tmp_path):
    """拒绝只跳过过闸的那些，同一轮里自动放行的仍然执行。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    llm = ScriptLLM(AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.txt"}, "id": "1"},
        {"name": "write_file", "args": {"path": "b.txt", "content": "B"}, "id": "2"}]),
        _done("已取消"))

    _, events = _run(tmp_path, llm, confirm=lambda a: False)

    tools_ev = {e["name"]: e for e in events if e["type"] == "tool"}
    assert tools_ev["read_file"]["ok"] is True
    assert tools_ev["write_file"]["ok"] is False
    assert not (tmp_path / "b.txt").exists()


# ---------- 验收闸：改完东西不能只凭模型自己说「完成了」 ----------

def _verify_run(tmp_path, llm, verify_cmd, rounds=2):
    cfg, tools = _tools(tmp_path)
    cfg.verify_cmd = verify_cmd
    cfg.max_verify_rounds = rounds
    events = []
    ans = run_agent("目标", llm=llm, tools=tools, cfg=cfg,
                    emit=events.append, confirm=lambda a: True, memories=[])
    return ans, events


_PASS = "python -c \"pass\""
_FAIL = "python -c \"import sys; sys.exit(1)\""


def test_no_verify_cmd_keeps_the_old_behaviour(tmp_path):
    """没配验收命令就整个不启用——「随手写个脚本」这类任务没有测试可跑，
    不该被卡住。"""
    _, events = _verify_run(tmp_path, ScriptLLM(
        _call("write_file", {"path": "a.txt", "content": "x"}), _done("完成")), "")

    assert not [e for e in events if e["type"] == "tool" and "验收" in e["name"]]
    assert events[-1]["type"] == "final" and events[-1]["ok"] is True


def test_read_only_task_is_not_verified(tmp_path):
    """什么都没改就别跑验收——纯问答跑一遍测试既慢又莫名其妙。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    _, events = _verify_run(tmp_path, ScriptLLM(
        _call("read_file", {"path": "a.txt"}), _done("看完了")), _PASS)

    assert not [e for e in events if e["type"] == "tool" and "验收" in e["name"]]


def test_verify_runs_after_a_write_and_passes(tmp_path):
    """动过文件就得验收一次，过了才允许收尾。"""
    ans, events = _verify_run(tmp_path, ScriptLLM(
        _call("write_file", {"path": "a.txt", "content": "x"}), _done("改好了")), _PASS)

    checks = [e for e in events if e["type"] == "tool" and "验收" in e["name"]]
    assert len(checks) == 1 and checks[0]["ok"] is True
    assert ans == "改好了"
    assert events[-1]["type"] == "final" and events[-1]["ok"] is True


def test_failed_verify_is_fed_back_and_the_agent_keeps_going(tmp_path):
    """验收不过要把失败输出喂回去让它继续修，而不是就这么收尾。"""
    llm = ScriptLLM(
        _call("write_file", {"path": "a.txt", "content": "坏的"}),
        _done("我改好了"),                       # 第一次声称完成 → 验收会红
        _call("write_file", {"path": "a.txt", "content": "好的"}, cid="2"),
        _done("这次真好了"))
    cfg, tools = _tools(tmp_path)
    cfg.verify_cmd = _FAIL
    cfg.max_verify_rounds = 2
    events = []
    run_agent("目标", llm=llm, tools=tools, cfg=cfg,
              emit=events.append, confirm=lambda a: True, memories=[])

    fed = [m.content for msgs in llm.seen for m in msgs
           if isinstance(m, HumanMessage) and "验收" in m.content]
    assert fed, "失败输出没有喂回模型"
    assert "exit=1" in fed[0]
    assert llm.n > 2, "模型应当被叫回去继续改"


def test_verify_nudge_resets_the_loop_breaker(tmp_path):
    """验收失败会把模型叫回去重查，这时重读同一个文件是**正当**的。

    真机踩到过：验收一红，模型回头重读源文件想弄清哪里不对，读到第 3 次就被
    熔断当成死循环掐了。熔断防的是"原地打转"，而拿到新信息之后回头看不是
    打转——所以验收喂回失败时要把这两本账清零。verify_rounds 仍然兜着底，
    不会因此变成无限循环。
    """
    (tmp_path / "b.txt").write_text("不变的内容", encoding="utf-8")
    read_b = {"path": "b.txt"}
    llm = ScriptLLM(
        _call("read_file", read_b, cid="1"),                        # 累计 1
        _call("read_file", read_b, cid="2"),                        # 累计 2 → 提醒
        _call("write_file", {"path": "a.txt", "content": "x"}, cid="3"),
        _done("我改好了"),                                           # → 验收红，喂回
        _call("read_file", read_b, cid="4"),                        # 不清零的话这里第 3 次 → 熔断
        _done("这次真好了"))                                          # → 验收再红 → 收场

    ans, events = _verify_run(tmp_path, llm, _FAIL, rounds=2)

    assert "死循环" not in ans, "验收把模型叫回来重查，不该被熔断误杀"
    assert "验收未通过" in ans
    assert len([e for e in events if e["type"] == "tool" and "验收" in e["name"]]) == 2


def test_verify_gives_up_after_the_round_limit_and_says_so(tmp_path):
    """一直不过就如实说「改完了但没通过」，不能硬说完成。"""
    llm = ScriptLLM(_call("write_file", {"path": "a.txt", "content": "x"}), _done("完成了"))
    ans, events = _verify_run(tmp_path, llm, _FAIL, rounds=2)

    checks = [e for e in events if e["type"] == "tool" and "验收" in e["name"]]
    assert len(checks) == 2, f"应当只试 2 轮，实际 {len(checks)}"
    final = events[-1]
    assert final["ok"] is False, "没通过验收不能标成绿色成功"
    assert "验收未通过" in ans


class _Boom:
    """调用模型时抛一个带 status_code 的错，模仿 SDK 的行为。"""

    def __init__(self, code):
        self.code = code
        self.n = 0

    def chat(self, messages, tools):
        self.n += 1
        e = RuntimeError("Error code: %s - {'error': {'message': 'nope'}}" % self.code)
        e.status_code = self.code
        raise e


def test_terminal_model_error_ends_cleanly_with_actionable_text(tmp_path):
    """key 无效时要给一句能照着做的话，而不是把 SDK 的原始报文糊到页面上。

    原来这个异常会一路冒到 web 层，变成
    「出错：RuntimeError: Error code: 401 - {'error': {...}}」——用户看不出
    该去改什么，只会一遍遍重发。
    """
    llm = _Boom(401)
    ans, events = _run(tmp_path, llm)

    assert "api_key" in ans and "config.json" in ans
    assert "{'error'" not in ans, "不该把原始报文糊出来"
    final = [e for e in events if e["type"] == "final"][-1]
    assert final["ok"] is False, "这是错误收场，页面要标红"
    assert llm.n == 1, "终止类错误不该再试第二次"


def test_retryable_model_error_says_it_may_work_later(tmp_path):
    """限流和 key 无效得说得不一样：一个值得稍后再来，一个改配置才行。"""
    ans, _ = _run(tmp_path, _Boom(429))

    assert "稍后" in ans or "限流" in ans
    assert "api_key" not in ans


def test_sandbox_error_tells_the_model_to_stop_trying_outside(tmp_path):
    """光回一句「路径超出工作目录」，模型可能换个同样越界的路径再试。"""
    _, events = _run(tmp_path, ScriptLLM(
        _call("read_file", {"path": "..\\..\\evil.txt"}), _done("换个路径，完成")))

    err = [e["result"] for e in events if e["type"] == "tool"][0]
    assert "工作目录之外" in err or "不要再试" in err, err


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
    # 断言的是「越界这件事被讲清楚并喂了回去」，不是某个异常类名——
    # 类名是实现细节，讲不讲得清楚才是模型能不能自己修的关键
    assert tool_events and "工作目录" in tool_events[0]["result"]
    assert not (tmp_path.parent.parent / "evil.txt").exists()


def test_missing_tool_argument_is_fed_back_not_fatal(tmp_path):
    """模型漏传必填参数时，错误要喂回去让它补，不能让整个会话陪葬。

    具体是哪个异常类型不重要（工具对象化之后由 pydantic 校验抛出，不再是
    KeyError），重要的是它变成一条工具结果、并且点明缺了什么。
    """
    ans, events = _run(tmp_path, ScriptLLM(
        _call("write_file", {"path": "x.txt"}),
        _done("补上参数重来，完成")))
    assert "完成" in ans
    errs = [e["result"] for e in events if e["type"] == "tool" and "工具执行出错" in e["result"]]
    assert errs, "漏参数没有作为工具结果喂回去"
    assert "content" in errs[0], f"报错没点明缺的是哪个参数：{errs[0]!r}"


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


def test_max_steps_gives_a_clean_ending(tmp_path):
    """步数用尽要有个明确的收场。换成图之后这条路是捕获 GraphRecursionError，
    漏捕的话异常会一路冒到 web 层，变成一条「出错：」把会话打死。

    每次换不同的参数，免得先被熔断拦下——这里要测的是步数上限本身。
    """
    class Endless:
        def __init__(self):
            self.n = 0

        def chat(self, messages, tools):
            self.n += 1
            return AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": f"no{self.n}.txt"},
                 "id": str(self.n)}])

    ans, events = _run(tmp_path, Endless())

    assert ans == "已达最大步数，停止。"
    assert events[-1]["type"] == "final" and events[-1]["ok"] is True
    assert len([e for e in events if e["type"] == "step"]) == 20, \
        "max_steps 仍应表示「最多几轮模型调用」"


def test_same_tool_over_and_over_gets_nudged(tmp_path):
    """同一个工具反复变着参数用，两把尺子都看不见——它们都要求参数相同。

    真机上撞到过：让它分析安全问题，它连着 17 次 search_in_files（每次换个关键词），
    一个文件都没读，直接烧到步数上限、最后只吐出「已达最大步数」。这不是死循环，
    是**换着姿势原地转**，得单独有把尺子量。
    """
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    seq = [_call("search_in_files", {"pattern": f"关键词{i}"}, cid=str(i))
           for i in range(10)] + [_done("好了")]
    llm = ScriptLLM(*seq)

    _, events = _run(tmp_path, llm)

    nudged = [e for e in events
              if e["type"] == "tool" and "换一种" in (e.get("result") or "")]
    assert nudged, "连着十次同一个工具都没被提醒"
    assert "search_in_files" in nudged[0]["result"]


def test_reading_several_files_in_a_row_is_fine(tmp_path):
    """但连读几个文件是正当的——阈值不能低到把正常干活也拦了。"""
    for i in range(4):
        (tmp_path / f"f{i}.txt").write_text("x", encoding="utf-8")
    seq = [_call("read_file", {"path": f"f{i}.txt"}, cid=str(i)) for i in range(4)] + [_done()]

    _, events = _run(tmp_path, ScriptLLM(*seq))

    assert not [e for e in events
                if e["type"] == "tool" and "换一种" in (e.get("result") or "")]


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
