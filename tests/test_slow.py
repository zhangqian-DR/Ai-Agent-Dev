"""slow 路：规划 → 执行 → 汇总。

和 fast 的区别只在两头：开头多一个 planner，结尾多一个 synth。中间的执行完全复用
`agent ⇄ tools`——三道闸（确认、熔断、验收）因此自动生效，不必在新节点里重接一遍。
并行执行是后一个阶段的事，这里仍是串行。
"""
import copy

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.loop import run_agent
from app.config import Config
from app.store.db import Store
from app.tools.registry import build_tools
from app.tools.web import SearchProvider


class FakeProvider(SearchProvider):
    def search(self, q, max_results=5):
        return []


class ScriptLLM:
    """按脚本依次返回；脚本用完后一直返回最后一条。每次返回新拷贝——
    add_messages 按 id 合并，同一个对象返回两次会变成「替换」而不是「追加」。"""

    def __init__(self, *script):
        self.script = list(script)
        self.seen = []
        self.n = 0

    def chat(self, messages, tools):
        self.seen.append((copy.deepcopy(messages), tools))
        item = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return copy.deepcopy(item)


def _plan(*steps, cid="p1"):
    return AIMessage(content="", tool_calls=[
        {"name": "update_plan", "args": {"steps": list(steps), "current": 1}, "id": cid}])


def _call(name, args, cid="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def _say(text):
    return AIMessage(content=text)


def _run(tmp_path, llm, path="slow", confirm=lambda a: True):
    cfg = Config("", "", "qwen-max", tmp_path)
    tools = build_tools(cfg, Store(str(tmp_path / "t.db")), FakeProvider())
    events = []
    ans = run_agent("分析一下这个目录", llm=llm, tools=tools, cfg=cfg,
                    emit=events.append, confirm=confirm, memories=[], path=path)
    return ans, events, tools


def test_planner_runs_first_and_fills_the_panel(tmp_path):
    """slow 路开头先出计划，而且直接复用现成的 update_plan 与计划面板。"""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    llm = ScriptLLM(_plan("读 a.txt", "总结"),
                    _call("read_file", {"path": "a.txt"}),
                    _say("读完了"),
                    _say("汇总：a.txt 里是 hi"))

    ans, events, tools = _run(tmp_path, llm)

    plans = [e for e in events if e["type"] == "plan"]
    assert plans, "slow 路没有产出计划"
    assert plans[0]["steps"] == ["读 a.txt", "总结"]
    assert tools.plan == ["读 a.txt", "总结"]


def test_planner_only_gets_the_plan_tool(tmp_path):
    """规划那一步不该让模型顺手就动手——只绑 update_plan，它想调 write_file 也调不出来。"""
    llm = ScriptLLM(_plan("看看"), _say("做完了"), _say("汇总"))

    _run(tmp_path, llm)

    first_tools = llm.seen[0][1]
    assert [t.name for t in first_tools] == ["update_plan"], \
        f"planner 绑了别的工具：{[t.name for t in first_tools]}"


def test_execution_gets_all_the_tools_back(tmp_path):
    """规划完就该正常干活，工具要全给回来。"""
    llm = ScriptLLM(_plan("看看"), _say("做完了"), _say("汇总"))

    _run(tmp_path, llm)

    exec_tools = llm.seen[1][1]
    assert len(exec_tools) == 8


def test_final_answer_comes_from_the_synthesizer(tmp_path):
    """收尾那句是 synth 写的，不是执行阶段最后一条消息。"""
    llm = ScriptLLM(_plan("看看"),
                    _say("我执行完了"),          # 执行阶段的收尾话
                    _say("【汇总】完整结论"))     # synth 写的

    ans, events, _ = _run(tmp_path, llm)

    assert ans == "【汇总】完整结论"
    finals = [e for e in events if e["type"] == "final"]
    assert len(finals) == 1, f"final 只该发一次，实际 {len(finals)} 次"
    assert finals[0]["content"] == "【汇总】完整结论"


def test_synthesizer_binds_no_tools(tmp_path):
    """汇总阶段只要一段话，给了工具反而可能又跑去干活。"""
    llm = ScriptLLM(_plan("看看"), _say("做完了"), _say("汇总"))

    _run(tmp_path, llm)

    assert llm.seen[-1][1] == [], "synth 不该绑工具"


def test_synthesizer_sees_more_history_than_a_normal_call(tmp_path):
    """synth 存在的理由：长任务里早期观察会被裁掉，收尾那句就成了「只看得见
    最近几步」的结论。它用更大的窗口通读全程。"""
    (tmp_path / "a.txt").write_text("x" * 9000, encoding="utf-8")
    llm = ScriptLLM(_plan("读三次"),
                    _call("read_file", {"path": "a.txt"}, cid="1"),
                    _call("read_file", {"path": "a.txt"}, cid="2"),
                    _say("看完了"),
                    _say("汇总"))

    _run(tmp_path, llm)

    exec_windows = [len(m) for m, t in llm.seen[1:-1]]
    synth_window = len(llm.seen[-1][0])
    assert synth_window >= max(exec_windows), \
        f"synth 看到的历史（{synth_window} 条）不比执行阶段（{exec_windows}）多"


def test_fast_path_has_neither_planner_nor_synth(tmp_path):
    """fast 路一个字都不该变。"""
    llm = ScriptLLM(_say("直接答完"))

    ans, events, _ = _run(tmp_path, llm, path="fast")

    assert ans == "直接答完"
    assert llm.n == 1, "fast 路只该调一次模型"
    assert not [e for e in events if e["type"] == "plan"]


def test_planner_that_skips_the_plan_does_not_break_the_run(tmp_path):
    """模型不肯出计划时照样能干活——规划是加分项，不是必经关卡。"""
    llm = ScriptLLM(_say("我不列计划"), _say("直接做完了"), _say("汇总"))

    ans, events, _ = _run(tmp_path, llm)

    assert ans == "汇总"
    assert not [e for e in events if e["type"] == "plan"]
