"""slow 路的反思回环：synth 写完，critic 评一次，不合格就带着意见回去补。

两条原则：

1. **有机器判据就不用 LLM 评审**。配了 verify_cmd 的项目，测试绿不绿是客观的，
   比让模型评自己可靠得多；那条路已经由 verify 节点管着，critic 不重复插手。
2. **评审要有事实可依**。光问模型「你答得全不全」，它多半说全。所以把机器能查
   的东西一并给它：读过哪些文件、工作目录里还有哪些没读过。
"""
import copy

from langchain_core.messages import AIMessage

from app.agent.loop import run_agent
from app.config import Config
from app.store.db import Store
from app.tools.registry import build_tools
from app.tools.web import SearchProvider


class FakeProvider(SearchProvider):
    def search(self, q, max_results=5):
        return []


class ScriptLLM:
    def __init__(self, *script):
        self.script = list(script)
        self.seen = []
        self.n = 0

    def chat(self, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        item = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        return copy.deepcopy(item)


def _plan(*steps):
    return AIMessage(content="", tool_calls=[
        {"name": "update_plan", "args": {"steps": list(steps), "current": 1}, "id": "p1"}])


def _call(name, args, cid="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def _say(t):
    return AIMessage(content=t)


def _run(tmp_path, llm, verify_cmd="", rounds=1):
    cfg = Config("", "", "qwen-max", tmp_path, verify_cmd=verify_cmd)
    cfg.max_critic_rounds = rounds
    tools = build_tools(cfg, Store(str(tmp_path / "t.db")), FakeProvider())
    events = []
    ans = run_agent("分析一下", llm=llm, tools=tools, cfg=cfg, emit=events.append,
                    confirm=lambda a: True, memories=[], path="slow")
    return ans, events


def test_critic_passes_and_the_answer_stands(tmp_path):
    llm = ScriptLLM(_plan("看看"), _say("做完了"), _say("汇总结论"), _say("通过"))

    ans, events = _run(tmp_path, llm)

    assert ans == "汇总结论"
    assert len([e for e in events if e["type"] == "final"]) == 1


def test_critic_sends_it_back_with_the_critique(tmp_path):
    """不合格时要把**评审意见**带回去，而不是只说一句「重做」。"""
    llm = ScriptLLM(_plan("看看"),
                    _say("做完了"),
                    _say("草率的结论"),
                    _say("不合格：你没有读 b.txt"),   # critic
                    _say("补读完了"),
                    _say("完整的结论"),
                    _say("通过"))

    ans, _ = _run(tmp_path, llm, rounds=2)

    # 只找喂回去的**意见**，别把评审的提问也算进来（两者都含「评审」二字）
    fed = [m.content for msgs in llm.seen for m in msgs
           if getattr(m, "type", "") == "human" and m.content.startswith("评审意见：")]
    assert fed, "评审意见没有喂回去"
    assert "b.txt" in fed[0], "带回去的必须是具体意见，不是一句重做"
    assert ans == "完整的结论"


def test_critic_gives_up_after_the_round_limit(tmp_path):
    """一直不合格也要收场，并且如实说——不能无限转。"""
    llm = ScriptLLM(_plan("看看"), _say("做完了"), _say("结论"), _say("不合格：还差得远"))

    ans, events = _run(tmp_path, llm, rounds=1)

    assert "评审未通过" in ans
    assert events[-1]["ok"] is False


def test_machine_judge_wins_over_llm_critic(tmp_path):
    """配了验收命令就不再起 LLM 评审——测试绿不绿是客观的，比模型评自己可靠。"""
    llm = ScriptLLM(_plan("看看"),
                    _call("write_file", {"path": "a.txt", "content": "x"}),
                    _say("改完了"),
                    _say("汇总"))

    ans, events = _run(tmp_path, llm, verify_cmd="python -c \"pass\"")

    assert [e for e in events if e["type"] == "tool" and "验收" in e["name"]]
    assert ans == "汇总"
    assert llm.n == 4, f"不该多出一次 LLM 评审，实际调了 {llm.n} 次"


def test_critic_is_told_which_files_were_never_read(tmp_path):
    """光问模型「答得全不全」，它多半说全。给它机器查得到的事实。"""
    (tmp_path / "读过的.txt").write_text("x", encoding="utf-8")
    (tmp_path / "没读的.txt").write_text("y", encoding="utf-8")
    llm = ScriptLLM(_plan("看看"),
                    _call("read_file", {"path": "读过的.txt"}),
                    _say("看完了"),
                    _say("汇总"),
                    _say("通过"))

    _run(tmp_path, llm)

    critique_prompt = llm.seen[-1][-1].content
    assert "没读的.txt" in critique_prompt, f"没把未读文件告诉评审：{critique_prompt!r}"
    assert "读过的.txt" in critique_prompt


def test_fast_path_has_no_critic(tmp_path):
    """反思回环是 slow 路专有的，fast 一个字不变。"""
    cfg = Config("", "", "qwen-plus", tmp_path)
    tools = build_tools(cfg, Store(str(tmp_path / "t2.db")), FakeProvider())
    llm = ScriptLLM(_say("直接答完"))

    ans = run_agent("干活", llm=llm, tools=tools, cfg=cfg, emit=lambda e: None,
                    confirm=lambda a: True, memories=[], path="fast")

    assert ans == "直接答完" and llm.n == 1
