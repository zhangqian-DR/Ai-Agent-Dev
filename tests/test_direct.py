"""direct 路：一次问答，不绑任何工具。

这条路的价值不在省那几百个 token，而在**把「回答问题」和「能动手」物理隔开**——
没有绑工具，模型就算想调 write_file 也调不出来。
"""
from langchain_core.messages import AIMessage

from app.agent.direct import DirectRunner
from app.config import Config


class RecordingLLM:
    def __init__(self, text="我是一个本地 AI 助手。"):
        self.text = text
        self.seen_tools = "没调过"
        self.seen_msgs = None
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        self.seen_tools = tools
        self.seen_msgs = messages
        return AIMessage(content=self.text)


def _run(tmp_path, llm, goal="你能做什么", memories=()):
    cfg = Config("", "", "qwen-flash", tmp_path)
    events = []
    r = DirectRunner(llm=llm, cfg=cfg, emit=events.append).start(goal, list(memories), "t1")
    return r, events


def test_no_tools_are_bound(tmp_path):
    """这是 direct 存在的全部理由：绑不上工具，就不可能动手。"""
    llm = RecordingLLM()

    _run(tmp_path, llm)

    assert llm.seen_tools == [], f"direct 路不该绑任何工具，实际绑了 {llm.seen_tools}"


def test_one_call_and_done(tmp_path):
    llm = RecordingLLM("答案")

    r, events = _run(tmp_path, llm)

    assert llm.calls == 1, "问一次就够，没有循环"
    assert r == {"done": True, "answer": "答案"}
    assert [e["type"] for e in events] == ["step", "final"]
    assert events[-1]["ok"] is True


def test_uses_a_shorter_prompt_than_the_react_one(tmp_path):
    """ReAct 那份提示词在讲 8 个工具和逐步执行的纪律，direct 一条都用不上。
    留着既费 token 又会让模型以为自己能调工具。"""
    from app.agent.prompt import system_prompt

    llm = RecordingLLM()
    _run(tmp_path, llm)
    used = llm.seen_msgs[0].content
    react = system_prompt(tmp_path, [])

    assert len(used) < len(react)
    assert "update_plan" not in used, "别在没工具的路上提工具名"
    assert "write_file" not in used


def test_memories_still_reach_the_model(tmp_path):
    """记忆是跟着用户走的，不该因为换了条路就丢。"""
    llm = RecordingLLM()

    _run(tmp_path, llm, memories=[{"fact": "用户主要写 Java", "is_negative": False},
                                  {"fact": "别用 tab", "is_negative": True}])

    text = llm.seen_msgs[0].content
    assert "Java" in text
    assert "禁止" in text, "负向记忆的标记也要跟着走"


def test_model_failure_is_explained_not_dumped(tmp_path):
    """direct 也走错误三分——不能因为路径简单就退回糊原始报文。"""
    class Boom:
        def chat(self, messages, tools):
            e = RuntimeError("Error code: 401 - {'error': {}}")
            e.status_code = 401
            raise e

    r, events = _run(tmp_path, Boom())

    assert "api_key" in r["answer"] and "{'error'" not in r["answer"]
    assert events[-1]["ok"] is False


def test_resume_is_not_expected(tmp_path):
    """direct 路没有闸，永远不会有人来 resume。真被调到说明上游算错了路径。"""
    import pytest

    runner = DirectRunner(llm=RecordingLLM(), cfg=Config("", "", "m", tmp_path),
                          emit=lambda e: None)
    with pytest.raises(RuntimeError):
        runner.resume(True, "t1")
