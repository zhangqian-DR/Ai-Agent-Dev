"""direct 路：一次问答，不绑任何工具。

为什么不塞进那张图：它没有循环、没有工具、没有确认闸，也就不可能 interrupt，
硬套一个 StateGraph 只是给一次函数调用套了个壳。

这条路的价值不在省 token（实测 8 个工具的 schema 是 783 tokens/轮，占一次闲聊
请求的 68%，但折成钱可以忽略），而在**把「回答问题」和「能动手」物理隔开**——
没绑工具，模型就算想调 write_file 也调不出来。

对外故意做成和 ``AgentRunner`` 一样的 ``start`` / ``resume``，web 层那段驱动代码
因此一个字都不用改。
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.errors import explain
from app.agent.prompt import direct_prompt


class DirectRunner:
    def __init__(self, *, llm, cfg, emit):
        self.llm, self.cfg, self.emit = llm, cfg, emit

    def start(self, goal: str, memories: list, thread_id: str) -> dict:
        messages = [SystemMessage(content=direct_prompt(self.cfg.work_dir, memories)),
                    HumanMessage(content=goal)]
        # 步数条上仍然走一格：页面不必为这条路单开一套显示
        self.emit({"type": "step", "n": 1, "max": 1,
                   "chars": sum(len(m.content) for m in messages)})
        try:
            reply = self.llm.chat(messages, [])      # 空 tools —— 这条路的全部意义
        except Exception as e:
            msg = f"出错：{explain(e)}"
            self.emit({"type": "final", "content": msg, "ok": False})
            return {"done": True, "answer": msg}

        answer = reply.content or ""
        self.emit({"type": "final", "content": answer, "ok": True})
        return {"done": True, "answer": answer}

    def resume(self, approved: bool, thread_id: str) -> dict:
        # 没有闸就不会有人来恢复。真被调到，说明上游把路径算错了，
        # 与其静默返回一个空答案，不如响亮地炸掉。
        raise RuntimeError("direct 路没有确认闸，不应该被 resume")
