from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI


class LLMClientError(RuntimeError):
    pass


def _to_langchain(messages: list[dict]) -> list:
    """OpenAI 那套 dict 消息 → LangChain 消息对象。

    转换只发生在这一层：``loop.py`` 的历史、``context.py`` 的分组裁剪、落库的
    格式仍然全是 dict，换模型框架不该把这些一起卷进去。

    注意 ``tool_call_id`` 必须跟着 tool 消息走——丢了模型侧直接 400，这正是
    context.py 整套「按组裁剪、绝不产生孤儿 tool 消息」逻辑在防的事。
    """
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(HumanMessage(content=content))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id") or ""))
        else:                                    # assistant
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                # LangChain 的 tool_calls 里 args 是 dict，OpenAI 那边是 JSON 字符串
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                calls.append({"name": fn.get("name") or "", "args": args,
                              "id": tc.get("id") or ""})
            out.append(AIMessage(content=content, tool_calls=calls))
    return out


class LLMClient:
    """OpenAI 兼容接口的薄封装（千问 DashScope），内部走 LangChain ``ChatOpenAI``。

    对外仍是 ``chat(messages, tools) -> {"content", "tool_calls"}``：上层的
    ReAct 循环和它的测试假件都是按这个签名写的，换框架不必惊动它们。
    换 Claude / GPT 只需要改 config.json 的 base_url 与 model。
    """

    def __init__(self, cfg, timeout: float = 90.0, max_retries: int = 2):
        self.cfg = cfg
        # 超时必须设：agent 跑在后台线程里，请求挂住的话整个会话就卡死了，
        # 页面只会一直显示"执行中"，用户没有任何办法取消。
        self._model = ChatOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "EMPTY",
            model=cfg.model,
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        if not self.cfg.api_key:
            raise LLMClientError(
                "config.json 里的 api_key 是空的。去阿里云百炼申请一个千问 key 填进去，"
                "或把 base_url 指向别的 OpenAI 兼容服务。")
        # tools 为空时不能 bind——绑一个空列表，部分兼容实现会报 400
        model = self._model.bind_tools(tools) if tools else self._model
        msg = model.invoke(_to_langchain(messages))

        calls = [{"id": tc.get("id") or "", "name": tc.get("name") or "",
                  "args": tc.get("args") or {}}
                 for tc in (msg.tool_calls or [])]
        # 参数不是合法 JSON 时，LangChain 把这次调用放进 invalid_tool_calls 而不是
        # tool_calls。两边都要读：漏掉的话这次调用凭空消失，模型等不到任何反馈就
        # 卡住了。给空参数让工具层报错，错误喂回模型触发反思，比整个会话中断好。
        calls += [{"id": tc.get("id") or "", "name": tc.get("name") or "", "args": {}}
                  for tc in (getattr(msg, "invalid_tool_calls", None) or [])]
        return {"content": msg.content, "tool_calls": calls}
