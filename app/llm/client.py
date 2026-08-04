from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class LLMClientError(RuntimeError):
    pass


def parse_tool_calls(msg: AIMessage) -> list[dict]:
    """把模型回复里的工具调用摊平成 ``{"id", "name", "args"}``。

    参数不是合法 JSON 时，LangChain 把这次调用放进 ``invalid_tool_calls`` 而不是
    ``tool_calls``。两边都要读：漏掉的话这次调用凭空消失，模型等不到任何反馈就
    卡住了。给空参数让工具层报错，错误喂回模型触发反思，比整个会话中断好。
    """
    calls = [{"id": tc.get("id") or "", "name": tc.get("name") or "",
              "args": tc.get("args") or {}}
             for tc in (getattr(msg, "tool_calls", None) or [])]
    calls += [{"id": tc.get("id") or "", "name": tc.get("name") or "", "args": {}}
              for tc in (getattr(msg, "invalid_tool_calls", None) or [])]
    return calls


class LLMClient:
    """OpenAI 兼容接口的薄封装（千问 DashScope），内部走 LangChain ``ChatOpenAI``。

    对外只有一个 ``chat(messages, tools) -> AIMessage``——收发的都是 LangChain
    消息对象。这个签名同时是**测试的注入接缝**：假模型实现同一个方法即可。
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

    def chat(self, messages: list, tools: list) -> AIMessage:
        if not self.cfg.api_key:
            raise LLMClientError(
                "config.json 里的 api_key 是空的。去阿里云百炼申请一个千问 key 填进去，"
                "或把 base_url 指向别的 OpenAI 兼容服务。")
        # tools 为空时不能 bind——绑一个空列表，部分兼容实现会报 400
        model = self._model.bind_tools(tools) if tools else self._model
        return model.invoke(messages)
