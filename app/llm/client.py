from __future__ import annotations

import json
from openai import OpenAI


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    """OpenAI 兼容接口的薄封装（千问 DashScope）。

    只做两件事：把消息 + 工具定义发出去，把返回的 `tool_calls` 解析成
    `{"id", "name", "args"}`。换 Claude / GPT 只需要改 config.json 的
    base_url 与 model。
    """

    def __init__(self, cfg, timeout: float = 90.0, max_retries: int = 2):
        self.cfg = cfg
        # 超时必须设：agent 跑在后台线程里，请求挂住的话整个会话就卡死了，
        # 页面只会一直显示"执行中"，用户没有任何办法取消。
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "EMPTY",
            timeout=timeout,
            max_retries=max_retries,
        )

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        if not self.cfg.api_key:
            raise LLMClientError(
                "config.json 里的 api_key 是空的。去阿里云百炼申请一个千问 key 填进去，"
                "或把 base_url 指向别的 OpenAI 兼容服务。")
        kwargs = {"model": self.cfg.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # 参数不是合法 JSON 时不能直接崩：给空参数让工具层报错，
                # 错误会喂回模型触发反思，比整个会话中断好。
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "args": args})
        return {"content": msg.content, "tool_calls": calls}
