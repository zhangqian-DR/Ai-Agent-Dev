import pytest
from langchain_core.messages import AIMessage

from app.config import Config
from app.llm.client import LLMClient, LLMClientError


class _FakeModel:
    """假的 chat model：记下收到了什么，返回预置的 AIMessage。

    换成 LangChain 之后测试的接缝在这里——不再是 monkeypatch openai SDK 的
    `chat.completions.create`，而是替掉整个 model 对象。
    """

    def __init__(self, reply):
        self.reply = reply
        self.bound = None          # bind_tools 收到的东西，没调过就是 None
        self.seen = None           # invoke 收到的消息

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.seen = messages
        return self.reply


def _client(tmp_path, reply=None, key="k"):
    c = LLMClient(Config("http://x", key, "qwen-plus", tmp_path))
    fake = _FakeModel(reply if reply is not None else AIMessage(content="ok"))
    c._model = fake
    return c, fake


def test_chat_parses_tool_calls(tmp_path):
    reply = AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}])
    c, _ = _client(tmp_path, reply)

    out = c.chat([{"role": "user", "content": "hi"}], [])

    assert out["tool_calls"][0]["name"] == "read_file"
    assert out["tool_calls"][0]["args"] == {"path": "a.txt"}
    assert out["tool_calls"][0]["id"] == "c1"


def test_chat_plain_text(tmp_path):
    c, _ = _client(tmp_path, AIMessage(content="答案"))

    out = c.chat([{"role": "user", "content": "hi"}], [])

    assert out["content"] == "答案" and out["tool_calls"] == []


def test_broken_arguments_do_not_crash(tmp_path):
    """模型偶尔会吐出截断的 JSON。LangChain 把这类调用放进 invalid_tool_calls
    而不是 tool_calls——两边都得读，否则这次调用会凭空消失，模型等不到任何
    反馈就卡住了。给空参数让工具层报错，错误喂回去触发反思。"""
    reply = AIMessage(content="", tool_calls=[], invalid_tool_calls=[
        {"name": "read_file", "args": '{"path":"a.tx', "id": "c1",
         "error": "unterminated string"}])
    c, _ = _client(tmp_path, reply)

    out = c.chat([{"role": "user", "content": "hi"}], [])

    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["name"] == "read_file"
    assert out["tool_calls"][0]["args"] == {}


def test_tools_omitted_when_empty(tmp_path):
    """tools 为空时不能去 bind——绑一个空列表，部分兼容实现会报 400。"""
    c, fake = _client(tmp_path)

    c.chat([{"role": "user", "content": "hi"}], [])

    assert fake.bound is None


def test_tools_are_bound_when_present(tmp_path):
    c, fake = _client(tmp_path)
    schema = {"type": "function", "function": {"name": "read_file"}}

    c.chat([{"role": "user", "content": "hi"}], [schema])

    assert fake.bound == [schema]


def test_history_dicts_are_converted_to_langchain_messages(tmp_path):
    """loop.py 与 context.py 仍用 OpenAI 那套 dict 消息，转换只发生在这一层。
    tool 消息的 tool_call_id 必须跟着走，丢了模型侧直接 400。"""
    c, fake = _client(tmp_path)
    history = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "读 a.txt"},
        {"role": "assistant", "content": "好", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "hi"},
    ]

    c.chat(history, [])

    kinds = [m.type for m in fake.seen]
    assert kinds == ["system", "human", "ai", "tool"]
    assert fake.seen[2].tool_calls[0]["args"] == {"path": "a.txt"}
    assert fake.seen[3].tool_call_id == "c1"


def test_empty_api_key_fails_with_actionable_message(tmp_path):
    """key 没填时要说人话，而不是抛一个 401 让人猜。"""
    c, _ = _client(tmp_path, key="")
    with pytest.raises(LLMClientError, match="api_key"):
        c.chat([{"role": "user", "content": "hi"}], [])
