import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.config import Config
from app.llm.client import LLMClient, LLMClientError, parse_tool_calls


# ---------- 解析模型回复里的工具调用 ----------

def test_parses_tool_calls():
    msg = AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}])

    calls = parse_tool_calls(msg)

    assert calls == [{"id": "c1", "name": "read_file", "args": {"path": "a.txt"}}]


def test_plain_text_has_no_calls():
    assert parse_tool_calls(AIMessage(content="答案")) == []


def test_broken_arguments_still_produce_a_call():
    """模型偶尔会吐出截断的 JSON。LangChain 把这类调用放进 invalid_tool_calls
    而不是 tool_calls——两边都得读，否则这次调用会凭空消失，模型等不到任何
    反馈就卡住了。给空参数让工具层报错，错误喂回去触发反思。"""
    msg = AIMessage(content="", tool_calls=[], invalid_tool_calls=[
        {"name": "read_file", "args": '{"path":"a.tx', "id": "c1",
         "error": "unterminated string"}])

    calls = parse_tool_calls(msg)

    assert calls == [{"id": "c1", "name": "read_file", "args": {}}]


def test_valid_and_invalid_calls_are_both_kept():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "read_file", "args": {"path": "a.txt"}, "id": "c1"}],
        invalid_tool_calls=[{"name": "write_file", "args": "{bad", "id": "c2",
                             "error": "x"}])

    assert [c["id"] for c in parse_tool_calls(msg)] == ["c1", "c2"]


# ---------- chat() 本身 ----------

class _FakeModel:
    """假的 chat model：记下收到了什么，返回预置的 AIMessage。"""

    def __init__(self, reply):
        self.reply = reply
        self.bound = None          # bind_tools 收到的东西，没调过就是 None
        self.seen = None

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.seen = messages
        return self.reply


def _client(tmp_path, key="k"):
    c = LLMClient(Config("http://x", key, "qwen-plus", tmp_path))
    fake = _FakeModel(AIMessage(content="ok"))
    c._model = fake
    return c, fake


def test_tools_omitted_when_empty(tmp_path):
    """tools 为空时不能去 bind——绑一个空列表，部分兼容实现会报 400。"""
    c, fake = _client(tmp_path)

    c.chat([HumanMessage(content="hi")], [])

    assert fake.bound is None


def test_tools_are_bound_when_present(tmp_path):
    c, fake = _client(tmp_path)
    schema = {"type": "function", "function": {"name": "read_file"}}

    c.chat([HumanMessage(content="hi")], [schema])

    assert fake.bound == [schema]


def test_messages_are_passed_through_untouched(tmp_path):
    """收发的都是 LangChain 消息对象，这一层不再做任何格式转换。"""
    c, fake = _client(tmp_path)
    msgs = [HumanMessage(content="hi")]

    out = c.chat(msgs, [])

    assert fake.seen is msgs
    assert isinstance(out, AIMessage)


def test_empty_api_key_fails_with_actionable_message(tmp_path):
    """key 没填时要说人话，而不是抛一个 401 让人猜。"""
    c, _ = _client(tmp_path, key="")
    with pytest.raises(LLMClientError, match="api_key"):
        c.chat([HumanMessage(content="hi")], [])
