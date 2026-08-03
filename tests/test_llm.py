from types import SimpleNamespace

import pytest

from app.config import Config
from app.llm.client import LLMClient, LLMClientError


class _FakeToolCall:
    def __init__(self, args='{"path":"a.txt"}'):
        self.id = "c1"
        self.function = SimpleNamespace(name="read_file", arguments=args)


def _completion(content, tool_calls):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _client(tmp_path, key="k"):
    return LLMClient(Config("http://x", key, "qwen-plus", tmp_path))


def test_chat_parses_tool_calls(monkeypatch, tmp_path):
    c = _client(tmp_path)
    monkeypatch.setattr(c._client.chat.completions, "create",
                        lambda **kw: _completion(None, [_FakeToolCall()]))
    out = c.chat([{"role": "user", "content": "hi"}], [])
    assert out["tool_calls"][0]["name"] == "read_file"
    assert out["tool_calls"][0]["args"] == {"path": "a.txt"}


def test_chat_plain_text(monkeypatch, tmp_path):
    c = _client(tmp_path)
    monkeypatch.setattr(c._client.chat.completions, "create",
                        lambda **kw: _completion("答案", None))
    out = c.chat([{"role": "user", "content": "hi"}], [])
    assert out["content"] == "答案" and out["tool_calls"] == []


def test_broken_arguments_do_not_crash(monkeypatch, tmp_path):
    """模型偶尔会吐出截断的 JSON。不能让整个会话崩掉——
    给空参数让工具层报错，错误喂回模型触发反思。"""
    c = _client(tmp_path)
    monkeypatch.setattr(c._client.chat.completions, "create",
                        lambda **kw: _completion(None, [_FakeToolCall('{"path":"a.tx')]))
    out = c.chat([{"role": "user", "content": "hi"}], [])
    assert out["tool_calls"][0]["args"] == {}


def test_tools_omitted_when_empty(monkeypatch, tmp_path):
    """tools 为空时不能把 tools=[] 发出去，部分兼容实现会报 400。"""
    c = _client(tmp_path)
    seen = {}
    monkeypatch.setattr(c._client.chat.completions, "create",
                        lambda **kw: (seen.update(kw), _completion("ok", None))[1])
    c.chat([{"role": "user", "content": "hi"}], [])
    assert "tools" not in seen


def test_empty_api_key_fails_with_actionable_message(tmp_path):
    """key 没填时要说人话，而不是抛一个 401 让人猜。"""
    with pytest.raises(LLMClientError, match="api_key"):
        _client(tmp_path, key="").chat([{"role": "user", "content": "hi"}], [])
