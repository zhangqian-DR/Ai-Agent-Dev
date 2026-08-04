import pytest
from pydantic import BaseModel, ValidationError

from app.agent.errors import CORRECTABLE, RETRYABLE, TERMINAL, classify, explain
from app.llm.client import LLMClientError
from app.safety.sandbox import SandboxError


class _Status(Exception):
    """模仿 OpenAI SDK 的错误：带 status_code。按鸭子类型认，不 import openai
    ——换任何 OpenAI 兼容客户端都还能用。"""

    def __init__(self, code, msg="boom"):
        super().__init__(msg)
        self.status_code = code


@pytest.mark.parametrize("code", [401, 403])
def test_bad_key_is_terminal(code):
    """key 无效时重试一万次也一样，不该让用户以为再发一次就好。"""
    assert classify(_Status(code)) == TERMINAL


def test_unknown_model_is_terminal():
    assert classify(_Status(404)) == TERMINAL


def test_empty_key_is_terminal():
    assert classify(LLMClientError("api_key 是空的")) == TERMINAL


@pytest.mark.parametrize("code", [429, 500, 502, 503])
def test_rate_limit_and_server_errors_are_retryable(code):
    assert classify(_Status(code)) == RETRYABLE


@pytest.mark.parametrize("name", ["APITimeoutError", "APIConnectionError"])
def test_network_errors_are_retryable(name):
    exc = type(name, (Exception,), {})("网络抖了一下")
    assert classify(exc) == RETRYABLE


def test_non_ascii_credentials_are_terminal():
    """key 里混进全角字符或中文时，HTTP 头编码就先炸了，根本走不到 401。

    真机踩到过：报的是 UnicodeEncodeError: 'ascii' codec can't encode characters，
    用户完全看不出是 key 的问题。复制粘贴时混进全角字符是很常见的事故。
    """
    exc = UnicodeEncodeError("ascii", "sk-测试", 3, 5, "ordinal not in range(128)")

    assert classify(exc) == TERMINAL
    msg = explain(exc)
    assert "api_key" in msg and "ASCII" in msg
    assert "codec" not in msg, "别把 Python 的原始报错糊出来"


def test_sandbox_violation_is_correctable():
    """越界是模型自己能修的——换个工作目录内的路径就行。"""
    assert classify(SandboxError("路径超出工作目录")) == CORRECTABLE


def test_bad_arguments_are_correctable():
    class M(BaseModel):
        path: str

    with pytest.raises(ValidationError) as e:
        M()
    assert classify(e.value) == CORRECTABLE


def test_unknown_errors_default_to_correctable():
    """认不出来的一律当「模型有机会自己修」——把任务打死是更坏的默认。"""
    assert classify(RuntimeError("谁知道呢")) == CORRECTABLE


# ---------- 给人看的话 ----------

def test_terminal_message_says_what_to_fix():
    msg = explain(_Status(401))
    assert "api_key" in msg
    assert "重试" in msg, "要明说重试没用，否则用户会一直重发"


def test_retryable_message_says_it_may_work_later():
    msg = explain(_Status(429))
    assert "稍后" in msg or "重试" in msg


def test_explain_never_dumps_a_raw_traceback():
    """给用户看的是人话，不是 Error code: 401 - {'error': {...}} 这种。"""
    msg = explain(_Status(401, "Error code: 401 - {'error': {'message': 'invalid'}}"))
    assert "{'error'" not in msg


def test_sandbox_hint_tells_the_model_to_stop_trying():
    """光说「路径超出工作目录」，模型可能换个同样越界的路径再试一次。"""
    msg = explain(SandboxError("路径超出工作目录：..\\..\\x"))
    assert "工作目录" in msg
