"""错误三分：这次失败该怎么收场？

一视同仁地把所有异常都变成一句「工具执行出错：XxxError: ...」会丢掉最要紧的
一个区别——**重试到底有没有用**。沙箱越界是模型换个路径就能修的，而 api_key
无效换多少次路径、重发多少遍都一样。分不清这两者，用户只会对着一句
``Error code: 401 - {'error': {...}}`` 反复重发。

- ``RETRYABLE``   网络抖动 / 限流 / 5xx —— 过一会儿再来大概率就好了
- ``CORRECTABLE`` 参数不对 / 路径越界 —— 模型自己换个做法能修，喂回去让它反思
- ``TERMINAL``    key 无效 / 模型名不对 —— 重试没有意义，直接收场并说清楚改哪里

按**鸭子类型**认 OpenAI SDK 的错误（看 ``status_code`` 属性）而不是 import 它的
异常类：换任何 OpenAI 兼容客户端都还能用，测试里也不必构造真的 SDK 异常。
"""
from __future__ import annotations

from app.llm.client import LLMClientError
from app.safety.sandbox import SandboxError

RETRYABLE = "retryable"
CORRECTABLE = "correctable"
TERMINAL = "terminal"

_NETWORK = {"APITimeoutError", "APIConnectionError", "Timeout",
            "ConnectionError", "ReadTimeout", "ConnectTimeout"}


def classify(exc: BaseException) -> str:
    if isinstance(exc, LLMClientError):          # api_key 压根没填
        return TERMINAL
    if isinstance(exc, UnicodeEncodeError):      # 凭据里有非 ASCII，见 explain
        return TERMINAL
    if isinstance(exc, SandboxError):
        return CORRECTABLE

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (401, 403, 404):
            return TERMINAL
        if status == 429 or status >= 500:
            return RETRYABLE

    if type(exc).__name__ in _NETWORK:
        return RETRYABLE

    # 认不出来的一律当「模型有机会自己修」——把任务打死是更坏的默认
    return CORRECTABLE


def explain(exc: BaseException) -> str:
    """给人看的一句话。不把 SDK 的原始报文糊出来——那种
    ``Error code: 401 - {'error': {...}}`` 对用户没有任何指导意义。"""
    status = getattr(exc, "status_code", None)

    if isinstance(exc, LLMClientError):
        return str(exc)
    if isinstance(exc, UnicodeEncodeError):
        # HTTP 头只能是 ASCII，所以这一步在鉴权之前就炸了，压根拿不到 401。
        # 复制粘贴 key 时混进全角字符或中文是很常见的事故。
        return ("config.json 里的 api_key 或 base_url 含有非 ASCII 字符"
                "（HTTP 头只能是 ASCII）。复制粘贴时容易混进全角字符或中文，"
                "重试没有用，去掉它们。")
    if isinstance(exc, SandboxError):
        return (f"{exc}。这个路径在工作目录之外，不要再试工作目录之外的路径，"
                f"换成工作目录内的相对路径。")
    if status in (401, 403):
        return (f"模型服务拒绝了这个 api_key（HTTP {status}）。"
                f"重试没有用，去 config.json 检查 api_key 是否有效、是否有该模型的权限。")
    if status == 404:
        return ("模型服务找不到这个 model（HTTP 404）。"
                "重试没有用，去 config.json 检查 model 名字拼对了没有。")
    if status == 429:
        return "模型服务限流了（HTTP 429）。稍后重试，或把请求放慢一些。"
    if isinstance(status, int) and status >= 500:
        return f"模型服务暂时故障（HTTP {status}）。这通常是对方的问题，稍后重试。"
    if type(exc).__name__ in _NETWORK:
        return f"连不上模型服务（{type(exc).__name__}）。检查网络或代理，稍后重试。"

    return f"{type(exc).__name__}: {exc}"
