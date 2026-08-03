"""对话历史裁剪（滑动窗口）。

为什么不能按"条数/字符数"简单切窗：
OpenAI 协议要求每条 ``role="tool"`` 的消息，其 ``tool_call_id`` 必须由前面
某条带 ``tool_calls`` 的 assistant 声明过。裁剪时若丢掉 assistant 却留下它的
tool 响应，请求会被直接拒绝（400）。ReAct 循环多跑几轮必然踩到。

所以裁剪的最小单位不是"一条消息"，而是一个 **组（group）**：

- 独立消息：user / 不带 tool_calls 的 assistant
- ``assistant(tool_calls)`` + 其后全部 tool 响应 —— 绑成一个整体，要么全留要么全丢

预算用字符数估算 token（中文约 1:1，英文约 4:1），不求精确，只求不超窗。
计量必须包含 ``tool_calls`` 里的参数：``write_file`` 的文件内容就在
``arguments`` 里，是整段历史中最大的一块负载。
"""
from __future__ import annotations


def _msg_size(m: dict) -> int:
    n = len(str(m.get("content") or ""))
    for tc in (m.get("tool_calls") or []):
        fn = tc.get("function") or {}
        n += len(str(fn.get("name") or "")) + len(str(fn.get("arguments") or ""))
    return n


def _size(msgs) -> int:
    return sum(_msg_size(m) for m in msgs)


def history_size(messages: list[dict]) -> int:
    """当前历史占用的字符数。页面用它显示上下文占用条——裁剪一旦触发 agent 就会
    "忘掉"前面的步骤，让预算可见，用户能预判而不是突然发现它失忆。"""
    return _size(messages)


def _group(body: list[dict]) -> list[list[dict]]:
    """把消息序列切成不可分割的组。

    孤儿 tool 消息（前面没有声明过它的 assistant）直接丢弃 —— 留着必然 400。
    """
    groups: list[list[dict]] = []
    open_ids: set = set()
    for m in body:
        role = m.get("role")
        if role == "tool":
            if groups and m.get("tool_call_id") in open_ids:
                groups[-1].append(m)
            continue
        if role == "assistant" and m.get("tool_calls"):
            open_ids = {tc.get("id") for tc in m["tool_calls"]}
        else:
            open_ids = set()
        groups.append([m])
    return groups


def trim_history(messages: list[dict], max_chars: int = 24000, keep_last: int = 10) -> list[dict]:
    """裁剪对话历史，保证：

    1. 开头的 system 消息永远保留；
    2. 绝不产生孤儿 tool 消息（不会从一组工具调用中间下刀）；
    3. 结果落在 ``max_chars`` 预算内 —— 预算是硬约束，``keep_last``
       （最多保留最近几组）只在放得下时才被满足；
    4. 至少保留最后一组，否则 agent 会丢掉当前这轮在做什么。
       该组自身超预算时仍然保留：此时真正的防线是工具层的输出截断
       （``read_file`` ≤1MB、命令输出 ≤2000 字符），裁剪已无能为力。

    未超预算时原样返回（同一对象，不复制）。
    """
    if not messages:
        return messages
    if _size(messages) <= max_chars:
        return messages

    system = messages[:1] if messages[0].get("role") == "system" else []
    groups = _group(messages[len(system):])
    if not groups:
        return list(system)
    groups = groups[-keep_last:] if keep_last else groups

    budget = max_chars - _size(system)
    selected: list[list[dict]] = []
    used = 0
    for i, g in enumerate(reversed(groups)):
        size = _size(g)
        if i > 0 and used + size > budget:
            break
        selected.insert(0, g)
        used += size

    return list(system) + [m for g in selected for m in g]
