"""对话历史裁剪（滑动窗口）。

为什么不能按"条数/字符数"简单切窗：
每条 ``ToolMessage`` 的 ``tool_call_id`` 必须由前面某条 ``AIMessage`` 的
``tool_calls`` 声明过。裁剪时若丢掉 AIMessage 却留下它的工具响应，请求会被
直接拒绝（400）。ReAct 循环多跑几轮必然踩到。**LangGraph 不管这件事**——
它只负责把消息存进 state，送给模型的窗口仍然要自己收着。

所以裁剪的最小单位不是"一条消息"，而是一个 **组（group）**：

- 独立消息：Human / System / 不带 tool_calls 的 AIMessage
- ``AIMessage(tool_calls=...)`` + 其后全部 ToolMessage —— 绑成一个整体，
  要么全留要么全丢

预算用字符数估算 token（中文约 1:1，英文约 4:1），不求精确，只求不超窗。
计量必须包含 ``tool_calls`` 里的参数：``write_file`` 的文件内容就在
参数里，是整段历史中最大的一块负载。
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage


def _all_calls(m) -> list:
    """一条 AIMessage 声明过的全部工具调用。

    参数不是合法 JSON 时 LangChain 把调用放进 ``invalid_tool_calls``，只看
    ``tool_calls`` 的话，这类调用对应的 ToolMessage 会被当成孤儿丢掉——而它
    恰恰是喂错误回去让模型自纠的那条消息。
    """
    return list(getattr(m, "tool_calls", None) or []) + \
        list(getattr(m, "invalid_tool_calls", None) or [])


def _msg_size(m) -> int:
    n = len(str(m.content or ""))
    for tc in _all_calls(m):
        # 参数在内存里是 dict，但发给模型时会序列化成 JSON——按 JSON 的体积算
        n += len(str(tc.get("name") or ""))
        args = tc.get("args")
        n += len(args if isinstance(args, str)          # invalid 的原样是字符串
                 else json.dumps(args or {}, ensure_ascii=False))
    return n


def _size(msgs) -> int:
    return sum(_msg_size(m) for m in msgs)


def history_size(messages: list) -> int:
    """当前历史占用的字符数。页面用它显示上下文占用条——裁剪一旦触发 agent 就会
    "忘掉"前面的步骤，让预算可见，用户能预判而不是突然发现它失忆。"""
    return _size(messages)


def _group(body: list) -> list[list]:
    """把消息序列切成不可分割的组。

    孤儿 ToolMessage（前面没有声明过它的 AIMessage）直接丢弃 —— 留着必然 400。
    """
    groups: list[list] = []
    open_ids: set = set()
    for m in body:
        if isinstance(m, ToolMessage):
            if groups and m.tool_call_id in open_ids:
                groups[-1].append(m)
            continue
        calls = _all_calls(m) if isinstance(m, AIMessage) else []
        open_ids = {tc.get("id") for tc in calls} if calls else set()
        groups.append([m])
    return groups


def trim_history(messages: list, max_chars: int = 24000, keep_last: int = 10) -> list:
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

    system = messages[:1] if isinstance(messages[0], SystemMessage) else []
    groups = _group(messages[len(system):])
    if not groups:
        return list(system)
    groups = groups[-keep_last:] if keep_last else groups

    budget = max_chars - _size(system)
    selected: list[list] = []
    used = 0
    for i, g in enumerate(reversed(groups)):
        size = _size(g)
        if i > 0 and used + size > budget:
            break
        selected.insert(0, g)
        used += size

    return list(system) + [m for g in selected for m in g]
