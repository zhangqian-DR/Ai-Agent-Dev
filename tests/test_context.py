import json

from app.agent.context import trim_history


# ---------- 构造消息的小工具 ----------

def _sys(text="S"):
    return {"role": "system", "content": text}


def _user(text):
    return {"role": "user", "content": text}


def _assistant_calls(*ids, content="", arguments='{"path":"a.txt"}'):
    """一条带 tool_calls 的 assistant 消息（OpenAI 协议格式）。"""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": i, "type": "function",
             "function": {"name": "read_file", "arguments": arguments}}
            for i in ids
        ],
    }


def _tool(tid, content):
    return {"role": "tool", "tool_call_id": tid, "content": content}


def _round(i, tool_output_len=1000):
    """一轮完整的工具调用：user → assistant(tool_calls) → tool 响应。"""
    tid = f"t{i}"
    return [_user(f"任务{i}"), _assistant_calls(tid), _tool(tid, "o" * tool_output_len)]


def _payload(msgs):
    """一段历史真实发给模型的字符量：content + tool_calls 里的参数都算。"""
    n = 0
    for m in msgs:
        n += len(str(m.get("content") or ""))
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            n += len(str(fn.get("name") or "")) + len(str(fn.get("arguments") or ""))
    return n


def _assert_protocol_valid(msgs):
    """OpenAI 协议要求：每条 role="tool" 的 tool_call_id，
    必须由它前面某条 assistant 的 tool_calls 声明过。否则请求直接 400。"""
    declared = set()
    for i, m in enumerate(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            declared |= {tc["id"] for tc in m["tool_calls"]}
        elif m["role"] == "tool":
            assert m["tool_call_id"] in declared, (
                f"第 {i} 条是孤儿 tool 消息（tool_call_id={m['tool_call_id']} "
                f"无对应 assistant.tool_calls），发给模型会被拒 400。\n"
                f"实际返回：{[x['role'] for x in msgs]}"
            )


# ---------- 原计划已有的两条 ----------

def test_no_trim_when_small():
    msgs = [_sys(), _user("hi")]
    assert trim_history(msgs) == msgs


def test_keeps_system_and_recent():
    msgs = [_sys()] + [_user("x" * 1000) for _ in range(30)]
    out = trim_history(msgs, max_chars=5000, keep_last=5)
    assert out[0]["role"] == "system"      # system 永远保留
    assert len(out) < len(msgs)            # 发生了裁剪
    assert out[-1] == msgs[-1]             # 最近的保留


# ---------- 缺陷 1：裁剪切开了 assistant/tool 配对 ----------

def test_tool_response_group_is_not_split():
    """按条数切窗会从一组工具调用的中间下刀，留下孤儿 tool 消息。
    多轮工具调用后必然触发，模型侧直接 400。"""
    msgs = [_sys()]
    for i in range(6):
        msgs += _round(i)

    out = trim_history(msgs, max_chars=2500, keep_last=10)

    _assert_protocol_valid(out)
    assert out[0]["role"] == "system"
    assert out[1]["role"] != "tool"        # 裁剪后的第一条不能是工具响应


def test_group_with_multiple_tool_calls_kept_whole():
    """一条 assistant 可以并发多个 tool_calls，对应多条 tool 响应，
    它们必须整体保留或整体丢弃。"""
    msgs = [_sys(), _user("旧" * 3000)]
    msgs += [_user("干活"), _assistant_calls("a1", "a2", "a3"),
             _tool("a1", "r1"), _tool("a2", "r2"), _tool("a3", "r3")]

    out = trim_history(msgs, max_chars=1000, keep_last=10)

    _assert_protocol_valid(out)
    tool_ids = [m["tool_call_id"] for m in out if m["role"] == "tool"]
    assert tool_ids == ["a1", "a2", "a3"]  # 三条响应一条都不能少


# ---------- 缺陷 2：预算漏算 tool_calls 体积 ----------

def test_tool_call_arguments_count_toward_budget():
    """write_file 的文件内容在 tool_calls.arguments 里，是历史中最大的负载。
    只统计 content 会严重低估，导致该裁的时候不裁。"""
    big = json.dumps({"path": "a.txt", "content": "x" * 5000})
    msgs = [_sys()]
    for i in range(4):
        tid = f"w{i}"
        msgs += [_assistant_calls(tid, arguments=big), _tool(tid, "已写入")]

    out = trim_history(msgs, max_chars=2000, keep_last=10)

    assert len(out) < len(msgs), "arguments 未计入预算，超窗的历史被原样放行"
    _assert_protocol_valid(out)


# ---------- 边界 ----------

def test_keeps_at_least_the_last_group():
    """预算再紧也要留下最近一组，否则 agent 丢掉当前这轮在做什么。"""
    msgs = [_sys()] + _round(0, tool_output_len=5000)

    out = trim_history(msgs, max_chars=10, keep_last=10)

    assert out[0]["role"] == "system"
    assert len(out) > 1
    _assert_protocol_valid(out)


def test_orphan_tool_message_is_dropped():
    """输入本身就带孤儿 tool 消息时，裁剪不应把它留在结果里。"""
    msgs = [_sys(), _tool("ghost", "y" * 3000)] + _round(1)

    out = trim_history(msgs, max_chars=1200, keep_last=10)

    _assert_protocol_valid(out)
    assert all(m.get("tool_call_id") != "ghost" for m in out)


def test_trimmed_result_fits_budget():
    """裁剪后必须真的落回预算内（除非只剩最后一组）。"""
    msgs = [_sys()]
    for i in range(8):
        msgs += _round(i, tool_output_len=800)

    out = trim_history(msgs, max_chars=3000, keep_last=10)

    assert _payload(out) <= 3000
    _assert_protocol_valid(out)
