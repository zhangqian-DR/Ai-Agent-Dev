from __future__ import annotations

import json
from app.agent.prompt import system_prompt
from app.agent.context import history_size, trim_history
from app.safety.commands import needs_confirmation

# 历史里单个参数值保留的字符数（工具已执行完，模型不需要再看一遍完整内容）
_ARG_VALUE_LIMIT = 200
# 同一工具 + 同一参数 + 同一结果 出现几次时提醒 / 终止
_REPEAT_NUDGE = 2
_REPEAT_ABORT = 3


def _shrink_tool_args(msg: dict, value_limit: int = _ARG_VALUE_LIMIT) -> None:
    """工具执行完之后，把历史里的超长参数压掉。

    write_file 的文件内容就在 `tool_calls.arguments` 里，动辄几十 KB，是整段
    历史最大的一块负载。工具层的大小上限管不到它——那是模型的输出，等工具看到
    时 token 已经花掉、消息也已进历史。只有在这里压缩才真正省下上下文。

    只砍长字符串值，`path` 这类短字段原样保留；结果仍是合法 JSON。
    """
    for tc in msg.get("tool_calls") or []:
        raw = tc["function"]["arguments"]
        if len(raw) <= value_limit * 2:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            tc["function"]["arguments"] = json.dumps(
                {"_omitted": f"参数共 {len(raw)} 字符，该调用已执行完毕"}, ensure_ascii=False)
            continue
        if not isinstance(parsed, dict):
            continue
        slim = {}
        for k, v in parsed.items():
            if isinstance(v, str) and len(v) > value_limit:
                slim[k] = v[:value_limit] + f"…（省略 {len(v) - value_limit} 字符，该调用已执行完毕）"
            else:
                slim[k] = v
        tc["function"]["arguments"] = json.dumps(slim, ensure_ascii=False)


def run_agent(goal, *, llm, tools, cfg, emit, confirm, memories) -> str:
    messages = [
        {"role": "system", "content": system_prompt(cfg.work_dir, memories)},
        {"role": "user", "content": goal},
    ]
    # 熔断不按异常计数：工具失败时大多是返回错误字符串而非抛异常，只数异常的话
    # 几乎永远不触发。改用两把尺子量同一件事——「在原地打转」：
    #
    #   同参同果累计（outcome_counts）：抓 A→B→A→B 这种交替死循环，
    #       跨步数累加，但要求结果逐字相同。
    #   同参连续（consecutive）：run_command 的输出常带耗时/时间戳，逐字比对
    #       永远不相等，上面那把尺子对它形同虚设；而连着跑同一条命令本身就
    #       说明没在推进。只数「连续」不数累计，是为了不误杀隔了很多步之后
    #       正当地重读同一个文件。
    outcome_counts: dict[str, int] = {}
    last_sig, consecutive = None, 0
    for step in range(1, cfg.max_steps + 1):
        messages = trim_history(messages)
        emit({"type": "step", "n": step, "max": cfg.max_steps, "chars": history_size(messages)})
        resp = llm.chat(messages, tools.schemas())
        content, calls = resp["content"], resp["tool_calls"]
        # 没有 tool_calls 时这段 content 就是最终回答，只以 final 发一次。
        # 两个都发的话页面上同一段话会显示两遍，数据库里也会存两条。
        if content and calls:
            emit({"type": "assistant", "content": content})
        if not calls:
            emit({"type": "final", "content": content or ""})
            return content or ""
        # 记录 assistant 的 tool_calls 到历史（OpenAI 协议要求）
        assistant_msg = {"role": "assistant", "content": content or "",
            "tool_calls": [{"id": c["id"], "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}} for c in calls]}
        messages.append(assistant_msg)
        for c in calls:
            name, args = c["name"], c["args"]
            # 确认闸和工具执行共用同一个兜底：生成确认卡片的 preview 也要过沙箱，
            # 模型给个越界路径（SandboxError）或漏传参数（KeyError）时，这些异常
            # 原来在 try 之外，会一路冒到 web 层把整个会话打死——而同样越界的
            # read_file 只是返回一句错误让模型反思。统一喂回去，它才有机会重来。
            try:
                if needs_confirmation(name, args):
                    if not confirm({"name": name, "preview": tools.preview(name, args)}):
                        result = "用户拒绝了该操作，已跳过。"
                        emit({"type": "tool", "name": name, "result": result})
                        messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                        continue
                result = tools.execute(name, args)
            except Exception as e:               # 工具异常→结构化喂回，触发反思
                result = f"工具执行出错：{type(e).__name__}: {e}"

            key = json.dumps([name, args, result], ensure_ascii=False, sort_keys=True)
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
            sig = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
            consecutive = consecutive + 1 if sig == last_sig else 1
            last_sig = sig
            seen = max(outcome_counts[key], consecutive)
            if seen >= _REPEAT_ABORT:
                stop = f"同一操作（{name}）已连续第 {seen} 次原地打转，判定为死循环，已终止。"
                emit({"type": "final", "content": stop})
                return stop
            if seen == _REPEAT_NUDGE:
                result += ("\n\n[系统提示] 这个操作你已经执行过一次，没有任何进展。"
                           "不要再重复，先分析失败原因，换一种方法。")

            if name == "update_plan":
                emit({"type": "plan", "steps": tools.plan, "current": tools.plan_current})
            else:
                emit({"type": "tool", "name": name, "result": result})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
        _shrink_tool_args(assistant_msg)
    emit({"type": "final", "content": "已达最大步数，停止。"})
    return "已达最大步数，停止。"
