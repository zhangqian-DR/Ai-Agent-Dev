from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from app.agent.prompt import system_prompt
from app.agent.context import history_size, trim_history
from app.llm.client import parse_tool_calls
from app.safety.commands import needs_confirmation

# 历史里单个参数值保留的字符数（工具已执行完，模型不需要再看一遍完整内容）
_ARG_VALUE_LIMIT = 200
# 同一工具 + 同一参数 + 同一结果 出现几次时提醒 / 终止
_REPEAT_NUDGE = 2
_REPEAT_ABORT = 3


def _shrink_tool_args(msg, value_limit: int = _ARG_VALUE_LIMIT) -> None:
    """工具执行完之后，把历史里的超长参数压掉。

    write_file 的文件内容就在 ``tool_calls`` 的参数里，动辄几十 KB，是整段
    历史最大的一块负载。工具层的大小上限管不到它——那是模型的输出，等工具看到
    时 token 已经花掉、消息也已进历史。只有在这里压缩才真正省下上下文。

    只砍长字符串值，``path`` 这类短字段原样保留。同时清掉 ``additional_kwargs``
    里那份原始的 tool_calls——真实模型回复会带着它，留着的话压缩就白做了。
    """
    for tc in (getattr(msg, "tool_calls", None) or []):
        args = tc.get("args")
        if not isinstance(args, dict):
            continue
        if len(json.dumps(args, ensure_ascii=False)) <= value_limit * 2:
            continue
        for k, v in list(args.items()):
            if isinstance(v, str) and len(v) > value_limit:
                args[k] = v[:value_limit] + f"…（省略 {len(v) - value_limit} 字符，该调用已执行完毕）"
    if isinstance(getattr(msg, "additional_kwargs", None), dict):
        msg.additional_kwargs.pop("tool_calls", None)


def run_agent(goal, *, llm, tools, cfg, emit, confirm, memories) -> str:
    messages = [
        SystemMessage(content=system_prompt(cfg.work_dir, memories)),
        HumanMessage(content=goal),
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
        reply = llm.chat(messages, tools.tools())
        content, calls = reply.content, parse_tool_calls(reply)
        # 没有 tool_calls 时这段 content 就是最终回答，只以 final 发一次。
        # 两个都发的话页面上同一段话会显示两遍，数据库里也会存两条。
        if content and calls:
            emit({"type": "assistant", "content": content})
        if not calls:
            emit({"type": "final", "content": content or "", "ok": True})
            return content or ""
        messages.append(reply)
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
                        # 同理：被拒绝这件事由标志说了算，不靠 result 的开头几个字
                        emit({"type": "tool", "name": name, "result": result, "ok": False})
                        messages.append(ToolMessage(content=result, tool_call_id=c["id"]))
                        continue
                result = tools.execute(name, args)
            except Exception as e:               # 工具异常→结构化喂回，触发反思
                result = f"工具执行出错：{type(e).__name__}: {e}"

            key = json.dumps([name, args, result], ensure_ascii=False, sort_keys=True, default=str)
            outcome_counts[key] = outcome_counts.get(key, 0) + 1
            sig = json.dumps([name, args], ensure_ascii=False, sort_keys=True, default=str)
            consecutive = consecutive + 1 if sig == last_sig else 1
            last_sig = sig
            seen = max(outcome_counts[key], consecutive)
            if seen >= _REPEAT_ABORT:
                stop = f"同一操作（{name}）已连续第 {seen} 次原地打转，判定为死循环，已终止。"
                emit({"type": "final", "content": stop, "ok": True})
                return stop
            if seen == _REPEAT_NUDGE:
                result += ("\n\n[系统提示] 这个操作你已经执行过一次，没有任何进展。"
                           "不要再重复，先分析失败原因，换一种方法。")

            if name == "update_plan":
                emit({"type": "plan", "steps": tools.plan, "current": tools.plan_current})
            else:
                emit({"type": "tool", "name": name, "result": result, "ok": True})
            messages.append(ToolMessage(content=result, tool_call_id=c["id"]))
        _shrink_tool_args(reply)
    emit({"type": "final", "content": "已达最大步数，停止。", "ok": True})
    return "已达最大步数，停止。"
