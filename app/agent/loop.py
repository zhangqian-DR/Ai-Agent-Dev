"""ReAct 循环——现在由 LangGraph 的 StateGraph 驱动。

图很简单：``agent ⇄ tools``，一个环。真正的收益不在这个形状（手写 for 也能画出
同样的形状），而在阶段 4：暂停/恢复是 LangGraph 的一个原语（``interrupt()`` +
checkpointer），确认闸和崩溃恢复都是它的触发器。

**没有用预制的 ToolNode**：我们的工具执行要过确认闸、要走熔断、要按工具类型发
不同的事件、``update_plan`` 还要单独处理。包一层 ToolNode 比自己写这个节点更长，
所以这里是自己写的。

对外仍是 ``run_agent(goal, *, llm, tools, cfg, emit, confirm, memories) -> str``，
事件协议一个字没变——web 层和页面都不知道底下换了驱动。
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.prompt import system_prompt
from app.agent.context import history_size, trim_history
from app.llm.client import parse_tool_calls
from app.safety.commands import needs_confirmation

# 历史里单个参数值保留的字符数（工具已执行完，模型不需要再看一遍完整内容）
_ARG_VALUE_LIMIT = 200
# 同一工具 + 同一参数 + 同一结果 出现几次时提醒 / 终止
_REPEAT_NUDGE = 2
_REPEAT_ABORT = 3


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    step: int
    plan: list
    plan_current: int
    # 熔断的两本账，见 _tools_node
    counts: dict
    last_sig: Optional[str]
    consecutive: int
    # 非 None 就是「不是模型给的最终回答」的收场（熔断终止），直接结束
    final: Optional[str]


def _shrink_tool_args(msg, value_limit: int = _ARG_VALUE_LIMIT) -> None:
    """工具执行完之后，把历史里的超长参数压掉。

    write_file 的文件内容就在 ``tool_calls`` 的参数里，动辄几十 KB，是整段
    历史最大的一块负载。工具层的大小上限管不到它——那是模型的输出，等工具看到
    时 token 已经花掉、消息也已进历史。只有在这里压缩才真正省下上下文。

    只砍长字符串值，``path`` 这类短字段原样保留。同时清掉 ``additional_kwargs``
    里那份原始的 tool_calls——真实模型回复会带着它，留着的话压缩就白做了
    （实测：只压 tool_calls 时 wire 仍有 50180 字符，两处都清才降到 403）。
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


def build_graph(*, llm, tools, cfg, emit, confirm):
    """把两个节点接成图。所有外部依赖靠闭包带进去，节点本身只读写 state。"""

    def agent_node(state: AgentState) -> dict:
        step = state["step"] + 1
        # 只为这一次调用裁剪，不把裁剪结果写回 state：state 留着完整历史，
        # 阶段 4 的 checkpoint 才有东西可回放；送给模型的窗口另算。
        window = trim_history(state["messages"])
        emit({"type": "step", "n": step, "max": cfg.max_steps, "chars": history_size(window)})

        reply = llm.chat(window, tools.tools())
        calls = parse_tool_calls(reply)
        # 没有 tool_calls 时这段 content 就是最终回答，只以 final 发一次。
        # 两个都发的话页面上同一段话会显示两遍，数据库里也会存两条。
        if reply.content and calls:
            emit({"type": "assistant", "content": reply.content})
        if not calls:
            emit({"type": "final", "content": reply.content or "", "ok": True})
        return {"messages": [reply], "step": step}

    def tools_node(state: AgentState) -> dict:
        reply = state["messages"][-1]
        counts = dict(state["counts"])
        last_sig, consecutive = state["last_sig"], state["consecutive"]
        plan, plan_current = state["plan"], state["plan_current"]
        out: list[Any] = []

        calls = parse_tool_calls(reply)
        # 整盘门控：一轮里的危险操作**一次问完**，而不是逐条弹。
        # 这不只是 UX——闸必须在任何副作用之前、且一轮只有一个：带 checkpointer
        # 时 interrupt 恢复会让整个节点从头重跑，逐条弹的话前面已经执行过的工具
        # 会被再执行一遍。所以先算预览、问一次，再统一执行。
        approved = True
        gated = [c for c in calls if needs_confirmation(c["name"], c["args"])]
        if gated:
            actions = []
            for c in gated:
                try:
                    preview = tools.preview(c["name"], c["args"])
                except Exception as e:
                    # 预览生成不了不在这里定生死：照样交给 execute，
                    # 真正的错误由它抛出来喂回模型反思
                    preview = f"（无法生成预览：{type(e).__name__}: {e}）"
                actions.append({"name": c["name"], "preview": preview})
            approved = confirm({"actions": actions})

        for c in calls:
            name, args = c["name"], c["args"]
            # 工具异常统一兜住：模型给个越界路径（SandboxError）或漏传参数时，
            # 这些异常若冒到 web 层会把整个会话打死——而同样越界的 read_file
            # 只是返回一句错误让模型反思。统一喂回去，它才有机会重来。
            try:
                if not approved and needs_confirmation(name, args):
                    result = "用户拒绝了该操作，已跳过。"
                    # 被拒绝这件事由标志说了算，不靠 result 的开头几个字
                    emit({"type": "tool", "name": name, "result": result, "ok": False})
                    out.append(ToolMessage(content=result, tool_call_id=c["id"]))
                    continue
                result = tools.execute(name, args)
            except Exception as e:               # 工具异常→结构化喂回，触发反思
                result = f"工具执行出错：{type(e).__name__}: {e}"

            key = json.dumps([name, args, result], ensure_ascii=False, sort_keys=True, default=str)
            counts[key] = counts.get(key, 0) + 1
            sig = json.dumps([name, args], ensure_ascii=False, sort_keys=True, default=str)
            consecutive = consecutive + 1 if sig == last_sig else 1
            last_sig = sig
            seen = max(counts[key], consecutive)
            if seen >= _REPEAT_ABORT:
                stop = f"同一操作（{name}）已连续第 {seen} 次原地打转，判定为死循环，已终止。"
                emit({"type": "final", "content": stop, "ok": True})
                return {"messages": out, "counts": counts, "last_sig": last_sig,
                        "consecutive": consecutive, "final": stop}
            if seen == _REPEAT_NUDGE:
                result += ("\n\n[系统提示] 这个操作你已经执行过一次，没有任何进展。"
                           "不要再重复，先分析失败原因，换一种方法。")

            if name == "update_plan":
                plan, plan_current = tools.plan, tools.plan_current
                emit({"type": "plan", "steps": plan, "current": plan_current})
            else:
                emit({"type": "tool", "name": name, "result": result, "ok": True})
            out.append(ToolMessage(content=result, tool_call_id=c["id"]))

        # 压缩要在全部工具执行完之后。改的是 state 里那条 AIMessage，靠 add_messages
        # 的「同 id 替换」写回去——只在内存里改对象的话，带 checkpointer 时不落盘。
        _shrink_tool_args(reply)
        return {"messages": [reply, *out], "counts": counts, "last_sig": last_sig,
                "consecutive": consecutive, "plan": plan, "plan_current": plan_current}

    def after_agent(state: AgentState) -> str:
        return "tools" if parse_tool_calls(state["messages"][-1]) else END

    def after_tools(state: AgentState) -> str:
        return END if state["final"] is not None else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    g.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    return g.compile()


def run_agent(goal, *, llm, tools, cfg, emit, confirm, memories) -> str:
    graph = build_graph(llm=llm, tools=tools, cfg=cfg, emit=emit, confirm=confirm)
    init: AgentState = {
        "messages": [SystemMessage(content=system_prompt(cfg.work_dir, memories)),
                     HumanMessage(content=goal)],
        "step": 0, "plan": [], "plan_current": 0,
        "counts": {}, "last_sig": None, "consecutive": 0, "final": None,
    }
    # recursion_limit 数的是**节点执行次数**，不是模型轮次：一个 ReAct 回合是
    # agent + tools 两个节点。所以要让 max_steps 保持「最多几轮模型调用」这个
    # 含义（页面上的步数条也照旧显示它），限额得乘 2。
    # 不要写成 *2+1：多出来的那一次落在 agent 上，会白多跑一轮模型调用。
    limit = cfg.max_steps * 2
    try:
        out = graph.invoke(init, config={"recursion_limit": limit})
    except GraphRecursionError:
        stop = "已达最大步数，停止。"
        emit({"type": "final", "content": stop, "ok": True})
        return stop
    if out["final"] is not None:
        return out["final"]
    return out["messages"][-1].content or ""
