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
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from app.agent.prompt import system_prompt
from app.agent.context import history_size, trim_history
from app.agent.errors import explain
from app.llm.client import parse_tool_calls
from app.safety.commands import needs_confirmation
from app.tools import shell

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
    # 这一轮有没有真的动过东西（写文件、跑非只读命令）。只有动过才值得验收。
    touched: bool
    verify_rounds: int
    verify_passed: bool
    # 非 None 就是收场了：熔断终止、模型层出错、或验收给出的结论
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


# synth 用更大的窗口：它存在的理由就是通读全程。执行阶段每次调用都按默认预算
# 裁剪，长任务里早期观察会被裁掉，收尾那句就成了「只看得见最近几步」的结论。
_SYNTH_CHARS = 60_000

# 计划要写**要弄清楚什么**，不是**调哪个工具**。写成工具级步骤的话，模型会照着
# 一条条执行——实测出现过「列出所有文件 / 筛选 .py / 逐一读取」这样的计划，
# 结果 list_dir 被调了三遍，比不做规划还慢。
_PLAN_ASK = ("先别动手。用 update_plan 提交一份 2~4 条的调查提纲，current 填 1。\n"
             "每条写**要弄清楚什么问题**，不要写调用哪个工具、也不要把「列目录」"
             "「读文件」这种操作本身当成一步。只调这一个工具，不做别的。")

_SYNTH_ASK = ("以上是完整的执行过程。现在把结论完整地写给用户：覆盖你实际看过的"
              "全部材料，不要只总结最后几步；有没做到或没验证的地方要如实说明。"
              "不要再调用任何工具。")


def build_graph(*, llm, tools, cfg, emit, checkpointer, path: str = "fast"):
    """把节点接成图。所有外部依赖靠闭包带进去，节点本身只读写 state。

    ``path`` 决定两头：``slow`` 在开头加 planner、结尾加 synth，中间的执行完全
    复用 ``agent ⇄ tools``——三道闸（确认、熔断、验收）因此自动生效，不必在新
    节点里重接一遍。
    """
    is_slow = path == "slow"
    # slow 路的收尾归 synth，fast 路直接结束
    terminal = "synth" if is_slow else END

    def verify_due(state: AgentState) -> bool:
        """这一轮该不该跑验收命令。

        三个条件缺一不可：配了命令、真的动过东西、还没试满轮数。
        「动过东西」用 needs_confirmation 判——它本来就是"这个操作会不会改东西"
        的分类，不必另造一套；只读命令和纯问答因此不会触发验收。
        """
        return (bool(cfg.verify_cmd) and state["touched"]
                and state["verify_rounds"] < cfg.max_verify_rounds)

    def agent_node(state: AgentState) -> dict:
        step = state["step"] + 1
        # 只为这一次调用裁剪，不把裁剪结果写回 state：state 留着完整历史，
        # 阶段 4 的 checkpoint 才有东西可回放；送给模型的窗口另算。
        window = trim_history(state["messages"])
        emit({"type": "step", "n": step, "max": cfg.max_steps, "chars": history_size(window)})

        try:
            reply = llm.chat(window, tools.tools())
        except Exception as e:
            # 模型这一层的错误没有「喂回去让它自己修」这条路可走——模型都联系不上了。
            # 干净收场，并说清楚是「稍后再来」还是「去改配置」；不重试，SDK 自己
            # 已经对可重试的状态码退避过了，我们再来一遍就是双重重试。
            msg = f"出错：{explain(e)}"
            emit({"type": "final", "content": msg, "ok": False})
            return {"step": step, "final": msg}
        calls = parse_tool_calls(reply)
        # 没有 tool_calls 时这段 content 就是最终回答，只以 final 发一次。
        # 两个都发的话页面上同一段话会显示两遍，数据库里也会存两条。
        if reply.content and calls:
            emit({"type": "assistant", "content": reply.content})
        # 该验收、或后面还有 synth 时先不发 final——收没收场由那边说了算
        if not calls and not verify_due(state) and not is_slow:
            emit({"type": "final", "content": reply.content or "", "ok": True})
        return {"messages": [reply], "step": step}

    def planner_node(state: AgentState) -> dict:
        """先出计划再动手。

        只绑 ``update_plan`` 一个工具——规划那一步不该让模型顺手就动起手来。
        这样也不必新增「结构化输出」这条 LLM 接口：计划的形状本来就由
        UpdatePlanArgs 这个 pydantic 模型定死了，而且产出直接进现成的计划面板。
        模型不肯出计划也没关系，照样往下走：规划是加分项，不是必经关卡。
        """
        step = state["step"] + 1
        plan_tool = [t for t in tools.tools() if t.name == "update_plan"]
        msgs = trim_history(state["messages"]) + [HumanMessage(content=_PLAN_ASK)]
        emit({"type": "step", "n": step, "max": cfg.max_steps, "chars": history_size(msgs)})

        try:
            reply = llm.chat(msgs, plan_tool)
        except Exception as e:
            msg = f"出错：{explain(e)}"
            emit({"type": "final", "content": msg, "ok": False})
            return {"step": step, "final": msg}

        out: list[Any] = []
        plan, plan_current = state["plan"], state["plan_current"]
        for c in parse_tool_calls(reply):
            if c["name"] != "update_plan":
                continue
            try:
                result = tools.execute(c["name"], c["args"])
            except Exception as e:
                result = f"工具执行出错：{explain(e)}"
            plan, plan_current = tools.plan, tools.plan_current
            emit({"type": "plan", "steps": plan, "current": plan_current})
            out.append(ToolMessage(content=result, tool_call_id=c["id"]))
        return {"messages": [reply, *out], "step": step,
                "plan": plan, "plan_current": plan_current}

    def synth_node(state: AgentState) -> dict:
        """通读全程再写结论。

        执行阶段每次调用都按默认预算裁剪，长任务里早期观察会被裁掉——收尾那句
        因此常常是「只看得见最近几步」的结论。这里用大得多的窗口重看一遍。
        不绑工具：这一步只要一段话，给了工具反而可能又跑去干活。
        """
        msgs = trim_history(state["messages"], max_chars=_SYNTH_CHARS) + \
            [HumanMessage(content=_SYNTH_ASK)]
        try:
            reply = llm.chat(msgs, [])
        except Exception as e:
            msg = f"出错：{explain(e)}"
            emit({"type": "final", "content": msg, "ok": False})
            return {"final": msg}
        answer = reply.content or ""
        emit({"type": "final", "content": answer, "ok": True})
        return {"messages": [reply], "final": answer}

    def tools_node(state: AgentState) -> dict:
        reply = state["messages"][-1]
        counts = dict(state["counts"])
        last_sig, consecutive = state["last_sig"], state["consecutive"]
        plan, plan_current = state["plan"], state["plan_current"]
        touched = state["touched"]
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
            # 图在这里**停住并返回**，不占线程。恢复时整个节点从头重跑，
            # 这一行会直接返回恢复值而不再中断——所以上面这段只算预览、
            # 不能有副作用，下面的执行才是真正动手的地方。
            approved = interrupt({"actions": actions})

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
                # 「动过东西」直接复用安全阀的分类：需要人工确认的操作正是会改
                # 东西的那些。只读命令和纯查询不算，免得给纯问答也跑一遍验收。
                touched = touched or needs_confirmation(name, args)
            except Exception as e:
                # 工具异常一律喂回去让模型自己修（这类基本都是 CORRECTABLE），
                # 但用 explain 而不是干巴巴的类名——比如沙箱越界要明说「别再试
                # 工作目录之外的路径」，否则模型很可能换个同样越界的路径再来一次。
                result = f"工具执行出错：{explain(e)}"

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
                "consecutive": consecutive, "plan": plan, "plan_current": plan_current,
                "touched": touched}

    def verify_node(state: AgentState) -> dict:
        """模型说完成了，跑一遍验收命令看它说得对不对。

        这道闸和确认闸的位置类似，方向相反：确认闸拦危险操作，这道闸拦
        「没验证就说完成」。run_command 本来就在工具箱里、模型随时能跑测试，
        区别在于那是**可选的**，全凭它自觉；这里是流程里的一步。
        """
        rounds = state["verify_rounds"] + 1
        ok, out = shell.run_and_check(cfg.work_dir, cfg.verify_cmd,
                                      cfg.cmd_timeout, cfg.cmd_output_limit)
        emit({"type": "tool", "name": f"验收 {cfg.verify_cmd}", "result": out,
              "ok": ok, "badge": "通过" if ok else "未通过"})

        said = state["messages"][-1].content or ""
        if ok:
            if is_slow:                      # 后面还有 synth，别在这儿收场
                return {"verify_rounds": rounds, "verify_passed": True}
            emit({"type": "final", "content": said, "ok": True})
            return {"verify_rounds": rounds, "final": said}
        if rounds >= cfg.max_verify_rounds:
            # 如实说，不硬说完成——这正是这道闸存在的意义
            msg = (f"{said}\n\n⚠️ 验收未通过：`{cfg.verify_cmd}` 仍然失败，"
                   f"以上改动没有通过验证。")
            emit({"type": "final", "content": msg, "ok": False})
            return {"verify_rounds": rounds, "final": msg}

        # 把失败输出喂回去接着修。用 HumanMessage 而不是 ToolMessage：
        # 这次验收不对应任何 tool_call，挂个孤儿 ToolMessage 会被协议拒掉。
        nudge = HumanMessage(content=(
            f"验收命令 `{cfg.verify_cmd}` 没有通过：\n\n{out}\n\n"
            f"先修好再收尾，不要直接回复完成。"))
        # 顺手把熔断的两本账清零。模型刚拿到新信息，回头重读源文件想弄清哪里
        # 不对是**正当**的，不是原地打转——真机上就因此被误杀过一次：验收一红，
        # 它重读了两遍源文件，第 3 次就被掐了。verify_rounds 仍然兜着底，
        # 不会因为清零变成无限循环。
        return {"messages": [nudge], "verify_rounds": rounds,
                "counts": {}, "last_sig": None, "consecutive": 0}

    def after_agent(state: AgentState) -> str:
        if state["final"] is not None:          # 模型层出错，已经收过场了
            return END
        if parse_tool_calls(state["messages"][-1]):
            return "tools"
        return "verify" if verify_due(state) else terminal

    def after_verify(state: AgentState) -> str:
        if state["final"] is not None:      # 已经收场：fast 路过了，或轮数用尽
            return END
        if state["verify_passed"]:          # slow 路过了，收尾交给 synth
            return "synth"
        return "agent"                      # 没过，回去接着修

    def after_planner(state: AgentState) -> str:
        return END if state["final"] is not None else "agent"

    def after_tools(state: AgentState) -> str:
        return END if state["final"] is not None else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("verify", verify_node)
    # 分支表要跟着路径走：fast 路没有 synth 节点，映过去会在 compile 时报错
    extra = {"synth": "synth"} if is_slow else {}
    g.add_conditional_edges("agent", after_agent,
                            {"tools": "tools", "verify": "verify", END: END, **extra})
    g.add_conditional_edges("tools", after_tools, {"agent": "agent", END: END})
    g.add_conditional_edges("verify", after_verify,
                            {"agent": "agent", END: END, **extra})
    if is_slow:
        g.add_node("planner", planner_node)
        g.add_node("synth", synth_node)
        g.add_edge(START, "planner")
        g.add_conditional_edges("planner", after_planner, {"agent": "agent", END: END})
        g.add_edge("synth", END)
    else:
        g.add_edge(START, "agent")
    return g.compile(checkpointer=checkpointer)


class AgentRunner:
    """一轮任务的执行器：起一轮、或从确认闸恢复。

    ``start`` / ``resume`` 都是**跑到下一个停点为止**——要么任务结束，要么撞上
    确认闸。撞上闸时图直接返回，不占线程；页面回答之后再 ``resume``。
    这就是 LangGraph 换回来的东西：暂停/恢复是一个原语，确认闸只是它的触发器。
    """

    def __init__(self, *, llm, tools, cfg, emit, checkpointer, path: str = "fast"):
        self.cfg, self.emit = cfg, emit
        self.graph = build_graph(llm=llm, tools=tools, cfg=cfg, emit=emit,
                                 checkpointer=checkpointer, path=path)

    def _config(self, thread_id: str) -> dict:
        # recursion_limit 数的是**节点执行次数**，不是模型轮次：一个 ReAct 回合是
        # agent + tools 两个节点。所以要让 max_steps 保持「最多几轮模型调用」这个
        # 含义（页面上的步数条也照旧显示它），限额得乘 2。
        # 不要写成 *2+1：多出来的那一次落在 agent 上，会白多跑一轮模型调用。
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": self.cfg.max_steps * 2}

    def start(self, goal: str, memories: list, thread_id: str) -> dict:
        init: AgentState = {
            "messages": [SystemMessage(content=system_prompt(self.cfg.work_dir, memories)),
                         HumanMessage(content=goal)],
            "step": 0, "plan": [], "plan_current": 0,
            "counts": {}, "last_sig": None, "consecutive": 0,
            "touched": False, "verify_rounds": 0, "verify_passed": False,
            "final": None,
        }
        return self._drive(init, thread_id)

    def resume(self, approved: bool, thread_id: str) -> dict:
        return self._drive(Command(resume=bool(approved)), thread_id)

    def _drive(self, payload, thread_id: str) -> dict:
        """返回 ``{"done": True, "answer": str}`` 或 ``{"done": False, "pending": {...}}``。"""
        try:
            out = self.graph.invoke(payload, config=self._config(thread_id))
        except GraphRecursionError:
            stop = "已达最大步数，停止。"
            self.emit({"type": "final", "content": stop, "ok": True})
            return {"done": True, "answer": stop}
        pending = out.get("__interrupt__")
        if pending:
            return {"done": False, "pending": pending[0].value}
        if out["final"] is not None:
            return {"done": True, "answer": out["final"]}
        return {"done": True, "answer": out["messages"][-1].content or ""}


def run_agent(goal, *, llm, tools, cfg, emit, confirm, memories, path: str = "fast") -> str:
    """一路跑到底，撞上确认闸就用 ``confirm`` 回调同步问一次。

    web 层**不用**这个——它要的是「停下来、把闸交给页面、稍后恢复」，直接使唤
    ``AgentRunner``。这个包装留给测试和命令行：一个函数跑完一轮，省得每处都自己
    写 start/resume 循环。
    """
    runner = AgentRunner(llm=llm, tools=tools, cfg=cfg, emit=emit,
                         checkpointer=InMemorySaver(), path=path)
    r = runner.start(goal, memories, "local")
    while not r["done"]:
        r = runner.resume(confirm(r["pending"]), "local")
    return r["answer"]
