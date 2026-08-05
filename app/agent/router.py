"""把用户的目标分诊到三条路径之一。

**纯关键词，零 LLM 成本**——分诊本身不该再花一次模型调用。代价是它只看字面，
认不出来的一律走 fast（有工具、能干活，最安全的默认）。

三条路：

- ``direct``  元问题与闲聊。**不绑任何工具**，一次问答就完。
- ``fast``    目标具体的活儿。现有的 ReAct 循环。
- ``slow``    跨文件、要先想清楚再动。规划 + 反思。

判错的代价**不对称**，所以规则是往上偏的：把闲聊判成 fast 只是多花点 token，
把真任务判成 direct 则是让模型没有工具、只能凭空编。因此 ``direct`` 收得很窄，
只认明确的元问题。

关键词表怎么来的：先写一份带标注的语料（见 tests/test_router.py），再设计规则，
不是反过来。三条规则各自独立成立——

1. 分析类词说明的是**动作**；
2. 但动作没说明**目标有几个**，而那才是 fast 与 slow 的真正分界；
3. 「恰好一个已知文件」才算具体活儿，两个就是要跨文件比对了。
"""
from __future__ import annotations

import re
from typing import NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage

DIRECT = "direct"
FAST = "fast"
SLOW = "slow"

# 元问题与闲聊。只放**明确**指向 agent 自身的说法，宁窄勿宽。
_CHITCHAT = ("你好", "你是谁", "在吗", "谢谢", "能做什么", "有什么功能",
             "怎么用", "如何使用", "使用说明", "帮助", "介绍一下你")

# 动作：这类词说明用户要的是"想清楚"而不是"动手改"
_ANALYZE = ("分析", "梳理", "对比", "审计", "评估", "综述", "调研",
            "全面检查", "整体架构", "总结")

# 范围：明说了要覆盖很多东西
_WIDE = ("所有", "全部", "整个", "整体", "每个", "各个", "逐一", "项目")

# 像文件名的词：a.py / calc.txt / docs/README.md
_FILEISH = re.compile(r"[\w\-/\\]+\.[A-Za-z]{1,4}\b")


# 判定依据。落库之后才能回答「兜底那层到底被触发多少次」——不知道这个比例，
# 换嵌入路由也好、加分类器也好，都还是拍脑袋。
FALLBACK = "fallback"          # 什么都没命中，退回 fast
_BY_LLM = "llm"                # 兜底分类器判的


class Decision(NamedTuple):
    path: str
    reason: str


def decide(goal: str) -> Decision:
    """分诊，并说清是靠哪条规则判的。"""
    t = (goal or "").strip()
    if any(k in t for k in _CHITCHAT):
        return Decision(DIRECT, "chitchat")

    wide = any(k in t for k in _WIDE)
    # 「恰好一个文件、且没说所有/整个」= 目标明确的具体活儿
    concrete = len(_FILEISH.findall(t)) == 1 and not wide

    if any(k in t for k in _ANALYZE):
        return Decision(FAST, "analyze_concrete") if concrete \
            else Decision(SLOW, "analyze_wide")
    if wide and not concrete:
        # 没有分析词，但明说了范围很大 —— 不确定就往上升
        return Decision(SLOW, "wide")
    return Decision(FAST, FALLBACK)


def route(goal: str) -> str:
    return decide(goal).path


_CLASSIFY = """把用户的请求分到三类之一，只回一个词，不要解释。

direct —— 在问助手自己（能做什么、怎么用、你是谁），不需要碰任何文件
fast   —— 目标具体的活儿：改某个文件、跑个命令、读一个文件
slow   —— 跨多个文件、要先想清楚再动：分析、梳理、对比、审计整个项目

只回 direct、fast、slow 三者之一。"""


def refine(goal: str, decision: Decision, llm) -> Decision:
    """关键词没命中时，用最便宜那档模型补判一次。

    **只在 fallback 上跑**——命中关键词的快路是零成本的，不能因为加了兜底就
    每条请求都付一次调用。实测这一次调用约 0.27s / 122 tokens（提示词短、
    只输出一个词），而它能接住的正是「说法一换关键词就失手」那类：8 条改写过
    的说法里，关键词判对 3 条，分类器 8 条全对。

    看不懂的回复、或者调用本身挂了，都退回关键词的结论——分类只是锦上添花，
    不能让整轮任务陪它一起死。
    """
    if decision.reason != FALLBACK:
        return decision
    try:
        reply = llm.chat([SystemMessage(content=_CLASSIFY),
                          HumanMessage(content=goal)], [])
    except Exception:
        return decision
    said = (getattr(reply, "content", "") or "").strip().lower()
    for p in (DIRECT, FAST, SLOW):
        if p in said:
            return Decision(p, _BY_LLM)
    return decision
