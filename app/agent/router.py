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


def route(goal: str) -> str:
    t = (goal or "").strip()
    if any(k in t for k in _CHITCHAT):
        return DIRECT

    wide = any(k in t for k in _WIDE)
    # 「恰好一个文件、且没说所有/整个」= 目标明确的具体活儿
    concrete = len(_FILEISH.findall(t)) == 1 and not wide

    if any(k in t for k in _ANALYZE):
        return FAST if concrete else SLOW
    # 没有分析词，但明说了范围很大 —— 不确定就往上升
    return SLOW if wide and not concrete else FAST
