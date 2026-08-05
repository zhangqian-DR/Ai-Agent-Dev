"""分诊要能说出**是靠哪条规则判的**，以及关键词不命中时的兜底分类。

为什么要记依据：路由判错现在没有任何痕迹，用户只会觉得"它今天有点笨"。而且不知道
`fallback` 占多大比例，就没法判断兜底那层值不值得留。
"""
from langchain_core.messages import AIMessage

from app.agent.router import FALLBACK, decide, refine, route


def test_reason_says_which_rule_fired():
    assert decide("你能做什么").reason == "chitchat"
    assert decide("分析这个项目所有会写盘的地方").reason == "analyze_wide"
    assert decide("总结一下 calc.py 讲了什么").reason == "analyze_concrete"
    assert decide("帮我看看这个项目").reason == "wide"
    assert decide("把日志里带 ERROR 的行挑出来").reason == FALLBACK


def test_decide_agrees_with_route():
    """route() 是 decide() 的薄包装，两者永远不能给出不同的路径。"""
    for text in ("你好", "读一下 main.py", "审计一下安全阀", "帮我看看这个项目", "随便什么"):
        assert decide(text).path == route(text), text


class _Fake:
    def __init__(self, text):
        self.text, self.seen, self.calls = text, None, 0

    def chat(self, messages, tools):
        self.calls += 1
        self.seen = (messages, tools)
        return AIMessage(content=self.text)


def test_refine_only_runs_on_fallback():
    """命中关键词的快路是零成本的，不能因为加了兜底就每条都付一次调用。"""
    llm = _Fake("slow")
    d = decide("分析这个项目所有会写盘的地方")

    out = refine("分析这个项目所有会写盘的地方", d, llm)

    assert llm.calls == 0, "关键词已经判出来了，不该再问模型"
    assert out is d


def test_refine_upgrades_a_fallback():
    """兜底层要接的正是「说法一换关键词就失手」的情况。"""
    goal = "帮我把这堆代码理一理，看看结构上有什么毛病"
    d = decide(goal)
    assert d.path == "fast" and d.reason == FALLBACK      # 关键词失手

    out = refine(goal, d, _Fake("slow"))

    assert out.path == "slow"
    assert out.reason == "llm", "要能看出这条是分类器判的，不是关键词"


def test_refine_binds_no_tools_and_asks_for_one_word():
    llm = _Fake("direct")
    refine("你这个东西是干嘛使的", decide("你这个东西是干嘛使的"), llm)

    messages, tools = llm.seen
    assert tools == [], "分类不需要工具"
    assert len(messages) == 2, "系统提示 + 用户目标，不带历史"


def test_garbage_from_the_classifier_keeps_the_keyword_result():
    """模型回了看不懂的东西，就当它没说过——退回关键词的结论。"""
    goal = "把日志里带 ERROR 的行挑出来"
    d = decide(goal)

    out = refine(goal, d, _Fake("我觉得这个问题很有意思呢"))

    assert out.path == d.path and out.reason == FALLBACK


def test_classifier_failure_never_breaks_the_task():
    """分类只是锦上添花。它挂了要退回关键词的结论，不能让整轮任务陪葬。"""
    class Boom:
        def chat(self, m, t):
            raise RuntimeError("模型挂了")

    goal = "把日志里带 ERROR 的行挑出来"
    d = decide(goal)

    out = refine(goal, d, Boom())

    assert out.path == d.path and out.reason == FALLBACK
