"""路由的语料就是它的规格。

这份表是先写标注、再设计关键词的——反过来做就是拿答案凑题目。每条都带一句
「为什么该走这条」，改关键词表时先问自己那句话还成不成立。

判错的代价**不对称**，所以有一条单独的测试盯着最危险的那种：
    真任务 → direct   最危险：没绑工具，模型只能瞎编
    slow  → fast      次之：少了规划，但仍然能干活
    fast  → slow      只是贵一点慢一点
"""
import pytest

from app.agent.router import route

CORPUS = [
    # ---- direct：元问题与闲聊，不需要碰任何文件 ----
    ("你好", "direct", "纯打招呼"),
    ("你是谁", "direct", "元问题"),
    ("你能做什么", "direct", "元问题"),
    ("有什么功能", "direct", "元问题"),
    ("怎么用啊", "direct", "元问题"),
    ("使用说明看哪里", "direct", "元问题"),
    ("谢谢", "direct", "闲聊"),
    ("介绍一下你自己", "direct", "元问题"),

    # ---- fast：目标具体的活儿 ----
    ("读一下 main.py", "fast", "单文件读取"),
    ("修好 calc.py 里的 bug", "fast", "具体修改"),
    ("把 utils.py 的 parse_date 改成支持 ISO 8601", "fast", "具体修改"),
    ("在 workspace 里新建一个 hello.py，打印 hello", "fast", "具体创建"),
    ("跑一下测试", "fast", "单个动作"),
    ("搜一下哪里用了 shell=True", "fast", "单次搜索"),
    ("把这个函数改个名字叫 parse_iso", "fast", "具体修改"),
    ("删掉 tmp 目录里的临时文件", "fast", "具体动作"),
    ("看看 requirements.txt 里都装了什么", "fast", "单文件读取"),
    ("为什么这个测试跑不过？帮我修一下", "fast", "调试要跑起来看，不是通读分析"),
    ("把 README 里的测试数量更新成最新的", "fast", "具体修改"),
    ("git status 看一下", "fast", "单个命令"),

    # ---- slow：跨文件、要先想清楚再动 ----
    ("分析这个项目所有会写盘的地方，各自有什么防护", "slow", "跨文件分析"),
    ("梳理一下 app/agent 这几个模块的关系", "slow", "跨文件梳理"),
    ("对比 fs.py 和 shell.py 的错误处理有什么不同", "slow", "两个目标，要跨文件比对"),
    ("审计一下安全阀有没有漏洞", "slow", "审计"),
    ("评估一下现在的熔断策略够不够用", "slow", "评估"),
    ("总结一下这个项目的整体架构", "slow", "综述"),
    ("全面检查一遍有没有没关的文件句柄", "slow", "全面检查"),
    ("调研一下换成异步会有哪些影响", "slow", "调研"),

    # ---- 查资料：不碰文件，但要用 web_search，所以**不能**判进 direct ----
    # 真机撞出来的：「搜一下 LangGraph 的 checkpointer 怎么用」被判成 direct，
    # 于是模型答「我不能联网查询信息」——它确实不能，因为那条路没绑工具。
    ("搜一下 LangGraph 的 checkpointer 怎么用", "fast", "要联网查，得有 web_search"),
    ("帮我查下 RAG 相关资料", "fast", "要联网查"),
    ("上网查查 qwen3 的定价", "fast", "要联网查"),
    ("查一下今天的天气", "fast", "要联网查"),
    ("百度一下 langgraph 官网", "fast", "要联网查"),

    # ---- 边界：这几条最能说明规则对不对 ----
    ("总结一下 calc.py 这个文件讲了什么", "fast", "有分析词，但只点了一个文件"),
    ("为什么 add 函数返回值不对", "fast", "有为什么，但目标是一个函数"),
    ("帮我看看这个项目", "slow", "没有分析词，但范围是整个项目——不确定就往上升"),
    ("分析一下 a.txt 第三行是什么", "fast", "有分析词，但目标极具体"),
]


@pytest.mark.parametrize("text,want,why", CORPUS,
                         ids=[c[0][:20] for c in CORPUS])
def test_route(text, want, why):
    got = route(text)
    assert got == want, f"「{text}」应当走 {want}（{why}），实际判成了 {got}"


def test_no_real_task_is_ever_routed_to_direct():
    """最危险的一类错单独盯着：direct 路不绑任何工具，一旦把真任务判进去，
    模型没有工具可用，只能凭空编。宁可把闲聊判成 fast，也不能反过来。"""
    misrouted = [(t, w) for t, w, _ in CORPUS if w != "direct" and route(t) == "direct"]
    assert not misrouted, f"这些真任务被判成了 direct：{misrouted}"


def test_unknown_input_falls_back_to_fast():
    """认不出来的一律走 fast——有工具、能干活，是最安全的默认。"""
    for text in ("", "   ", "asdfghjkl", "?????", "帮我"):
        assert route(text) == "fast", text
