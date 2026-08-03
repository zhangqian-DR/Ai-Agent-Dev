import pytest

from app.tools.web import (DashScopeProvider, DuckDuckGoProvider, SearchProvider,
                           build_provider, web_search)


class FakeProvider(SearchProvider):
    def search(self, query, max_results=5):
        return [{"title": "T1", "href": "http://a", "body": "结果内容"}]


class BrokenProvider(SearchProvider):
    def search(self, query, max_results=5):
        raise RuntimeError("https://duckduckgo.com/ 202 Ratelimit")


def test_web_search_formats():
    out = web_search("python", FakeProvider())
    assert "T1" in out and "http://a" in out and "结果内容" in out


def test_web_search_never_raises_on_provider_failure():
    """搜索源挂掉时不能把异常抛出去。原来异常一路冒到 loop 变成
    「工具执行出错：RatelimitException」，模型看不懂，会连试两轮才放弃。"""
    out = web_search("python", BrokenProvider())
    assert isinstance(out, str)
    assert "不可用" in out
    assert "不要重试" in out, "必须明确叫停，否则模型会反复重试同一个坏工具"


def test_web_search_handles_empty_result():
    class Empty(SearchProvider):
        def search(self, query, max_results=5):
            return []
    assert "未搜到" in web_search("python", Empty())


def test_web_search_tolerates_missing_href():
    """千问自带搜索拿不到 URL，只有标题和摘要——不能因此崩掉或输出 None。"""
    class NoHref(SearchProvider):
        def search(self, query, max_results=5):
            return [{"title": "标题", "body": "摘要"}]
    out = web_search("q", NoHref())
    assert "标题" in out and "摘要" in out and "None" not in out


# ---------- 千问自带联网搜索 ----------

class _FakeCompletions:
    def __init__(self, text):
        self.text = text
        self.seen = {}

    def create(self, **kw):
        self.seen.update(kw)
        from types import SimpleNamespace
        msg = SimpleNamespace(content=self.text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _dashscope(text):
    from types import SimpleNamespace
    p = DashScopeProvider.__new__(DashScopeProvider)
    p.model = "qwen-max"
    p._completions = _FakeCompletions(text)
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=p._completions))
    return p


def test_dashscope_provider_parses_lines():
    p = _dashscope("标题一 | https://a.com | 摘要一\n标题二 | | 摘要二")
    rows = p.search("测试", max_results=5)
    assert [r["title"] for r in rows] == ["标题一", "标题二"]
    assert rows[0]["href"] == "https://a.com"
    assert rows[1]["href"] == ""          # 没有链接是常态，不能报错


def test_dashscope_provider_turns_on_search():
    """忘了传 enable_search 的话，模型会一本正经地编造带 URL 的假新闻。"""
    p = _dashscope("t | u | b")
    p.search("测试")
    assert p._completions.seen["extra_body"]["enable_search"] is True


def test_dashscope_provider_ignores_garbage_lines():
    p = _dashscope("这是开场白\n标题 | https://a | 摘要\n\n以上就是结果")
    assert [r["title"] for r in p.search("q")] == ["标题"]


# ---------- provider 选择 ----------

def test_build_provider_by_config(tmp_path):
    from app.config import Config
    assert isinstance(build_provider(Config("", "", "qwen-max", tmp_path,
                                            search_provider="duckduckgo")), DuckDuckGoProvider)
    assert isinstance(build_provider(Config("", "", "qwen-max", tmp_path,
                                            search_provider="dashscope")), DashScopeProvider)


def test_build_provider_rejects_unknown(tmp_path):
    from app.config import Config
    with pytest.raises(ValueError, match="search_provider"):
        build_provider(Config("", "", "qwen-max", tmp_path, search_provider="谷歌"))
