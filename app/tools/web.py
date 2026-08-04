from __future__ import annotations

from abc import ABC, abstractmethod

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        ...

class DuckDuckGoProvider(SearchProvider):
    """免 key，而且**带 URL**——dashscope 那条路拿不到链接。

    走 ``ddgs``（原 ``duckduckgo-search`` 改名后的包，它自己会聚合多个搜索源）。
    旧的 duckduckgo-search 对网络出口很敏感，从这台机器上一搜就 Ratelimit；
    换成 ddgs 之后实测能正常返回。返回字段仍是 title / href / body。
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

class DashScopeProvider(SearchProvider):
    """用千问自带的联网搜索，复用同一个 api_key，不必再申请搜索服务。

    局限：OpenAI 兼容模式不返回结构化来源，拿得到标题和摘要，**拿不到 URL**。
    但结果是真实联网抓的——不开 enable_search 时模型会一本正经地编造带真实
    域名的假新闻，那比没有搜索更危险。
    """

    _PROMPT = ("搜索：{q}\n\n只返回搜索结果本身，每条一行，格式为\n"
               "标题 | 链接 | 一句话摘要\n"
               "最多 {n} 条。没有链接就把中间一栏留空。不要写任何额外说明。")

    def __init__(self, cfg):
        from openai import OpenAI
        self.model = cfg.model
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY",
                              timeout=60.0, max_retries=1)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self._PROMPT.format(q=query, n=max_results)}],
            extra_body={"enable_search": True},
        )
        text = resp.choices[0].message.content or ""
        rows = []
        for line in text.splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3 or not parts[0]:
                continue                      # 开场白、空行、"以上就是结果" 之类一律丢掉
            href = parts[1]
            if not href.startswith("http"):
                href = ""                     # [链接] / (#) 这类占位符不要当成真 URL
            rows.append({"title": parts[0], "href": href, "body": " ".join(parts[2:])})
            if len(rows) >= max_results:
                break
        return rows

def build_provider(cfg) -> SearchProvider:
    name = getattr(cfg, "search_provider", "dashscope")
    if name == "duckduckgo":
        return DuckDuckGoProvider()
    if name == "dashscope":
        return DashScopeProvider(cfg)
    raise ValueError(f"未知的 search_provider：{name!r}（可选 dashscope / duckduckgo）")

def web_search(query: str, provider: SearchProvider) -> str:
    try:
        results = provider.search(query, max_results=5)
    except Exception as e:
        # 不能把异常抛出去。原来它会一路冒到 loop 变成「工具执行出错：RatelimitException」，
        # 模型看不懂这是"搜索源坏了"，会换个措辞连试两轮才放弃。
        return (f"联网搜索不可用（{type(e).__name__}: {str(e)[:120]}）。"
                "这是搜索源本身的问题，换关键词也没用，不要重试。"
                "改用其它办法完成任务，或直接告诉用户搜索功能当前不可用。")
    if not results:
        return f"未搜到：{query}"
    lines = []
    for r in results[:5]:
        title = r.get("title") or ""
        href = r.get("href") or ""
        body = (r.get("body") or "")[:200]
        lines.append(f"- {title}" + (f"\n  {href}" if href else "") + f"\n  {body}")
    return "\n".join(lines)
