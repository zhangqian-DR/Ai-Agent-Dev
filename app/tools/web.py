from __future__ import annotations

from abc import ABC, abstractmethod

class SearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        ...

class DuckDuckGoProvider(SearchProvider):
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

def web_search(query: str, provider: SearchProvider) -> str:
    results = provider.search(query, max_results=5)
    if not results:
        return f"未搜到：{query}"
    lines = []
    for r in results[:5]:
        lines.append(f"- {r.get('title','')}\n  {r.get('href','')}\n  {r.get('body','')[:200]}")
    return "\n".join(lines)
