from __future__ import annotations

from urllib.parse import quote_plus

from nexus.system.platform import Platform
from nexus.tools.base import Param, Risk, Tool, ToolResult

SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q={}",
    "duckduckgo": "https://duckduckgo.com/?q={}",
    "bing": "https://www.bing.com/search?q={}",
    "youtube": "https://www.youtube.com/results?search_query={}",
}


def make_web_tools(platform: Platform) -> list[Tool]:
    def open_in_browser(url: str, data: dict) -> ToolResult:
        try:
            platform.open_uri(url)
        except OSError as exc:
            return ToolResult(False, f"Could not open {url}: {exc}", data)
        # The OS accepted the request; whether a tab actually appeared can't be checked.
        return ToolResult(True, f"Asked the system to open {url} in your default browser.",
                          data, verified=None)

    def open_url(url: str) -> ToolResult:
        return open_in_browser(url, {"url": url})

    def search_web(query: str, engine: str) -> ToolResult:
        url = SEARCH_ENGINES[engine].format(quote_plus(query))
        return open_in_browser(url, {"url": url, "query": query, "engine": engine})

    return [
        Tool(
            name="open_url",
            description="Open a web page (http/https) in the default browser.",
            risk=Risk.SAFE,
            params=(Param("url", "url", "full web address, e.g. https://github.com"),),
            run=open_url,
            describe=lambda a: f"Open {a['url']} in your browser",
        ),
        Tool(
            name="search_web",
            description="Search the web (or YouTube) in the default browser.",
            risk=Risk.SAFE,
            params=(
                Param("query", "str", "what to search for"),
                Param("engine", "str", "search engine", required=False,
                      default="google", choices=tuple(SEARCH_ENGINES)),
            ),
            run=search_web,
            describe=lambda a: f"Search {a['engine']} for {a['query']!r}",
        ),
    ]
