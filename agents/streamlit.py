import streamlit as st
import re
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader, SeleniumURLLoader
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from typing import List

load_dotenv()

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="centered")

MAX_CHARS_PER_PAGE = 4000


# ---------- Agent setup (built once, cached across reruns) ----------
@st.cache_resource(show_spinner=False)
def build_agent():
    model = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
    )

    ddg = DuckDuckGoSearchResults(output_format="list", max_results=5)

    @tool
    def web_search(query: str) -> str:
        """Search the web and get a list of results with titles, snippets, and links.
        Use this first to discover relevant pages before deciding whether to scrape any."""
        results = ddg.invoke(query)
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r.get('title', 'No title')}\n"
                f"   URL: {r.get('link', 'N/A')}\n"
                f"   Snippet: {r.get('snippet', '')}"
            )
        return "\n\n".join(formatted)

    def _truncate(text: str, limit: int = MAX_CHARS_PER_PAGE) -> str:
        if len(text) > limit:
            return text[:limit] + "\n...[truncated]"
        return text

    @tool
    def scrape_pages(urls: List[str]) -> str:
        """Fetch and read the full text content of one or more static webpages.
        Pass a list of exact URLs returned by web_search — multiple URLs can be
        scraped in a single call. Does NOT render JavaScript — if a page comes
        back empty or clearly incomplete, try scrape_pages_dynamic instead."""
        try:
            loader = WebBaseLoader(web_paths=urls)
            docs = loader.load()
            if not docs:
                return f"Could not load any content from: {urls}"
            results = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown URL")
                results.append(f"--- Content from {source} ---\n{_truncate(doc.page_content)}")
            return "\n\n".join(results)
        except Exception as e:
            return f"Error scraping {urls}: {e}"

    @tool
    def scrape_pages_dynamic(urls: List[str]) -> str:
        """Fetch and read JavaScript-rendered or anti-bot-protected webpages using
        a real headless browser (Selenium). Slower, so only use when scrape_pages
        fails or returns empty/garbled content (e.g. Substack and other SPA-style
        sites need this). Pass a list of exact URLs."""
        try:
            loader = SeleniumURLLoader(urls=urls, headless=True, continue_on_failure=True)
            docs = loader.load()
            if not docs:
                return f"Could not load any content from: {urls}"
            results = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown URL")
                results.append(f"--- Content from {source} ---\n{_truncate(doc.page_content)}")
            return "\n\n".join(results)
        except Exception as e:
            return f"Error scraping {urls} with Selenium: {e}"

    tools = [web_search, scrape_pages, scrape_pages_dynamic]

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a research assistant with three tools: web_search, scrape_pages, "
            "and scrape_pages_dynamic.\n"
            "Workflow:\n"
            "1. Use web_search to find relevant pages for the user's question.\n"
            "2. If snippets aren't enough, call scrape_pages with one or more of the "
            "most relevant URLs — you can scrape several pages in one call to "
            "cross-check or combine information.\n"
            "3. If scrape_pages returns empty, garbled, or clearly incomplete content "
            "(e.g. a 'JavaScript required' message), retry that same URL with "
            "scrape_pages_dynamic instead.\n"
            "4. If your first attempt didn't give what you needed, try a different "
            "link from the search results rather than giving up.\n"
            "5. Only answer once you have enough information. Be concise, and mention "
            "which source(s) you used."
        ),
    )


agent = build_agent()

# ---------- UI ----------
st.title("🔎 Research Agent")
st.caption("Search the web, scrape pages when needed, and answer — powered by `create_agent` + Groq (Qwen3-32B)")

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"/"assistant", "content": str}

if "scrape_log" not in st.session_state:
    st.session_state.scrape_log = []  # list of {"tool": str, "urls": list[str] | None, "content": str}

chat_tab, scraped_tab = st.tabs(["💬 Chat", "📄 Scraped Content"])

with chat_tab:
    # Replay prior turns
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something that might need a web search...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            steps_container = st.status("Working on it...", expanded=True)
            final_state = None
            pending_calls = {}  # tool_call_id -> {"name": str, "args": dict}

            for chunk in agent.stream({"messages": [HumanMessage(query)]}, stream_mode="values"):
                latest = chunk["messages"][-1]
                role = latest.__class__.__name__

                if role == "AIMessage" and getattr(latest, "tool_calls", None):
                    for tc in latest.tool_calls:
                        steps_container.write(f"🔧 Calling **{tc['name']}** with `{tc['args']}`")
                        pending_calls[tc["id"]] = {"name": tc["name"], "args": tc["args"]}

                elif role == "ToolMessage":
                    call_info = pending_calls.get(latest.tool_call_id, {})
                    tool_name = call_info.get("name", getattr(latest, "name", "unknown_tool"))
                    preview = (latest.content or "")[:300]
                    steps_container.write(f"📄 {tool_name} result (preview): {preview}...")

                    # Log full, untruncated content for scrape tools into the second tab
                    if tool_name in ("scrape_pages", "scrape_pages_dynamic"):
                        st.session_state.scrape_log.append({
                            "tool": tool_name,
                            "urls": call_info.get("args", {}).get("urls"),
                            "content": latest.content or "",
                        })

                final_state = chunk

            steps_container.update(label="Done", state="complete", expanded=False)

            final_answer = final_state["messages"][-1].content
            st.markdown(final_answer)

        st.session_state.messages.append({"role": "assistant", "content": final_answer})

def split_scraped_content(combined: str) -> list[tuple[str, str]]:
    """Split a scrape tool's combined output (multiple '--- Content from URL ---'
    blocks) into a list of (url, content) pairs for individual display."""
    pattern = r"--- Content from (.*?) ---\n"
    parts = re.split(pattern, combined)
    # parts[0] is empty/junk before the first marker; then alternates url, content, url, content...
    pages = []
    for i in range(1, len(parts), 2):
        url = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        pages.append((url, content))
    return pages if pages else [("unknown source", combined)]


with scraped_tab:
    if not st.session_state.scrape_log:
        st.info("No pages scraped yet — ask something that needs deeper detail than a search snippet.")
    else:
        page_counter = 0
        for entry in reversed(st.session_state.scrape_log):
            for url, content in split_scraped_content(entry["content"]):
                page_counter += 1
                with st.expander(f"{entry['tool']} → {url}", expanded=(page_counter == 1)):
                    st.text_area(
                        "Full scraped content",
                        value=content,
                        height=400,
                        key=f"scrape_{id(entry)}_{url}",
                    )