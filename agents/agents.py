from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()

# --- Model ---
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
)

# --- Tools ---

# output_format="list" -> list[dict] with keys: snippet, title, link
# This is what gives the agent actual URLs to scrape, unlike DuckDuckGoSearchRun.
ddg = DuckDuckGoSearchResults(output_format="list", max_results=5)


@tool
def web_search(query: str) -> str:
    """Search the web and get a list of results with titles, snippets, and links.
    Use this first to discover relevant pages before deciding whether to scrape one."""
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


@tool
def scrape_page(url: str) -> str:
    """Fetch and read the full text content of a specific webpage URL.
    Use this AFTER web_search when a snippet isn't enough detail to answer
    the question and you need the actual page content. Pass one exact URL
    returned by web_search."""
    try:
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()
        if not docs:
            return f"Could not load any content from {url}"

        content = docs[0].page_content
        # Trim aggressively — full pages can blow past context limits and cost,
        # and most of it is nav/footer noise anyway (see WebBaseLoader limitations).
        max_chars = 4000
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[truncated]"
        return content
    except Exception as e:
        return f"Error scraping {url}: {e}"


tools = [web_search, scrape_page]

# --- Agent ---
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a research assistant with two tools: web_search and scrape_page.\n"
        "Workflow:\n"
        "1. Use web_search to find relevant pages for the user's question.\n"
        "2. If the search snippets are too shallow to answer confidently, pick the "
        "most relevant URL from the results and call scrape_page on it to read the "
        "full content.\n"
        "3. Repeat scraping a different link if the first one didn't have what you "
        "needed — don't give up after one scrape if better sources are available.\n"
        "4. Only answer once you have enough information. Be concise, and mention "
        "which source(s) you used."
    ),
)

if __name__ == "__main__":
    query = "compare and tell me what are the best blogs and give me links and their author names around data version control : git and why it fails for ml systems over substack"

    print("=== Live steps ===")
    final_state = None
    for chunk in agent.stream({"messages": [HumanMessage(query)]}, stream_mode="values"):
        latest = chunk["messages"][-1]
        role = latest.__class__.__name__

        if latest.content:
            print(f"\n[{role}] {latest.content}")
        elif getattr(latest, "tool_calls", None):
            for tc in latest.tool_calls:
                print(f"\n[{role}] calling tool -> {tc['name']}({tc['args']})")

        final_state = chunk  # last chunk yielded == final state, same run, no second call

    # Derived from the SAME run above, not a separate invoke() — guaranteed to match what was just streamed
    print("\n=== Final answer ===")
    print(final_state["messages"][-1].content)