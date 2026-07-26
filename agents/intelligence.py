from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
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
search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for current, real-time information (news, prices, recent events)."""
    return search.invoke(query)


tools = [web_search]


# --- Agent ---
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=(
        "You are a helpful research assistant. Use web_search for anything "
        "time-sensitive or factual you're unsure about. Be concise and cite "
        "what you found when relevant. just for context its 30th June 2026 as of today"
    ),
)

if __name__ == "__main__":
    query = "tell me about what protest is happening on jantar mantar recently against the education minister and also tell me about open ai news for 30th june"

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
