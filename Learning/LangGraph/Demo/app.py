from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import TavilySearchResults

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# System prompt: guides the LLM to stop looping and synthesize after 1-2 tool calls
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful AI assistant with access to web search and Wikipedia tools. "
    "Use the tools when you need to look up current or factual information. "
    "After retrieving results from the tools, synthesize the information and provide a clear, "
    "concise answer to the user. Do NOT call tools multiple times for the same question. "
    "Once you have enough information, respond directly without calling any more tools."
))

class DataState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


# --- Pydantic input schemas for reliable Groq tool-call parsing ---
class SearchInput(BaseModel):
    query: str = Field(description="The search query string")


@tool("browse_web", args_schema=SearchInput)
def browse_web(query: str) -> str:
    """Search the web for current information using Tavily."""
    search = TavilySearchResults(max_results=1, topic="general")
    return search.invoke(query)


@tool("wiki_search", args_schema=SearchInput)
def wiki_search(query: str) -> str:
    """Search Wikipedia for encyclopedic information."""
    # NOTE: api_wrapper= keyword arg is required; WikipediaQueryRun is a Pydantic BaseTool
    search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500))
    return search.run(query)

tools = [browse_web, wiki_search]

def create_graph():
    graph = StateGraph(DataState)
    tool_node = ToolNode([browse_web, wiki_search])

    # No tool_choice="auto" — let the model decide naturally when to stop calling tools
    model_w_tools = model.bind_tools([browse_web, wiki_search])

    def invoke_model(state: DataState):
        # Prepend system prompt so the model knows when to stop using tools
        messages = [SYSTEM_PROMPT] + state.messages
        return {"messages": model_w_tools.invoke(messages)}

    graph.add_node("brain_llama70b", invoke_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "brain_llama70b")
    graph.add_conditional_edges("brain_llama70b", tools_condition)
    graph.add_edge("tools", "brain_llama70b")

    # recursion_limit is a hard safety cap to stop infinite loops (default is 25)
    agent = graph.compile()
    return agent

def react_agent():  
    # memory=MemorySaver()
    agent = create_react_agent(model, tools)
    return agent

# agent = create_graph()

agent = react_agent()