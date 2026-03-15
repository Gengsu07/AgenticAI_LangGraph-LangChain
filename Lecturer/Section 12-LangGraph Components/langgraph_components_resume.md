# 📚 Resume: Section 12 - LangGraph Components

## Overview
Section 12 covers all the **core building blocks** of LangGraph, from basic chatbot creation to advanced ReAct agents. The section progressively builds understanding through 7 key topics.

---

## 1. 🤖 Simple Chatbot with Streaming (Notebook 1)

### What Happens
Build a simple chatbot using LangGraph with **streaming** capabilities — the ability to send responses token by token in real-time.

### Key Concepts

#### Step 1: Define the State
The **State** is the data structure shared between all nodes. It uses `TypedDict` with `Annotated` and `add_messages` reducer:

```python
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

> **Why `Annotated[list, add_messages]`?**  
> The `add_messages` reducer tells LangGraph to **append** new messages to the list instead of replacing it. This preserves the full conversation history.

#### Step 2: Create a Node (Function)
A **node** is a function that takes the state, does something, and returns an update:

```python
def superbot(state: State):
    return {"messages": [llm.invoke(state['messages'])]}
```

#### Step 3: Build the Graph
Connect everything using `StateGraph`, then **compile**:

```python
graph = StateGraph(State)
graph.add_node("SuperBot", superbot)
graph.add_edge(START, "SuperBot")
graph.add_edge("SuperBot", END)
graph_builder = graph.compile(checkpointer=memory)
```

#### Step 4: Streaming Responses
Two streaming modes:

| Mode | What it returns |
|------|----------------|
| `stream_mode="updates"` | Only the **changes** after each node runs |
| `stream_mode="values"` | The **full state** after each node runs |

**Token streaming** uses `.astream_events()` to get tokens one by one as the LLM generates them:
```python
async for event in graph_builder.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

Each event has: `event` (type), `name`, `data`, `metadata` (containing `langgraph_node`).

---

## 2. 📋 State Schema with TypedDict (Notebook 3 - Part 1)

### What Happens
Learn different ways to define the **State Schema** — the structure that all nodes use to communicate.

### Key Concept: TypedDict
```python
from typing_extensions import TypedDict
from typing import Literal

class TypedDictState(TypedDict):
    name: str
    game: Literal["cricket", "badminton"]
```

> ⚠️ **Important**: TypedDict provides type **hints** only. They are checked by tools like `mypy` but are **NOT enforced at runtime**. Passing `name=123` (integer) still works!

---

## 3. 📋 State Schema with Dataclasses (Notebook 3 - Part 2)

### What Happens
Use Python's `dataclass` as an alternative state schema.

```python
from dataclasses import dataclass

@dataclass
class DataClassState:
    name: str
    game: Literal["badminton", "cricket"]
```

**Key Difference from TypedDict**: Access state fields with **dot notation** (`state.name`) instead of dictionary style (`state['name']`).

```python
def play_game(state: DataClassState):
    return {"name": state.name + " want to play "}
```

> **Invoking**: Must pass a dataclass instance: `graph.invoke(DataClassState(name="Krish", game="cricket"))`

---

## 4. ✅ State Schema with Pydantic (Notebook 4)

### What Happens
Use **Pydantic BaseModel** for **runtime data validation** — the most robust approach.

```python
from pydantic import BaseModel

class State(BaseModel):
    name: str
```
```python
# Option 1 - BaseMessage (most explicit)
messages: Annotated[list[BaseMessage], add_messages]

# Option 2 - AnyMessage (type alias, same thing)
from langchain_core.messages import AnyMessage
messages: Annotated[list[AnyMessage], add_messages]
```

### Why Pydantic is Special
Unlike TypedDict and Dataclass, Pydantic **validates data types at runtime**:

```python
# ✅ This works
graph.invoke({"name": "Hello"})

# ❌ This RAISES ValidationError!
graph.invoke({"name": 123})
# Error: Input should be a valid string [type=string_type, input_value=123]
```



### Comparison Table

| Feature | TypedDict | Dataclass | Pydantic |
|---------|-----------|-----------|----------|
| Type hints | ✅ | ✅ | ✅ |
| Runtime validation | ❌ | ❌ | ✅ |
| Access style | `state['key']` | `state.key` | `state.key` |
| Error on wrong type | ❌ | ❌ | ✅ (ValidationError) |

---

## 5. 🔗 Chains in LangGraph (Notebook 5)

### What Happens
Build **chains** — sequences of connected nodes that process data step by step.

### Key Concepts

#### Normal Edges
Connect nodes in a fixed sequence:
```python
builder.add_edge(START, "node_A")
builder.add_edge("node_A", "node_B")
builder.add_edge("node_B", END)
```
> Flow: `START → node_A → node_B → END`

#### Conditional Edges
Route to different nodes based on logic:
```python
def decide_play(state) -> Literal["cricket", "badminton"]:
    if random.random() < 0.5:
        return "cricket"
    else:
        return "badminton"

builder.add_conditional_edges("playgame", decide_play)
```
> Flow: `START → playgame → (cricket OR badminton) → END`

The function returns the **name of the next node** to execute.

---

## 6. 🛠️ Chatbots with Multiple Tools (Notebook 6)

### What Happens
Build chatbots that can use **multiple tools** — LLM decides which tool to call and when.

### Key Steps

#### Step 1: Define Tools
```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import ArxivAPIWrapper

search_tool = TavilySearchResults(max_results=2)
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
tools = [search_tool, arxiv_tool]
```

#### Step 2: Bind Tools to LLM
```python
llm_with_tools = llm.bind_tools(tools)
```

#### Step 3: Create Tool Node
LangGraph provides a built-in `ToolNode` that automatically executes the tool the LLM selects:
```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools=tools)
```

#### Step 4: Build Graph with Tool Routing
```python
from langgraph.prebuilt import tools_condition

graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
graph.add_edge("tools", "chatbot")
```

> The `tools_condition` function checks if the LLM wants to call a tool. If yes → goes to `tools` node. If no → goes to `END`.

### Flow Diagram
```
START → chatbot → (needs tool?) → tools → chatbot → ... → END
                     ↓ (no)
                    END
```

---

## 7. 🧠 ReAct Agents (Notebook 7)

### What Happens
Build a **ReAct Agent** — the most powerful pattern. The agent **Reasons** then **Acts** in a loop until it has enough information to answer.

### ReAct Pattern
```
Thought → Action → Observation → Thought → Action → ... → Final Answer
```

### Simplest Way: `create_react_agent`
LangGraph provides a **prebuilt** function:
```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke({"messages": "What is the weather in Tokyo?"})
```

> **Under the hood**, this builds the same graph pattern as Notebook 6 (chatbot + tools loop), but handles all the wiring automatically.

### With System Prompt
Add personality or instructions:
```python
system_prompt = "You are a helpful assistant. Use tools when needed."
agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
```

---

## 🗺️ LangGraph Core Architecture Summary

```
┌─────────────────────────────────────────────┐
│                 StateGraph                   │
│                                              │
│  ┌─────────┐    ┌──────┐    ┌─────────┐    │
│  │  START   │───▶│ Node │───▶│   END   │    │
│  └─────────┘    └──┬───┘    └─────────┘    │
│                    │              ▲          │
│              conditional         │          │
│                edge              │          │
│                    │        ┌────┘          │
│                    ▼        │              │
│               ┌──────┐     │              │
│               │ Node │─────┘              │
│               └──────┘                    │
│                                            │
│  State: Data shared between all nodes      │
│  Nodes: Functions that process state       │
│  Edges: Connections between nodes          │
│  Reducers: How state updates merge         │
│  Checkpointer: Saves conversation memory   │
└────────────────────────────────────────────┘
```

### Key Components Recap

| Component | What it does | Example |
|-----------|-------------|---------|
| **State** | Data shared between nodes | `TypedDict`, `Dataclass`, `Pydantic` |
| **Node** | Function that processes state | `def chatbot(state): ...` |
| **Edge** | Connection between nodes | `add_edge("A", "B")` |
| **Conditional Edge** | Dynamic routing | `add_conditional_edges("A", router_fn)` |
| **Reducer** | How to merge state updates | `Annotated[list, add_messages]` |
| **Checkpointer** | Memory for conversations | `MemorySaver()` |
| **ToolNode** | Executes tools selected by LLM | `ToolNode(tools=[...])` |
| **tools_condition** | Routes: tool call or end | `tools_condition` |
| **create_react_agent** | Prebuilt agent pattern | `create_react_agent(llm, tools)` |

---

## 📌 Best Practices from the Lectures

1. **Use Pydantic for State** when you need runtime validation (production apps)
2. **Use TypedDict** for quick prototyping (simplest)
3. **Always use `add_messages` reducer** for chat-based states to preserve history
4. **Use `MemorySaver`** checkpointer with `thread_id` for multi-turn conversations
5. **Use `stream_mode="updates"`** to see only what changed (efficient)
6. **Use `create_react_agent`** as a shortcut for common agent patterns
7. **Bind tools to LLM** before creating the graph node
