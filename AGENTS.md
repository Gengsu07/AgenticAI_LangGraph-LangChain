# AGENTS.md - Agentic AI LangGraph-LangChain Project

## Project Overview

This is a learning/educational codebase for LangChain and LangGraph. It contains Jupyter notebooks, Python scripts, and examples demonstrating:
- LangChain components (LCEL, Chains, Prompts, Output Parsers)
- LangGraph workflows (StateGraph, Agents, Tool Calling)
- FastAPI/LangServe deployment
- Streamlit client applications
- Integration with Groq, OpenAI, Ollama, and other LLM providers

## Environment Setup

```bash
# Install dependencies (uses uv if available, else pip)
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

## Running the Project

### Development Server
```bash
# Run main entry point
python main.py

# Run LangServe API server
python Lecturer/Section\ 7-LCEL/serve.py

# Run Streamlit client
streamlit run Lecturer/Section\ 7-LCEL/client.py
```

### Jupyter Notebooks
```bash
# Launch Jupyter
jupyter notebook
# or
jupyter lab
```

## Build/Lint/Test Commands

This project does not have formal test suites. For code quality:

```bash
# Type checking (if mypy is installed)
pip install mypy
mypy .

# Linting (if flake8 is installed)
pip install flake8
flake8 .

# Format code (if black is installed)
pip install black
black .
```

### Running Single Test

Since there are no formal tests, for any new tests added:

```bash
# With pytest
pytest tests/test_specific_file.py::test_function_name -v
pytest tests/ -k "test_name_pattern" -v

# Run a single test function
python -m pytest path/to/test.py::TestClass::test_method -v
```

## Code Style Guidelines

### Imports

- **Standard library first**, then third-party, then local
- Use explicit imports (avoid `import *` except for typing)
- Group imports with blank lines between groups:
  ```python
  import os
  import time
  from typing import Optional, List
  
  from dotenv import load_dotenv
  from fastapi import FastAPI
  from pydantic import BaseModel
  
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_groq import ChatGroq
  ```

### Formatting

- **Line length**: 100 characters max (follows Python Black default)
- **Indentation**: 4 spaces (no tabs)
- **Blank lines**: 2 between top-level definitions, 1 inside functions
- Use trailing commas for multi-line calls

### Types

- Use **Pydantic v2** (`from pydantic import BaseModel, Field`)
- Use `typing.Annotated` for state typing in LangGraph
- Use explicit type hints on all function signatures
- Use `Optional[X]` instead of `X | None` for compatibility
  ```python
  from typing import Optional, Annotated, List
  from pydantic import BaseModel, Field
  
  class DataState(BaseModel):
      messages: Annotated[list[BaseMessage], add_messages]
  
  def process_input(text: str, max_length: Optional[int] = None) -> str:
      ...
  ```

### Naming Conventions

- **Variables/functions**: `snake_case` (e.g., `invoke_chain`, `api_key`)
- **Classes**: `PascalCase` (e.g., `DataState`, `SearchInput`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `API_BASE_URL`, `MAX_RETRIES`)
- **Private variables**: Leading underscore (e.g., `_request_log`)
- **Files**: `snake_case.py`

### Error Handling

- Use specific exception types rather than bare `except:`
- Return error dicts in API code rather than raising for user-friendly errors
- Log exceptions with context
- Example pattern from `client.py`:
  ```python
  try:
      response = requests.post(url, json=payload, timeout=30)
      response.raise_for_status()
      return response.json()
  except requests.exceptions.ConnectionError:
      return {"error": "Cannot connect to API server"}
  except requests.exceptions.HTTPError as e:
      if e.response.status_code == 429:
          return {"error": "Rate limit exceeded"}
      return {"error": f"HTTP Error {e.response.status_code}"}
  except Exception as e:
      return {"error": f"Unexpected error: {str(e)}"}
  ```

### Configuration

- Use `python-dotenv` for environment variables
- Load in main entry points: `load_dotenv()`
- Access via `os.getenv("VAR_NAME", "default")`
- Never commit secrets - use `.env` files (already in `.gitignore`)

### LangGraph Patterns

- Use `StateGraph` with Pydantic state models
- Use `Annotated` with `add_messages` for message history
- Define tools with `@tool` decorator and `args_schema`
- Use `ToolNode` for tool execution
- Use `tools_condition` for conditional edges

```python
from typing import Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool

class DataState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]

@tool("search", args_schema=SearchInput)
def search(query: str) -> str:
    """Search for information."""
    ...
```

### Async/Await

- Use `async/await` for FastAPI endpoints and LangChain invoke
- Use `.ainvoke()` for async model calls in FastAPI
- Example:
  ```python
  @app.get("/ready")
  async def readiness_check():
      result = await model.ainvoke("test")
      return {"result": result}
  ```

### Documentation

- Add docstrings to public functions and classes
- Use Google-style or simple descriptive style
- Comment complex logic, not obvious code

## File Structure

```
/
├── main.py                 # Entry point
├── pyproject.toml         # Project config
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (do not commit)
├── Learning/              # Learning materials
│   ├── LangChain/         # LangChain tutorials
│   └── LangGraph/         # LangGraph tutorials
├── Lecturer/              # Instructor materials
│   └── Section*/          # Course sections
├── Notes/                 # Documentation notes
└── .vscode/               # VS Code settings
```

## Environment Variables

Required variables (see `.env` file):

```bash
GROQ_API_KEY=              # Groq LLM API key
LANGSMITH_API_KEY=         # LangSmith tracing
LANGSMITH_TRACING=         # Enable/disable tracing
LANGSMITH_PROJECT=         # Project name
TAVILY_API_KEY=            # Tavily search (optional)
OPENAI_API_KEY=            # OpenAI (optional)
```

## Common Patterns

### Tool Definition with Pydantic Schema
```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class SearchInput(BaseModel):
    query: str = Field(description="The search query")

@tool("web_search", args_schema=SearchInput)
def web_search(query: str) -> str:
    """Search the web for information."""
    ...
```

### StateGraph Workflow
```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(DataState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()
```

## Notes for Agents

- This is primarily a **learning repository** with notebooks and examples
- No formal CI/CD or test framework is set up
- Code style is informal but follows reasonable Python conventions
- When adding new code, match the existing patterns in the codebase
- Always respect the `.env` file for API keys - never log or expose secrets
