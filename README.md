# Composer

Thread-based agents with typed streaming, tool-hide rules, branch compression, and optional Django persistence.

## Table of contents

- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [Agent](#agent)
- [Thread](#thread)
- [Branch compression](#branch-compression)
- [Tool hiding](#tool-hiding)
- [Typed streaming](#typed-streaming)
- [MCP integration](#mcp-integration)
- [Parser (structured output)](#parser-structured-output)
- [Vector and images](#vector-and-images)
- [Chat persistence](#chat-persistence)
- [Configuration](#configuration)
- [Jupyter demo](#jupyter-demo)
- [Public API reference](#public-api-reference)
- [Development](#development)
- [Publish](#publish)

## Features

- Thread-centric conversations with `|`, `+`, and `>>` operators
- LangGraph `Agent` with sync/async invoke and streaming
- Branch compression for long contexts
- Tool-result hiding rules
- Typed stream events (`ThinkingEvent`, `AssistantEvent`, `ToolCallEvent`, `ToolResultEvent`)
- MCP client (tools, prompts, resources)
- Structured output via `Parser`
- Embeddings via `Vector`
- Vision messages via `ImageMessage` / `HumanMessage.attach`
- Optional Django/SQLite persistence (`ChatProject`, `ChatSession`)

## Install

```bash
pip install composer-agent
```

With chat persistence (Django + SQLite):

```bash
pip install composer-agent[django]
composer-migrate
```

The default database path is `~/.composer/db.sqlite3`. Override with `COMPOSER_DB_PATH`.

## Quick start

```python
from composer import Agent, HumanMessage, SystemMessage, Thread

agent = Agent(model="...", base_url="...", api_key="...")
thread = Thread()
SystemMessage("You are a helpful assistant.") | thread
thread.append(HumanMessage("Hello"))
reply = thread | agent
print(reply.content)
```

## Agent

`Agent` wraps LangGraph's `create_agent` with Composer's streaming middleware, MCP tool loading, and thread integration.

### Model configuration

Pass an OpenAI-compatible model string (as in [Quick start](#quick-start)), or inject a custom `BaseChatModel`:

```python
from langchain_openrouter import ChatOpenRouter
import os

llm = ChatOpenRouter(
    model="openrouter/free",
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    reasoning={"effort": "medium"},
)
agent = Agent(model=llm, system_prompt="You are a helpful assistant.")
```

Enable reasoning on any model via the `reasoning` parameter:

```python
agent = Agent(model="...", base_url="...", api_key="...", reasoning=True)
# or
agent = Agent(model="...", base_url="...", api_key="...", reasoning={"effort": "medium"})
```

### Invoke

```python
# Sync — appends AI reply to thread root
reply = agent.invoke(thread)
reply = thread | agent  # equivalent via AgentInvoke

# Async (e.g. in Jupyter)
reply = await agent.ainvoke(thread)
reply = await (thread | agent)
```

### Stream

```python
# Typed events (recommended)
for event in agent.stream_events(thread):
    if event.kind == "assistant":
        print(event.text, end="")

# Async
async for event in agent.astream_events(thread):
    if event.kind == "assistant":
        print(event.text, end="")

# Raw LangGraph chunks
for chunk in agent.stream(thread):
  ...
```

### Tools and MCP

Pass LangChain tools and/or an `MCPClient` instance:

```python
from composer import Agent, MCPClient

mcp = MCPClient(servers={"myserver": {"transport": "http", "url": "http://127.0.0.1:3333/mcp"}})
tools = await mcp.load_tools()
agent = Agent(model=llm, tools=tools)
```

Use `combine_tools()` to merge multiple tool sources.

### Auto tool execution

By default (`auto_tool_call=True`) the agent runs tools automatically. Set `auto_tool_call=False` to interrupt before tool execution — useful for manual approval:

```python
agent = Agent(model="...", tools=tools, auto_tool_call=False)
reply = agent.invoke(thread, auto_tool_call=False)

# Run a single tool call manually
result = agent.run_tool_call(tool_call_event.call, tools=tools, thread=thread)
```

### System prompt override

```python
reply = agent.invoke(thread, system_prompt_override="Answer in one sentence.")
```

## Thread

`Thread` is the conversation container. Messages are appended to the root thread; the model sees a filtered/compressed view via `messages_for_model()` or `thread.branch.active_view()`.

### Operators

```python
from composer import Thread, HumanMessage, SystemMessage, CompressedMessage

thread = Thread()

# Append a message
SystemMessage("You are helpful.") | thread
HumanMessage("Hello") | thread
thread.append(HumanMessage("Another message"))

# Invoke agent (compresses first if over token budget)
reply = thread | agent

# Combine two threads
combined = thread + other_thread

# Reset messages, optionally keeping a compressed summary
thread >> CompressedMessage("Prior context was summarized here.")

# Stream with compression prep (same lifecycle as invoke)
for event in thread.stream_events(agent):
    print(event.kind, event.text if hasattr(event, "text") else "")
```

### Configuration

| Option | Default | Purpose |
|--------|---------|---------|
| `auto_append_ai_message` | `True` | Append AI replies to root thread |
| `auto_append_tool_calls` | `True` | Append AI tool-call messages |
| `auto_append_tool_results` | `True` | Append tool results |
| `tool_hide_rules` | `[]` | List of `ToolResultHideRule` |
| `persist_tool_hides` | `False` | Write hides into stored messages |
| `max_messages_for_model` | `None` | Rolling message window for model view |
| `max_tokens_for_model` | `None` | Token budget trim for model view |
| `model_view_encoder` | `"cl100k_base"` | tiktoken encoder for counting |
| `compression_prompt` | `None` | Required for branch compression |
| `compression_max_tokens` | `96000` | Compress when active view exceeds this |
| `compression_tail_tokens` | `8000` | Tail preserved by token count |
| `compression_tail_messages` | `None` | Tail preserved by message count (overrides token tail) |

### Helpers

```python
thread.messages_for_model()   # filtered view sent to the model
thread.token_count()          # tokens in root thread
thread.add_tool_hide_rule(rule)
thread.copy()
thread[0]                     # slice/index like a list
```

## Branch compression

When the active view exceeds `compression_max_tokens`, Composer summarizes older turns into branch nodes. The root thread keeps the full history; summaries live on branch nodes only.

```python
thread = Thread(
    compression_prompt="Summarize prior turns for continuation.",
    compression_max_tokens=96_000,
    compression_tail_tokens=8_000,
)

SystemMessage("sys") | thread
HumanMessage("first turn") | thread
HumanMessage("second turn") | thread
HumanMessage("third turn") | thread

# Compression runs automatically before invoke/stream when over budget
view = thread.branch.active_view()  # collapsed view sent to model

# Manual compression
child = thread.branch.compress(agent)

# Switch branches
thread.branch.switch_to(some_node_id)
thread.branch.switch_to_tail()
history = thread.branch.history()
```

## Tool hiding

Collapse verbose tool outputs for the model without losing originals in storage.

```python
from composer import Thread, ToolMessage, ToolResultHideRule

thread = Thread(
    tool_hide_rules=[
        ToolResultHideRule(
            tool_name="browser_snapshot",
            server="butcher",          # optional MCP server filter
            keep_latest=1,
            hide_mode="invoke_only",   # or "persist"
            on_hide_message="[snapshot hidden]",
        )
    ]
)

thread.append(ToolMessage(content="first", name="butcher_browser_snapshot", tool_call_id="1"))
thread.append(ToolMessage(content="second", name="butcher_browser_snapshot", tool_call_id="2"))

model_msgs = thread.messages_for_model()
# Only the latest matching tool result is visible to the model
# Originals remain in thread.thread (root storage)

from composer import restore_hidden_tool_messages
restore_hidden_tool_messages(thread.thread)
```

`hide_mode="invoke_only"` hides only in the model view. `hide_mode="persist"` writes the hidden content into stored messages.

### Pattern matching

`tool_name` supports [fnmatch](https://docs.python.org/3/library/fnmatch.html) wildcards (`*`, `?`, `[...]`) against the full MCP tool name. When `server` is set, short patterns are prefixed automatically (e.g. `server="butcher", tool_name="request_*"` matches `butcher_request_*`).

| Pattern | Matches |
|---------|---------|
| `"butcher_browser_snapshot"` | Exact tool name |
| `"butcher_request_*"` | `butcher_request_snapshot`, `butcher_request_submit`, … |
| `"butcher_*_snapshot"` | Any butcher tool ending in `_snapshot` |
| `"*"` | All tools |
| `server="butcher", tool_name="*"` | All `butcher_*` tools |
| `server="butcher", tool_name="request_*"` | Same as `"butcher_request_*"` |

```python
from composer import Thread, ToolResultHideRule

# Hide older results for all butcher request tools; keep other tools visible
thread = Thread(
    tool_hide_rules=[
        ToolResultHideRule(
            tool_name="butcher_request_*",
            keep_latest=1,
            on_hide_message="[earlier request hidden]",
        ),
        # Or equivalently with server prefix:
        ToolResultHideRule(
            server="butcher",
            tool_name="request_*",
            keep_latest=1,
            on_hide_message="[earlier request hidden]",
        ),
        # Hide all tools from one MCP server except the latest of each
        ToolResultHideRule(server="butcher", tool_name="*"),
    ]
)

# Check whether a rule matches a tool name
from composer import ToolMessage, message_matches_rule

rule = ToolResultHideRule(tool_name="butcher_request_*")
msg = ToolMessage(content="...", name="butcher_request_snapshot", tool_call_id="1")
assert message_matches_rule(msg, rule)
```

## Typed streaming

Stream events have a `kind` field: `"thinking"`, `"assistant"`, `"tool_call"`, or `"tool_result"`.

```python
from composer import (
    ThinkingEvent,
    AssistantEvent,
    ToolCallEvent,
    ToolResultEvent,
)

for event in agent.stream_events(thread):
    if isinstance(event, ThinkingEvent):
        print("[think]", event.text, end="", flush=True)
    elif isinstance(event, AssistantEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallEvent):
        print(f"[tool] {event.call.name}({event.call.args})")
    elif isinstance(event, ToolResultEvent):
        print(f"[result] {event.text}")
```

`thread.stream_events(agent)` applies branch compression first, same as `thread | agent`.

## MCP integration

`MCPClient` wraps persistent MCP sessions with tool, prompt, and resource catalogs.

```python
from composer import MCPClient, Agent

mcp = MCPClient(
    servers={
        "butcher": {
            "transport": "http",
            "url": "http://127.0.0.1:3333/mcp",
        },
        # stdio example:
        # "myserver": {
        #     "transport": "stdio",
        #     "command": "python",
        #     "args": ["-m", "my_mcp_server"],
        # },
    },
    tool_name_prefix=True,  # tools become butcher_<name>, myserver_<name>, etc.
)

tools = await mcp.load_tools()
agent = Agent(model=llm, tools=tools)

# Prompts and resources
prompts = await mcp.get_prompts()
resources = await mcp.get_resources()
messages = await mcp.get_prompt("my_prompt", server="butcher", arguments={"key": "value"})

# Async context manager
async with MCPClient(servers={...}) as mcp:
    tools = await mcp.get_tools()
```

Call `load_tools()` or `ensure_connected()` before using MCP tools in sync agent paths outside async Jupyter.

## Parser (structured output)

`Parser` extracts structured data via Pydantic schemas.

```python
from pydantic import BaseModel
from composer import Parser, Thread, HumanMessage

class Person(BaseModel):
    name: str
    age: int

parser = Parser(
    model="...",
    base_url="...",
    api_key="...",
    schema=Person,
    method="function_calling",  # or "json_mode", "json_schema"
)

result = parser.parse("Alice is 30 years old")
print(result.name, result.age)

# Input can also be a Thread or message list
thread = Thread()
HumanMessage("Bob is 25") | thread
result = parser.parse(thread)
```

## Vector and images

### Embeddings

```python
from composer import Vector

vec = Vector(
    model="text-embedding-3-small",
    base_url="...",
    api_key="...",
    dim=512,           # optional truncation
    normalized=True,   # optional L2 normalization
)
embeddings = vec.vector(["doc one", "doc two"])  # numpy array
single = vec.vector("one document")
```

### Vision messages

```python
from composer import HumanMessage, ImageMessage, ImageAttach

# Attach image to a text message
msg = HumanMessage("What's in this image?").attach("photo.png", detail="high")

# Image-only message
msg = ImageMessage("photo.png", config=ImageAttach(max_size=1024, quality=85))

# ImageAttach options: max_size, format, quality, detail, brightness, contrast, saturation
```

## Chat persistence

See [docs/chat_persistence.md](docs/chat_persistence.md) for the full guide on `ChatProject` and `ChatSession` — create/list/delete projects and sessions, incognito mode, branch graphs, and resuming conversations.

```python
from composer import ChatProject, HumanMessage, SystemMessage

project = ChatProject.create("my-project")
session = project.new_session(name="main")
```

### Common workflows

```python
from composer import Agent, ChatProject, ChatSession, HumanMessage, SystemMessage

agent = Agent(model="...", base_url="...", api_key="...")

# Create and run
project = ChatProject.create("my-backend")
session = project.new_session(
    name="auth-debug",
    compression_prompt="Summarize the conversation so far for continuation.",
    compression_max_tokens=96_000,
    compression_tail_tokens=8_000,
)
SystemMessage("You are a coding assistant.") | session.thread
session.append(HumanMessage("Fix the login bug"))
reply = session.invoke(agent)

# Stream with persistence
for event in session.stream_events(agent):
    if event.kind == "assistant":
        print(event.text, end="")

# Resume later
session = ChatSession.load(session_id=session.id)
session.append(HumanMessage("Continue from here"))
reply = session.invoke(agent)

# Incognito (in-memory only) then promote to persistent
incognito = ChatSession.incognito(name="scratch")
SystemMessage("You are helpful.") | incognito.thread
incognito.append(HumanMessage("Hello"))
reply = incognito.invoke(agent)  # not persisted

saved = incognito.promote_to(project, name="saved-experiment")
print(saved.id)  # now has a database id
```

For full API tables (`ChatProject.create`, `list_all`, `delete_by`, branch switching, `branch_graph()`, etc.) see [docs/chat_persistence.md](docs/chat_persistence.md).

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `COMPOSER_DB_PATH` | `~/.composer/db.sqlite3` | SQLite database path |
| `COMPOSER_SECRET_KEY` | dev default | Django secret key |
| `BASE_URL` / `API_KEY` | — | LLM provider credentials (typically via `.env`) |

**Requirements:** Python >= 3.12

## Jupyter demo

[main.ipynb](main.ipynb) is the interactive demo entry point (not `main.py`). It demonstrates OpenRouter, MCP tools, streaming, compression, and tool hiding.

```bash
uv sync --group dev
jupyter notebook main.ipynb
```

Create a `.env` file with `BASE_URL` and `API_KEY` for the notebook's LLM setup.

## Public API reference

Exports from `composer` (grouped by area):

| Area | Names |
|------|-------|
| Agent | `Agent` |
| Thread | `Thread`, `HumanMessage`, `ImageMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`, `CompressedMessage`, `Message`, `TokenCalculator`, `EncoderType` |
| Streaming | `ThinkingEvent`, `AssistantEvent`, `ToolCallEvent`, `ToolResultEvent`, `StreamEvent`, `StreamEventKind`, `StreamEventProcessor`, `extract_thinking`, `extract_assistant_text` |
| Tools | `ToolCall`, `ToolResult`, `combine_tools`, `run_tool_call` |
| Tool hiding | `ToolResultHideRule`, `HideMode`, `get_original_content`, `is_hidden_for_model`, `message_matches_rule`, `resolve_full_tool_name`, `restore_hidden_tool_messages` |
| MCP | `MCPClient`, `MCPPromptInfo`, `MCPResourceInfo` |
| Parser | `Parser` |
| Vector | `Vector` |
| Images | `ImageAttach` |
| Persistence | `ChatProject`, `ChatSession` (lazy-loaded; requires `[django]` extra) |

## Development

```bash
uv sync --group dev
pytest
python manage.py migrate   # uses ./db.sqlite3 in the repo checkout
```

## Publish

```bash
uv sync --group dev
python -m build
twine upload --repository testpypi dist/*   # test first
twine upload dist/*                         # production PyPI
```
