# Chat Persistence — Projects & Sessions

Composer can persist chat threads and branch graphs to SQLite via Django. Use `ChatProject` and `ChatSession` for a Python-only API (no HTTP layer).

## Setup

Install dependencies and run migrations once:

```bash
uv sync
python manage.py migrate
```

The database file is created at `db.sqlite3` in the project root.

Django is bootstrapped automatically when you call `ChatProject` / `ChatSession` methods — you do not need to call `django.setup()` yourself.

## Concepts

| Concept | Description |
|---------|-------------|
| **Project** | A named container (e.g. `my-backend`, `my-app`). Holds many sessions. |
| **Session** | One conversation. Stores root messages, thread config, and the full branch graph. |
| **Root thread** | Canonical message history. All agent replies append here. |
| **Branch graph** | Compressed context windows. Summaries stay on branch nodes, not in root storage. |
| **Incognito session** | In-memory only. Nothing is written to the database until you promote it. |

## Quick start

```python
from composer import Agent, ChatProject, ChatSession, HumanMessage, SystemMessage

agent = Agent(model="...", base_url="...", api_key="...")

# Create a project and session
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
print(reply.content)
```

## ChatProject

### Create

```python
project = ChatProject.create("my-project")
```

Project names must be unique.

### Get by id or name

```python
project = ChatProject.get(name="my-project")
project = ChatProject.get(id="550e8400-e29b-41d4-a716-446655440000")
```

Raises `Project.DoesNotExist` if not found.

### List all projects

```python
for project in ChatProject.list_all():
    print(project.id, project.name, project.updated_at)
```

### Delete a project

Deletes the project and **all** its sessions (cascade).

```python
# Instance method
project = ChatProject.get(name="old-project")
project.delete()

# Class method
ChatProject.delete_by(name="old-project")
ChatProject.delete_by(id="550e8400-e29b-41d4-a716-446655440000")
```

### Sessions under a project

```python
# New persistent session
session = project.new_session(name="feature-x")

# New incognito session (not saved to DB)
session = project.new_incognito_session(name="scratch")
# or
session = project.new_session(name="scratch", incognito=True)

# List sessions (lightweight — no thread load)
for row in project.list_sessions():
    print(row["id"], row["name"], row["updated_at"])

# Load full session wrappers (reloads thread + branch from DB)
for session in project.sessions():
    print(session.id, len(session.thread))

# Get one session
session = project.get_session(name="feature-x")
session = project.get_session(id="...")
```

## ChatSession

### Load an existing session

```python
session = ChatSession.load(session_id="550e8400-e29b-41d4-a716-446655440000")

# Or by project + name
project = ChatProject.get(name="my-project")
session = ChatSession.load(project=project, name="feature-x")
```

`load()` automatically calls `continue_from_tail()` so you resume at the latest branch tip.

### Append messages

```python
session.append(HumanMessage("Next question"))
session.append(AIMessage("Draft reply"))  # persistent sessions auto-save
```

### Run the agent

```python
# Sync
reply = session.invoke(agent)

# Async
reply = await session.ainvoke(agent)
```

Both methods:
1. Switch to the latest branch tail
2. Run the agent on the active branch view
3. Append new messages to the root thread
4. Save to the database (skipped for incognito)

### Branch graph

```python
graph = session.branch_graph()
print(graph["head"], graph["tail"], graph["history"])
print(graph["nodes"])
```

### Switch branches

```python
session.branch.switch_to(some_node_id)
session.save()

# Jump back to latest branch
session.continue_from_tail()
```

### Explicit save

```python
session.save()  # no-op for incognito
```

### Delete a session

```python
session = ChatSession.load(session_id="...")
session.delete()

# Or by id without loading
ChatSession.delete_by_id("550e8400-e29b-41d4-a716-446655440000")
```

## Incognito mode

Incognito sessions behave like normal sessions in memory but **never touch the database** unless promoted.

```python
# Standalone incognito (no project)
session = ChatSession.incognito(
    name="quick-test",
    compression_prompt="Summarize...",
)

# Incognito under a project (for later promotion)
project = ChatProject.get(name="my-project")
session = project.new_incognito_session(name="experiment")

SystemMessage("You are helpful.") | session.thread
session.append(HumanMessage("Hello"))
reply = session.invoke(agent)  # not persisted

assert session.is_incognito
assert session.id is None
```

### Promote incognito → persistent

Save the current in-memory state as a new database session:

```python
project = ChatProject.get(name="my-project")
saved = session.promote_to(project, name="experiment-saved")
print(saved.id)  # now has a database id
```

### Discard incognito

```python
session.delete()  # clears in-memory state only
```

## Thread configuration

Pass any `Thread` keyword argument when creating a session:

```python
session = project.new_session(
    name="tools",
    compression_prompt="Summarize prior turns...",
    compression_max_tokens=96_000,
    compression_tail_tokens=8_000,
    max_tokens_for_model=32_000,
    persist_tool_hides=False,
)
```

Config is stored in the session row and restored on `load()`.

### Tool hide rules

```python
from composer import ToolResultHideRule

session = project.new_session(
    tool_hide_rules=[
        ToolResultHideRule(
            tool_name="browser_snapshot",
            hide_mode="invoke_only",
        ),
    ],
)
```

## Resume a conversation later

```python
session = ChatSession.load(session_id=session_id)
# Already at latest tail branch

session.append(HumanMessage("Continue from here"))
reply = session.invoke(agent)
```

## Listing everything

```python
for project in ChatProject.list_all():
    print(f"Project: {project.name} ({project.id})")
    for row in project.list_sessions():
        print(f"  Session: {row['name'] or row['id']} — updated {row['updated_at']}")
```

## Data storage notes

- **Root messages** (`StoredMessage`): human, AI, system, tool messages in order.
- **Branch nodes** (`BranchNodeRecord`): compression summaries and graph structure.
- **Compressed summaries** are not duplicated in root storage.
- **Delete project** removes all sessions, messages, and branch nodes for that project.

## API reference

### ChatProject

| Method | Description |
|--------|-------------|
| `create(name)` | Create a new project |
| `get(id=, name=)` | Fetch by UUID or name |
| `list_all()` | All projects |
| `delete()` | Delete this project and all sessions |
| `delete_by(id=, name=)` | Delete without loading wrapper |
| `new_session(name=, incognito=False, **kwargs)` | New session |
| `new_incognito_session(name=, **kwargs)` | Shorthand for incognito |
| `list_sessions()` | Lightweight session metadata list |
| `sessions()` | Full `ChatSession` wrappers |
| `get_session(id=, name=)` | Load one session |

### ChatSession

| Method / property | Description |
|-------------------|-------------|
| `incognito(project=, name=, **kwargs)` | New in-memory session |
| `load(session_id=)` / `load(project=, name=)` | Load from database |
| `delete_by_id(session_id)` | Delete without loading |
| `is_incognito` | Whether session is in-memory only |
| `id` | UUID or `None` if incognito |
| `thread` | Root `Thread` |
| `branch` | `ThreadBranchGraph` |
| `append(msg)` | Append and save |
| `save()` | Persist state |
| `invoke(agent)` / `ainvoke(agent)` | Run agent and save |
| `continue_from_tail()` | Switch to latest branch |
| `branch_graph()` | Graph dict for UI/debug |
| `promote_to(project, name=)` | Save incognito session to DB |
| `delete()` | Remove from DB or clear incognito |
