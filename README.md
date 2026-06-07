# Composer

Thread-based agents with typed streaming, tool-hide rules, branch compression, and optional Django persistence.

## Chat persistence

See [docs/chat_persistence.md](docs/chat_persistence.md) for the full guide on `ChatProject` and `ChatSession` — create/list/delete projects and sessions, incognito mode, branch graphs, and resuming conversations.

Quick setup:

```bash
python manage.py migrate
```

```python
from composer import ChatProject, HumanMessage, SystemMessage

project = ChatProject.create("my-project")
session = project.new_session(name="main")
```
