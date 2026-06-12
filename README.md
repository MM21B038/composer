# Composer

Thread-based agents with typed streaming, tool-hide rules, branch compression, and optional Django persistence.

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
reply = thread.invoke(agent)
print(reply.content)
```

## Chat persistence

See [docs/chat_persistence.md](docs/chat_persistence.md) for the full guide on `ChatProject` and `ChatSession` — create/list/delete projects and sessions, incognito mode, branch graphs, and resuming conversations.

```python
from composer import ChatProject, HumanMessage, SystemMessage

project = ChatProject.create("my-project")
session = project.new_session(name="main")
```

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
