from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, ParamSpec, TypeVar

import django
from django.conf import settings

P = ParamSpec("P")
R = TypeVar("R")

_db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="composer-db")


def ensure_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "composer_site.settings")
    if not settings.configured:
        django.setup()


def run_sync_db(fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a synchronous DB callable, offloading to a thread in async contexts."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return fn(*args, **kwargs)
    return _db_executor.submit(fn, *args, **kwargs).result()


def sync_db(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return run_sync_db(func, *args, **kwargs)

    return wrapper
