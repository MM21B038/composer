from __future__ import annotations

import os
import sys


def migrate() -> None:
    """Run Django migrations for Composer chat persistence."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "composer_site.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise SystemExit(
            "Django is required for migrations. Install with: pip install composer-agent[django]"
        ) from exc
    execute_from_command_line(["composer-migrate", "migrate", *sys.argv[1:]])
