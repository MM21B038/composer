from __future__ import annotations

import os

import django
from django.conf import settings


def ensure_django() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "composer_site.settings")
    if not settings.configured:
        django.setup()
