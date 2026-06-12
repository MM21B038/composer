import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "COMPOSER_SECRET_KEY",
    "django-insecure-composer-dev-only-change-in-production",
)

DEBUG = True

ALLOWED_HOSTS: list[str] = []

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "chat",
]

MIDDLEWARE: list[str] = []

ROOT_URLCONF = "composer_site.urls"

TEMPLATES: list[dict] = []

WSGI_APPLICATION = "composer_site.wsgi.application"

_default_db = Path.home() / ".composer" / "db.sqlite3"
if (BASE_DIR / "manage.py").exists():
    _default_db = BASE_DIR / "db.sqlite3"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": Path(os.environ.get("COMPOSER_DB_PATH", _default_db)),
    }
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

USE_TZ = True
TIME_ZONE = "UTC"
