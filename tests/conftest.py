"""
Shared test setup: etunc.config reads the committed example paths and logins (config/paths.example.yml,
config/auth.example.yml), never the machine's config/paths.yml and config/auth.yml, and the default
settings (no run file), whatever the environment.
"""

import os
from pathlib import Path

os.environ["ETUNC_PATHS"] = str(Path(__file__).resolve().parents[1] / "config" / "paths.example.yml")
os.environ["ETUNC_AUTH"] = str(Path(__file__).resolve().parents[1] / "config" / "auth.example.yml")
os.environ.pop("ETUNC_RUN", None)
