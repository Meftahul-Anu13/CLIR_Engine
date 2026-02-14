from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_config() -> dict:
    config_path = Path(__file__).with_name("config.json")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data
