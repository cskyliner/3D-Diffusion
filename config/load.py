from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required for sdfusion config loading.") from exc


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"none", "null"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_by_dotted_key(config: dict[str, Any], key: str, value: Any) -> None:
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax, got: {item}")
        key, value = item.split("=", 1)
        _set_by_dotted_key(resolved, key, _parse_scalar(value))
    return resolved


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return apply_overrides(config, overrides)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
