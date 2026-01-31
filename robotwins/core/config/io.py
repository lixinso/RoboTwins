from dataclasses import fields, is_dataclass, replace
from typing import Any, Dict, Type, TypeVar

from .schema import CameraConfig, EnvConfig, RunConfig

T = TypeVar("T")


def _update_dataclass(instance: T, data: Dict[str, Any]) -> T:
    if not is_dataclass(instance):
        raise TypeError("Expected dataclass instance.")
    updates = {}
    for f in fields(instance):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(getattr(instance, f.name)) and isinstance(value, dict):
            nested = _update_dataclass(getattr(instance, f.name), value)
            updates[f.name] = nested
        else:
            updates[f.name] = value
    return replace(instance, **updates)


def load_run_config(path: str) -> RunConfig:
    try:
        import yaml  # type: ignore
    except Exception:
        return RunConfig()

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    run_cfg = RunConfig()
    if isinstance(data, dict):
        run_cfg = _update_dataclass(run_cfg, data)
    return run_cfg


def default_run_config() -> RunConfig:
    return RunConfig()


__all__ = ["CameraConfig", "EnvConfig", "RunConfig", "load_run_config", "default_run_config"]
