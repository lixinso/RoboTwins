import random
from typing import Any, Tuple

from robotwins.core.config import load_run_config
from robotwins.sim.mujoco.runtime.factory import make_env

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None


def _sample_action(action_size: int) -> Any:
    if np is not None:
        return np.random.uniform(-1.0, 1.0, size=action_size)
    return [random.uniform(-1.0, 1.0) for _ in range(action_size)]


def run_training(config_path: str, steps: int = 100) -> Tuple[float, int]:
    cfg = load_run_config(config_path)
    env = make_env(cfg)
    obs = env.reset(seed=cfg.seed)
    total_reward = 0.0
    for _ in range(steps):
        action = _sample_action(getattr(env, "action_size", 0))
        obs, reward, done, _info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset(seed=cfg.seed)
    env.close()
    return total_reward, steps


__all__ = ["run_training"]
