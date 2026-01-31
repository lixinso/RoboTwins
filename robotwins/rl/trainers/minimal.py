import random
from typing import Any, Tuple

from robotwins.core.config import load_run_config
from robotwins.sim.mujoco.runtime.wheeled_env import WheeledVisionConfig, WheeledVisionEnv

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
    env_cfg = WheeledVisionConfig(
        model_path=cfg.mujoco.model_path,
        camera=cfg.env.camera,
        obs=cfg.env.obs,
        control_hz=cfg.env.control_hz,
        include_state=cfg.env.include_state,
        frame_stack=cfg.env.frame_stack,
        mock=cfg.mujoco.mock,
        render=cfg.mujoco.render,
    )
    env = WheeledVisionEnv(env_cfg)
    obs = env.reset(seed=cfg.seed)
    total_reward = 0.0
    for _ in range(steps):
        action = _sample_action(env.action_size)
        obs, reward, done, _info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset(seed=cfg.seed)
    env.close()
    return total_reward, steps


__all__ = ["run_training"]
