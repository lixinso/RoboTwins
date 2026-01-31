from __future__ import annotations

from typing import Any

from robotwins.core.config import RunConfig
from robotwins.sim.interfaces import SimEnv

from .humanoid_env import HumanoidVisionConfig, HumanoidVisionEnv
from .wheeled_env import WheeledVisionConfig, WheeledVisionEnv


def make_env(run_cfg: RunConfig) -> SimEnv:
    name = (run_cfg.env.name or "").strip().lower()
    if name in {"wheeled_tracking", "wheeled", "wheels"}:
        env_cfg = WheeledVisionConfig(
            model_path=run_cfg.mujoco.model_path,
            camera=run_cfg.env.camera,
            obs=run_cfg.env.obs,
            control_hz=run_cfg.env.control_hz,
            include_state=run_cfg.env.include_state,
            frame_stack=run_cfg.env.frame_stack,
            mock=run_cfg.mujoco.mock,
            render=run_cfg.mujoco.render,
        )
        return WheeledVisionEnv(env_cfg)

    if name in {"humanoid", "humanoid_render", "humanoid_tracking"}:
        env_cfg = HumanoidVisionConfig(
            model_path=run_cfg.mujoco.model_path,
            camera=run_cfg.env.camera,
            obs=run_cfg.env.obs,
            control_hz=run_cfg.env.control_hz,
            include_state=run_cfg.env.include_state,
            frame_stack=run_cfg.env.frame_stack,
            render=run_cfg.mujoco.render,
        )
        return HumanoidVisionEnv(env_cfg)

    raise ValueError(f"Unknown env name: {run_cfg.env.name!r}")


__all__ = ["make_env"]
