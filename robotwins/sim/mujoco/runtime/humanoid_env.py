from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Optional, Tuple

from robotwins.core.config import CameraConfig
from robotwins.sim.interfaces import Observation, SimEnv

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    import mujoco  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    mujoco = None


def _resolve_default_humanoid_model_path() -> str:
    """Resolve a humanoid MJCF path without vendoring assets.

    We use Gymnasium's packaged MuJoCo assets when available.
    """

    try:
        import gymnasium  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Gymnasium is required to resolve the default humanoid.xml. "
            "Either install gymnasium or set mujoco.model_path explicitly."
        ) from exc

    gym_root = Path(gymnasium.__file__).resolve().parent
    candidate = gym_root / "envs" / "mujoco" / "assets" / "humanoid.xml"
    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        "Could not find Gymnasium's humanoid.xml at the expected location. "
        "Set mujoco.model_path to an MJCF file path."
    )


@dataclass
class HumanoidVisionConfig:
    model_path: str = ""
    camera: CameraConfig = field(default_factory=CameraConfig)
    obs: str = "vision"
    control_hz: int = 50
    include_state: bool = True
    frame_stack: int = 1
    render: bool = True


class HumanoidVisionEnv(SimEnv):
    def __init__(self, config: HumanoidVisionConfig) -> None:
        self.config = config
        self._model = None
        self._data = None
        self._renderer = None
        self._frame_buffer: Deque[Any] = deque(maxlen=max(1, config.frame_stack))
        self.action_size = 0
        self._renderer_error: Optional[str] = None
        self._warned_renderer = False

        self._require_mujoco()

        model_path = (self.config.model_path or "").strip()
        if not model_path:
            model_path = _resolve_default_humanoid_model_path()

        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self.action_size = int(self._model.nu)

        if self.config.render:
            try:
                self._renderer = mujoco.Renderer(
                    self._model,
                    height=self.config.camera.height,
                    width=self.config.camera.width,
                )
            except Exception as exc:  # pragma: no cover - depends on runtime GL
                self._renderer = None
                self._renderer_error = str(exc)

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self._seed(seed)

        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._frame_buffer.clear()
        return self._make_observation()

    def step(self, action: Any) -> Tuple[Observation, float, bool, dict[str, Any]]:
        action_array = self._to_action_array(action)
        if np is not None:
            if action_array.shape[0] != self.action_size:
                raise ValueError("Action size does not match model actuators.")
            self._data.ctrl[:] = action_array
        else:
            if len(action_array) != self.action_size:
                raise ValueError("Action size does not match model actuators.")
            self._data.ctrl[:] = action_array

        for _ in range(self._control_substeps()):
            mujoco.mj_step(self._model, self._data)

        obs = self._make_observation()
        reward = 0.0
        done = False
        info: dict[str, Any] = {}
        return obs, reward, done, info

    def render(self, mode: str = "human") -> Any:
        if mode == "rgb_array":
            return self._get_image()
        if mode == "human":
            raise NotImplementedError("Viewer mode is managed by the example runner.")
        raise ValueError(f"Unknown render mode: {mode}")

    def close(self) -> None:
        self._renderer = None

    def get_mujoco_handles(self) -> Tuple[Any, Any]:
        if self._model is None or self._data is None:
            raise RuntimeError("MuJoCo model/data not initialized.")
        return self._model, self._data

    def get_timestep(self) -> float:
        if self._model is not None:
            return float(self._model.opt.timestep)
        return 1.0 / float(self.config.control_hz)

    def _control_substeps(self) -> int:
        dt = self.get_timestep()
        target_dt = 1.0 / float(self.config.control_hz)
        if dt <= 0:
            return 1
        return max(1, int(round(target_dt / dt)))

    def _require_mujoco(self) -> None:
        if mujoco is None:
            raise RuntimeError("MuJoCo is not installed. Install mujoco to use HumanoidVisionEnv.")

    def _seed(self, seed: int) -> None:
        import random

        random.seed(seed)
        if np is not None:
            np.random.seed(seed)

    def _to_action_array(self, action: Any) -> Any:
        if np is not None:
            return np.asarray(action, dtype=float).reshape(-1)
        return action

    def _make_observation(self) -> Observation:
        image = self._get_image()
        if self.config.frame_stack > 1:
            self._frame_buffer.append(image)
            image = self._stack_frames()
        obs: Observation = {"image": image}
        if self.config.include_state:
            obs["state"] = self._get_state()
        return obs

    def _get_state(self) -> Any:
        if self._data is None:
            return {}
        if np is None:
            return {"qpos": list(self._data.qpos), "qvel": list(self._data.qvel)}
        return {"qpos": self._data.qpos.copy(), "qvel": self._data.qvel.copy()}

    def _get_image(self) -> Any:
        if self._renderer is None or self._data is None:
            if not self._warned_renderer:
                if not self.config.render:
                    message = "rendering disabled"
                else:
                    message = self._renderer_error or "Renderer unavailable"
                print(f"warning: camera disabled ({message}); returning black frames")
                self._warned_renderer = True
            return self._black_frame()

        try:
            self._renderer.update_scene(self._data, camera=self.config.camera.name)
        except Exception:
            self._renderer.update_scene(self._data)

        image = self._renderer.render()
        if np is None:
            return image.tolist()
        return image

    def _black_frame(self) -> Any:
        height = self.config.camera.height
        width = self.config.camera.width
        if np is None:
            return [[[0, 0, 0] for _ in range(width)] for _ in range(height)]
        return np.zeros((height, width, 3), dtype=np.uint8)

    def _stack_frames(self) -> Any:
        frames = list(self._frame_buffer)
        if not frames:
            return None
        if np is None:
            return frames
        return np.concatenate(frames, axis=-1)


__all__ = ["HumanoidVisionConfig", "HumanoidVisionEnv"]
