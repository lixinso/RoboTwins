from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

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


@dataclass
class WheeledVisionConfig:
    model_path: str
    camera: CameraConfig = field(default_factory=CameraConfig)
    obs: str = "vision"
    control_hz: int = 50
    include_state: bool = True
    frame_stack: int = 2
    mock: bool = True
    render: bool = True


class WheeledVisionEnv(SimEnv):
    def __init__(self, config: WheeledVisionConfig) -> None:
        self.config = config
        self._model = None
        self._data = None
        self._renderer = None
        self._frame_buffer: Deque[Any] = deque(maxlen=max(1, config.frame_stack))
        self._mock_state = {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
        self.action_size = 2
        self._renderer_error: Optional[str] = None
        self._warned_renderer = False

        if not self.config.mock:
            self._require_mujoco()
            self._model = mujoco.MjModel.from_xml_path(self.config.model_path)
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

        if self.config.mock:
            self._mock_state = {"x": 0.0, "y": 0.0, "yaw": 0.0, "vx": 0.0, "vy": 0.0}
            self._frame_buffer.clear()
            obs = self._make_observation()
            return obs

        self._require_mujoco()
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._frame_buffer.clear()
        return self._make_observation()

    def step(self, action: Any) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.config.mock:
            self._mock_step(action)
            obs = self._make_observation()
            return obs, 0.0, False, {}

        self._require_mujoco()
        action_array = self._to_action_array(action)
        if np is not None:
            if action_array.shape[0] != self.action_size:
                raise ValueError("Action size does not match model actuators.")
            self._data.ctrl[:] = action_array
        else:
            if len(action_array) != self.action_size:
                raise ValueError("Action size does not match model actuators.")
            self._data.ctrl[:] = action_array
        mujoco.mj_step(self._model, self._data)
        obs = self._make_observation()
        return obs, 0.0, False, {}

    def render(self, mode: str = "human") -> Any:
        if mode == "rgb_array":
            return self._get_image()
        if mode == "human":
            raise NotImplementedError("Viewer mode is not wired in this stub.")
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

    def _require_mujoco(self) -> None:
        if mujoco is None:
            raise RuntimeError(
                "MuJoCo is not installed. Install mujoco to use WheeledVisionEnv."
            )

    def _seed(self, seed: int) -> None:
        import random

        random.seed(seed)
        if np is not None:
            np.random.seed(seed)

    def _mock_step(self, action: Any) -> None:
        left, right = self._coerce_action_pair(action)
        dt = 1.0 / float(self.config.control_hz)
        linear = (left + right) * 0.5
        angular = (right - left)
        self._mock_state["yaw"] += angular * dt
        self._mock_state["x"] += linear * dt
        self._mock_state["vx"] = linear

    def _coerce_action_pair(self, action: Any) -> Tuple[float, float]:
        if np is not None and hasattr(action, "shape"):
            data = np.asarray(action).reshape(-1)
            if data.size < 2:
                return float(data[0]), float(data[0])
            return float(data[0]), float(data[1])
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            return float(action[0]), float(action[1])
        value = float(action)
        return value, value

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
        if self.config.mock:
            return {
                "x": self._mock_state["x"],
                "y": self._mock_state["y"],
                "yaw": self._mock_state["yaw"],
                "vx": self._mock_state["vx"],
                "vy": self._mock_state["vy"],
            }
        if self._data is None:
            return {}
        if np is None:
            return {
                "qpos": list(self._data.qpos),
                "qvel": list(self._data.qvel),
            }
        return {"qpos": self._data.qpos.copy(), "qvel": self._data.qvel.copy()}

    def _get_image(self) -> Any:
        if self.config.mock:
            return self._mock_image()
        if self._renderer is None or self._data is None:
            if not self._warned_renderer:
                if not self.config.render:
                    message = "rendering disabled"
                else:
                    message = self._renderer_error or "Renderer unavailable"
                print(f"warning: camera disabled ({message}); returning black frames")
                self._warned_renderer = True
            return self._mock_image()
        self._renderer.update_scene(self._data, camera=self.config.camera.name)
        image = self._renderer.render()
        if np is None:
            return image.tolist()
        return image

    def _mock_image(self) -> Any:
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
