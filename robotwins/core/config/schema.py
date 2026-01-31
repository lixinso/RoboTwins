from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    name: str = "front"
    width: int = 128
    height: int = 128
    fps: int = 30


@dataclass
class EnvConfig:
    name: str = "wheeled_tracking"
    obs: str = "vision"
    control_hz: int = 50
    camera: CameraConfig = field(default_factory=CameraConfig)
    frame_stack: int = 2
    include_state: bool = True


@dataclass
class MujocoConfig:
    model_path: str = "examples/wheeled_tracking/models/wheeled.xml"
    mock: bool = True
    render: bool = True


@dataclass
class HumanoidConfig:
    # Rendering helper: keep the humanoid in the reset (upright) pose.
    # This is not a physics-based balance controller; it is intended for visualization.
    pose_lock: bool = False
    stabilize: bool = True
    kp: float = 50.0
    kd: float = 5.0


@dataclass
class RunConfig:
    seed: int = 0
    env: EnvConfig = field(default_factory=EnvConfig)
    mujoco: MujocoConfig = field(default_factory=MujocoConfig)
    humanoid: HumanoidConfig = field(default_factory=HumanoidConfig)
    algo: str = "ppo"
    notes: str = ""
