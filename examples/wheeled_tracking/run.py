import argparse
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from robotwins.core.config import load_run_config
from robotwins.sim.mujoco.runtime.wheeled_env import WheeledVisionConfig, WheeledVisionEnv

try:
    import mujoco
    import mujoco.viewer
except Exception:
    mujoco = None


def _terminate_existing_viewer(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pid_file.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
        _wait_for_exit(pid, timeout=2.0)
    except ProcessLookupError:
        pass
    except PermissionError:
        print("warning: could not terminate existing viewer process")
    pid_file.unlink(missing_ok=True)


def _wait_for_exit(pid: int, timeout: float) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _write_pid(pid_file: Path) -> None:
    try:
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        return


def _clear_pid(pid_file: Path) -> None:
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pid = None
        if pid is None or pid == os.getpid():
            pid_file.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/wheeled_tracking/config.yaml",
        help="Path to run config.",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to run.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force mock mode even if MuJoCo is installed.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable camera rendering (use black frames).",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a MuJoCo viewer window.",
    )
    parser.add_argument(
        "--keep-open",
        dest="keep_open",
        action="store_true",
        help="Keep the viewer open after stepping (default).",
    )
    parser.add_argument(
        "--no-keep-open",
        dest="keep_open",
        action="store_false",
        help="Close the viewer immediately after stepping.",
    )
    parser.set_defaults(keep_open=True)
    args = parser.parse_args()

    pid_file = Path(tempfile.gettempdir()) / "robotwins_viewer.pid"
    _terminate_existing_viewer(pid_file)

    run_cfg = load_run_config(args.config)
    if args.mock:
        run_cfg.mujoco.mock = True
    if args.no_render:
        run_cfg.mujoco.render = False

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
    env = WheeledVisionEnv(env_cfg)
    viewer = None
    running = True

    def _handle_signal(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        obs = env.reset(seed=run_cfg.seed)
        print(f"reset obs keys: {list(obs.keys())}")
        if args.viewer and not run_cfg.mujoco.mock:
            if mujoco is None:
                print("viewer disabled: mujoco.viewer not available")
            else:
                try:
                    model, data = env.get_mujoco_handles()
                    viewer = mujoco.viewer.launch_passive(model, data)
                except Exception as exc:
                    print(f"viewer disabled: {exc}")
                    print("hint: on macOS, run with `mjpython` to open a viewer window")
        for _ in range(args.steps):
            action = [0.5, 0.5]
            obs, _reward, _done, _info = env.step(action)
            if viewer is not None:
                viewer.sync()
                time.sleep(env.get_timestep())
        if viewer is not None and running and args.keep_open:
            _write_pid(pid_file)
            print("viewer running; press Ctrl+C or re-run script to replace it")
            while running:
                viewer.sync()
                time.sleep(env.get_timestep())
        print("run completed")
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        print(str(exc))
        return 1
    finally:
        if viewer is not None:
            viewer.close()
        _clear_pid(pid_file)
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
