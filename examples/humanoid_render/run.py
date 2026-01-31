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
from robotwins.sim.mujoco.runtime.factory import make_env

try:
    import mujoco  # type: ignore[import-not-found]
    import mujoco.viewer  # type: ignore[import-not-found]
except Exception:
    mujoco = None


def _env_timestep(env: object) -> float:
    get_timestep = getattr(env, "get_timestep", None)
    if callable(get_timestep):
        try:
            return float(get_timestep())
        except Exception:
            return 0.02
    return 0.02


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
        default="examples/humanoid_render/config.yaml",
        help="Path to run config.",
    )
    parser.add_argument("--steps", type=int, default=250, help="Number of steps to run.")
    parser.add_argument("--viewer", action="store_true", help="Open a MuJoCo viewer window.")
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
    env = make_env(run_cfg)

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

        if args.viewer:
            if mujoco is None:
                print("viewer disabled: mujoco.viewer not available")
            else:
                try:
                    model, data = env.get_mujoco_handles()  # type: ignore[attr-defined]
                    viewer = mujoco.viewer.launch_passive(model, data)
                except Exception as exc:
                    print(f"viewer disabled: {exc}")
                    print("hint: on macOS, run with `mjpython` to open a viewer window")

        for _ in range(args.steps):
            # Zero action keeps the humanoid stable-ish in default pose.
            action = [0.0] * getattr(env, "action_size", 0)
            obs, _reward, _done, _info = env.step(action)
            if viewer is not None:
                viewer.sync()
                time.sleep(_env_timestep(env))

        if viewer is not None and running and args.keep_open:
            _write_pid(pid_file)
            print("viewer running; press Ctrl+C or re-run script to replace it")
            while running:
                viewer.sync()
                time.sleep(_env_timestep(env))

        print("run completed")
    finally:
        if viewer is not None:
            viewer.close()
        _clear_pid(pid_file)
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
