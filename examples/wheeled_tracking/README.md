# Wheeled Tracking (Vision)

This is a vision-first wheeled example scaffold. It defines a camera-based observation
and expects a MuJoCo model to be added at `examples/wheeled_tracking/models/`.

Files:

- `config.yaml` defines the run and camera settings.
- `run.py` shows a minimal entrypoint for wiring the env.
- `train.py` runs a tiny random-policy training loop.
- `logs/sample_run.csv` is a minimal SysID log example (see `robotwins/sysid/schema.py`).

Next steps:

- add an MJCF model at `examples/wheeled_tracking/models/wheeled.xml`
- implement MuJoCo loading and camera capture in `robotwins/sim/mujoco/runtime/wheeled_env.py`

Viewer note (macOS):
- `mujoco.viewer.launch_passive` requires running your script with `mjpython` to open a GUI window.
