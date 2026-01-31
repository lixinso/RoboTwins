# Humanoid Render (MuJoCo)

This example renders a MuJoCo humanoid.

By default it loads Gymnasium's packaged `humanoid.xml` (so we don't vendor the model file).

## Run (headless)

```bash
python examples/humanoid_render/run.py --steps 250
```

## Run with viewer (macOS)

MuJoCo viewer on macOS typically requires launching via `mjpython`:

```bash
mjpython examples/humanoid_render/run.py --viewer --steps 250
```

## Customize model

Set `mujoco.model_path` in `config.yaml` to an MJCF file path.
