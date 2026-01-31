# RoboTwins Architecture (v1)

## Design principles

- Simulator-agnostic APIs above the sim layer (MuJoCo first).
- Twin-first workflow: calibrate before training.
- Transfer-first training: model real-world limits explicitly.
- Reproducibility by default: configs, seeds, and report artifacts.
- Headless-first performance with optional viewer attach.

## Layered architecture

1. Core (configs, registry, logging, determinism)
2. Sim backend (MuJoCo step/reset/render/state)
3. Twin (parameters + randomization + application)
4. SysID (real log ingestion, loss functions, optimizer, reports)
5. RL (algorithms, trainers, policies, buffers)
6. Eval (suites, metrics, sim-vs-real comparison)
7. Deploy (hardware adapters + safety)
8. UI/Viewer (debug visualization + telemetry)

## Recommended repo layout

```
RoboTwins/
  README.md
  LICENSE
  docs/
    PRD.md
    ARCHITECTURE.md
    ROADMAP.md
    CONFIGS.md
  robotwins/                 # package name can be robotwins or robotwins_core; project name RoboTwins
    core/
      config/
      registry/
      logging/
      utils/
    sim/
      interfaces.py
      mujoco/
        model/
        runtime/
        viewer/
    twin/
      parameters.py
      randomization/
        distributions.py
        curriculum.py
        apply.py
    sysid/
      datasets.py
      objectives.py
      optimize.py
      reports.py
    rl/
      algos/
        ppo.py
      trainers/
      policies/
    eval/
      suites/
      metrics/
      compare_sim_real.py
      regression/
    deploy/
      adapters/
      safety/
    cli/
      main.py
      commands/
  examples/
    mjcf_arm_reach/
    wheeled_tracking/
  tests/
```

## Core data models

### TwinParameters

- DynamicsParams
  - link mass scale
  - COM offsets (optional v1)
  - friction (surface + joint)
  - restitution (optional v1)
- ActuatorParams
  - torque limits
  - motor strength scale
  - action rate limits (slew)
- TimingParams
  - control delay (ms)
  - sensor delay (ms)
  - jitter model (optional v1)
- SensorParams
  - noise std
  - bias drift (optional v1)

### RunConfig

- task or env
- robot model
- seeds
- algorithm and hyperparams
- randomization ranges
- sysid settings
- eval suite selection

## Sim backend interface

The sim backend abstracts the physics engine and exposes a minimal API used by higher layers:

- `reset(seed) -> obs`
- `step(action) -> obs, reward, done, info`
- `get_state()` / `set_state(state)`
- `render(mode)`
- `set_twin_params(params)`

The MuJoCo backend should implement the interface and provide viewer utilities.

## System ID (calibration)

### Inputs

Real log dataset:

- `t`: timestamps
- `u(t)`: commands (action)
- `y(t)`: observed joint states or sensors

### Replay in sim

- Sim uses the same command stream `u(t)` (with modeled delay).
- Sim produces predicted observations `y_hat(t)`.

### Objective functions (v1)

- Trajectory loss:
  - `L_pos = mean ||q_real - q_sim||^2`
  - `L_vel = mean ||qd_real - qd_sim||^2`
- Optional action or saturation penalty to discourage unrealistic actuator behavior.
- Optional contact timing loss for tasks where contact matters.

### Optimization (v1)

Start with a gradient-free optimizer:

- CMA-ES or Nelder-Mead (pick one first)

Outputs:

- best-fit parameters
- report with before/after gap score

## Domain randomization engine

### Parameter groups

- dynamics: friction, damping, mass scale
- actuators: motor strength, torque limits, rate limits
- timing: delay and jitter (bounded)
- sensors: Gaussian noise + bias (optional)

### Curriculum

- Stage 0: narrow ranges (stabilize learning)
- Stage 1: widen core dynamics
- Stage 2: widen timing + sensors
- Stage 3: stress-test perturbations (disturbances)

## RL training harness (v1)

### Algorithm

- PPO baseline (stable default for continuous control)

### Transfer-critical constraints

- control frequency matching real robot
- action smoothing and rate limits
- saturation modeling
- observation noise and latency

### Outputs

- trained policy weights
- training metrics
- evaluation rollouts (seeded)
- exported policy format (optional later: ONNX)

## Evaluation and sim2real report

### Sim evaluation suite

- fixed seeds
- standard initial conditions
- randomization off and on
- robustness tests (disturbances)

### Real evaluation suite

- same scenario scripts (as close as possible)
- record logs in the same schema
- compute Reality Gap Score per test

### Report artifact (Markdown + JSON)

- twin parameters (calibrated + ranges)
- training config (hash)
- sim metrics
- real metrics
- sim vs real comparisons:
  - trajectory overlays
  - success rate
  - failure modes summary

## UI and visualization

### v1 (fast)

- MuJoCo viewer for debug
  - toggle viewer for a single env
  - pause, step, reset
- optional telemetry stream:
  - WebSocket output of reward/action norms/joint plots

### v2 (Unity viewer)

Unity acts as a render client:

- physics stays in MuJoCo (source of truth)
- Unity subscribes to robot/world transforms (ZMQ or UDP)
- Unity provides:
  - visuals
  - overlays (contacts, torques)
  - controls (reset, seed, scenario)

This keeps sim2real integrity while providing a platform UI.
