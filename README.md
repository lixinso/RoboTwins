# RoboTwins - Physics-First Sim2Real Robot Learning

**RoboTwins** is a physics-first sim2real toolkit that helps you build **digital twins** of robots (calibrated simulation models) and train **RL control policies** that transfer to real-world hardware.

Tagline: **Train in simulation. Win in reality.**

## Why RoboTwins exists

Simulation is cheap and safe, but policies often fail on real robots due to the reality gap:

- contact and friction mismatch
- actuator saturation and bandwidth limits
- control and sensor latency and jitter
- sensor noise and bias
- small modeling errors (mass and COM, damping)

RoboTwins makes calibration, randomization, and evaluation first-class citizens so you can systematically reduce the reality gap instead of hoping PPO "just works."

## Key ideas

- Twin-first workflow: calibrate your sim to match real logs before training.
- Transfer-first RL: train with realistic constraints (delay, noise, saturation).
- Reproducibility: every run is config-driven, seeded, and reportable.
- Viewer optional: train headless for speed; attach a viewer (MuJoCo viewer now, Unity viewer later).

## Project goals (v1)

From a minimal input set:

- robot model (MJCF or URDF)
- a small set of real logs (CSV or rosbag converted)
- task definition (env)

...produce:

- a calibrated twin (best-fit physics parameters)
- a robust policy trained under randomized physics, sensors, and timing
- an evaluation report showing sim vs real gap and transfer success

### What "done" looks like (v1 success metrics)

- Reality Gap Score improves after calibration (trajectory mismatch decreases).
- Transfer success rate on a simple real task exceeds a baseline controller.
- Time-to-first-transfer: a developer can go from clone to trained policy to evaluation report in a weekend.

## Scope (v1)

### In scope

1. Simulator backend: MuJoCo first (Mac-friendly, reliable rigid-body + contact)
2. Twin parameterization: friction, damping, motor strength scaling, latency, sensor noise
3. System ID calibration: replay real command sequences in sim and fit parameters
4. Domain randomization: parameter distributions + curriculum scheduler
5. RL training harness: PPO baseline (SAC optional later)
6. Evaluation suite: standard metrics + sim-vs-real comparisons + regression seeds
7. UI (v1): MuJoCo viewer toggle + simple telemetry streaming (optional)
8. UI (future): Unity render client visualization

### Not in scope (v1)

- New physics engine
- Photorealistic rendering or synthetic data pipelines
- Full autonomy stack (SLAM, planning)
- Multi-robot fleet infrastructure

## User stories (v1)

1. As a robotics engineer, I can calibrate friction and latency so sim tracks real joint trajectories.
2. As an RL practitioner, I can train a policy with randomized dynamics to be robust.
3. As a developer, I can attach a viewer to watch a single rollout and inspect contacts and torques.
4. As a researcher, I can reproduce results with configs and seeds and get a report artifact.

## High-level workflow

1. Define robot and task
   - Import MJCF (preferred) or URDF.
   - Define a Gymnasium-style task environment.
2. Collect real logs (minimum)
   - timestamps, commanded actions, measured joint positions and velocities, optional IMU.
3. Calibrate the twin (System ID)
   - Replay the command sequence in sim.
   - Optimize parameters to reduce mismatch.
4. Train policy with randomized twin
   - Use calibrated parameters as the center.
   - Randomize around them and include delay, noise, and saturation.
5. Evaluate sim2real
   - Run a fixed evaluation suite in sim.
   - Deploy to robot via an adapter.
   - Generate a sim-vs-real report.

## Technical architecture (v1)

MuJoCo is first, but everything above the sim layer should work with other backends later. See `docs/ARCHITECTURE.md` for full details.

Layers:

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

## CLI (recommended v1 commands)

- `robotwins init` - scaffold example project
- `robotwins calibrate --log data/run01.csv --out twin.yaml`
- `robotwins train --config train.yaml`
- `robotwins eval --suite core --out reports/`
- `robotwins viewer --run last` (debug)

## First tasks to build (recommended order)

1. Project skeleton + config + seeding
2. One MuJoCo environment (simple reaching or cartpole-like torque-limited)
3. Viewer toggle for a single env
4. Trajectory logger (sim rollouts saved to disk)
5. SysID minimal:
   - load a real-like dataset (even synthetic at first)
   - calibrate friction + motor scale + delay
6. Domain randomization around calibrated params
7. PPO baseline + evaluation suite
8. Sim2real report generation

## Docs

- `docs/PRD.md` - product requirements and v1 scope
- `docs/ARCHITECTURE.md` - technical design and system breakdown

## Getting started (current choice)

First example: wheeled with vision. See `examples/wheeled_tracking/`.
