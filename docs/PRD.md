# RoboTwins PRD (v1)

## One-liner

RoboTwins is a physics-first sim2real toolkit that builds calibrated digital twins and trains RL policies that transfer to real robot hardware.

## Problem

Policies trained in simulation often fail on real robots due to the reality gap:

- contact and friction mismatch
- actuator saturation and bandwidth limits
- control and sensor latency and jitter
- sensor noise and bias
- small modeling errors (mass and COM, damping)

RoboTwins exists to make calibration, randomization, and evaluation first-class workflows so transfer can be measured and improved.

## Goals (v1)

From a minimal input set:

- robot model (MJCF or URDF)
- a small set of real logs (CSV or rosbag converted)
- task definition (env)

...produce:

- a calibrated twin (best-fit physics parameters)
- a robust policy trained under randomized physics, sensors, and timing
- an evaluation report showing sim vs real gap and transfer success

## Success metrics (v1)

- Reality Gap Score improves after calibration (trajectory mismatch decreases).
- Transfer success rate on a simple real task exceeds a baseline controller.
- Time-to-first-transfer: a developer can go from clone to trained policy to evaluation report in a weekend.

## In scope (v1)

1. Simulator backend: MuJoCo first
2. Twin parameterization: friction, damping, motor strength scaling, latency, sensor noise
3. System ID calibration: replay real command sequences in sim and fit parameters
4. Domain randomization: parameter distributions + curriculum scheduler
5. RL training harness: PPO baseline (SAC optional later)
6. Evaluation suite: standard metrics + sim-vs-real comparisons + regression seeds
7. UI (v1): MuJoCo viewer toggle + simple telemetry streaming (optional)
8. UI (future): Unity render client visualization

## Not in scope (v1)

- New physics engine
- Photorealistic rendering or synthetic data pipelines
- Full autonomy stack (SLAM, planning)
- Multi-robot fleet infrastructure

## User stories (v1)

1. As a robotics engineer, I can calibrate friction and latency so sim tracks real joint trajectories.
2. As an RL practitioner, I can train a policy with randomized dynamics to be robust.
3. As a developer, I can attach a viewer to watch a single rollout and inspect contacts and torques.
4. As a researcher, I can reproduce results with configs and seeds and get a report artifact.

## Constraints

- MuJoCo should run headless for training and optionally attach a viewer.
- All runs are config-driven and seeded for reproducibility.
- Logs and reports are written in a simple, parseable schema (CSV/JSON/Markdown).

## Deliverables (v1)

- MuJoCo environment wrapper and one example task
- Config system with seeding and run registry
- Viewer toggle for a single environment
- Trajectory logger (sim rollouts saved to disk)
- SysID MVP: load real-like logs and calibrate friction + motor scale + delay
- Domain randomization around calibrated params
- PPO baseline training harness
- Evaluation suite with sim-vs-real report artifact

## Roadmap

### v0.1 (first runnable)

- MuJoCo env wrapper + one task
- config system + seeding
- basic viewer toggle
- simple PPO training loop

### v0.2 (twin + sysid MVP)

- log ingestion + replay
- optimize friction/motor scale/delay
- calibration report

### v0.3 (randomization + curriculum)

- parameter distributions
- curriculum scheduler
- robustness eval suite

### v0.4 (deploy adapter + safety)

- hardware adapter interface
- action clamps + watchdog

### v0.5 (Unity visualization client)

- state streaming protocol
- Unity scene mirroring robot transforms
- overlays + controls

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

## Open choices

- First example robot: arm, legged, or wheeled
- Observations: state-only or vision
