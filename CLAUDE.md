# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Racecar Gym is a gymnasium-compatible RL environment for miniature F1Tenth-like racecars using the PyBullet physics engine. Supports single-agent and multi-agent racing with PettingZoo API compatibility.

## Setup & Commands

```bash
# Install (editable mode)
pip install -e .

# Run tests
python -m pytest tests/

# Run a single example
python examples/gym_examples/simple_usage.py
```

No build step. No linter configured. Track assets download automatically on first use, or manually via `models/scenes/` with wget.

## RL Training Pipeline (`training/`)

A complete RL training and evaluation pipeline using **Stable-Baselines3 (PPO)**.

### Commands
- **Train (Headless):** `python training/train.py --num-cpu 4 --total-timesteps 1000000`
- **Train (Visual/Debug):** `python training/train.py --num-cpu 1 --render`
- **Evaluate (Follow-Cam):** `python training/enjoy.py training/checkpoints/ppo_step_XXXXX.zip`
- **Resume:** `python training/train.py --resume training/checkpoints/ppo_step_XXXXX.zip --total-timesteps 500000`

### Architecture Highlights
- **Model:** `MultiInputPolicy` using a custom `RacecarFeaturesExtractor`.
    - **Lidar Branch:** 1D-CNN (three Conv1d layers) for processing 1080 rays into a latent spatial vector.
    - **Sensor Branch:** MLP for Velocity, Acceleration, Time, and Progress Sensor data.
- **Environment Wrapper:** `MultiInputRacecarEnv`
    - **Obs Filter:** Removes `pose` (global world coordinates) to reduce noise.
    - **Progress Sensor:** Adds `progress_sensor` (scalar 0-1) to inform the agent of track location without absolute coordinates.
    - **Stuck Detection:** Truncates episode if forward progress isn't made for 150 consecutive steps (~3s).
    - **Action Remapping:** Policy output `0` maps to `0.3` throttle (mild forward bias) to encourage exploration.
- **Positioning:** `random_biased` mode (80% forward, 20% backward, +/- 15° orientation noise).
- **Training Config:** `scenarios/austria_train.yml` (terminate_on_collision=True, high progress reward).

## Core Improvements

- **Steering:** Increased `steering_multiplier` to 1.0 in `models/vehicles/racecar/racecar.yml` to allow full 24° steering lock (was previously restricted to 12°).
- **Reward Function:** Fixed `racecar_gym/tasks/progress_based.py` to remove `abs()` from progress delta, correctly handling laps and discouraging "backward" rewards.
- **Dataclasses:** Fixed mutable default issues in `racecar_gym/core/specs.py` using `field(default_factory=...)`.
- **Compat:** Added `get_wrapper_attr()` to `SingleAgentRaceEnv` and wrappers for SB3/Gymnasium compatibility.

## Architecture

### Core Layer (`racecar_gym/core/`)
Abstract interfaces defining the simulation contract:
- **World** (ABC): simulation lifecycle (init/reset/update/state), starting positions, rendering
- **Agent**: binds a Vehicle + Task together, orchestrates observe → control → reward → done
- **Vehicle/Sensor/Actuator** (ABCs): hardware abstraction for the racecar
- **specs.py**: YAML-backed dataclasses (`ScenarioSpec`, `AgentSpec`, `VehicleSpec`, `TaskSpec`) loaded via `yamldataclassconfig`
- **GridMap** (`gridmaps.py`): 2D map lookups (progress, obstacle distance, occupancy) from `.npz` files

### Bullet Implementation (`racecar_gym/bullet/`)
PyBullet-backed concrete implementations of core interfaces:
- **world.py**: physics stepping, collision detection, race state tracking (laps, checkpoints, wrong-way detection, rankings)
- **vehicle.py**: loads URDF models, manages sensors/actuators
- **sensors.py / actuators.py**: PyBullet sensor readings and motor/steering control
- **positioning.py**: starting position strategies (grid, random, random_bidirectional, random_ball, random_biased)
- **providers.py**: factory functions `load_world()` and `load_vehicle()` that wire configs to implementations

### Task System (`racecar_gym/tasks/`)
Registry-based task system. Tasks define reward functions and termination conditions:
- `maximize_progress` — primary task, progress along track centerline
- `maximize_progress_action_reg` — progress with action regularization
- `maximize_progress_ranked` — rank-discounted progress (multi-agent)
- `max_tracking` — waypoint following

Register new tasks via `register_task(name, class)` in `__init__.py`.

### Environment Layer (`racecar_gym/envs/`)
- **scenarios.py**: `SingleAgentScenario` / `MultiAgentScenario` — load YAML scenario files, wire up World + Agents
- **gym_api/**: Gymnasium env classes (`SingleAgentRaceEnv`, `MultiAgentRaceEnv`, vectorized variants, `ChangingTrack*` variants). Envs are auto-registered from YAML files in `scenarios/`.
- **gym_api/wrappers/**: observation flattening, action repeat, reset mode wrappers
- **pettingzoo_api/**: PettingZoo parallel environment wrapper for multi-agent

### Configuration Flow
Scenario YAML (`scenarios/*.yml`) → `ScenarioSpec` dataclass → `Scenario.from_spec()` → creates World + Agents with loaded vehicles and tasks. Vehicle configs live in `models/vehicles/racecar/racecar.yml`. Track scenes (SDF + maps) in `models/scenes/<track>/`.

### Env ID Convention
Auto-registered as `SingleAgent<Track>-v0` and `MultiAgent<Track>-v0` (e.g., `SingleAgentAustria-v0`). Available tracks: austria, barcelona, berlin, circle_cw/ccw, columbia, gbr, montreal, plechaty, torino, treitlstrasse.

## Key Dependencies

- **gymnasium** (not old `gym`) — env base classes and registration
- **pybullet** — physics simulation
- **stable-baselines3** — RL algorithms
- **torch** — backend for SB3 and CNN
- **pettingzoo** — multi-agent API
- **yamldataclassconfig** — YAML → dataclass config loading
- **numpy**, **scipy** — numerics and map processing
