# TurtleBot3 DRL Local Planner

This package contains the DRL local planner research framework for the warehouse AMR simulation. The current baseline is already working in `turtlebot3_gazebo`: Gazebo simulation, SLAM, map saving, Nav2 localization, and RViz navigation goals.

The goal here is to train and evaluate a DRL local planner in Gazebo, then later integrate it as a replacement for only the Nav2 local controller layer. The package does not replace map server, AMCL, global planning, mission logic, dashboard integration, warehouse order flow, or QR confirmation.

## Current Milestone

- Gazebo simulation works.
- SLAM works.
- Map saving works.
- Nav2 works with saved maps.
- RViz goal execution works.
- DRL local planner training and bypass simulation tests are active in this isolated package.
- Local-goal and hard-gap curriculum stages have completed in Gazebo.
- Difficult S-curve and winding-path stages still need more training.
- Real-robot DRL direct replacement is not validated yet.

## Safety Principle

The first inference mode is shadow mode. The DRL node subscribes to robot state, computes an action, and publishes to `/cmd_vel_drl`. It does not publish to `/cmd_vel` unless `publish_directly_to_cmd_vel:=true` is explicitly set.

Keep the Nav2 baseline available as fallback throughout training and evaluation.

## GAP_SAC Summary

The target algorithm is inspired by "Deep reinforcement learning for path planning of autonomous mobile robots in complicated environments". The paper proposes GAP_SAC: Gated Attention + Prioritized Experience Replay + Soft Actor-Critic.

Paper-derived components are implemented separately from ROS/Gazebo engineering adaptations. Any downsampling, thresholds, topic names, safety stops, or shadow-mode behavior are implementation adaptations for TurtleBot3/Gazebo.

## Paper-Derived Equations

State:

```text
s_t = [LiDAR_t, DG_t, DO_t, AO_t, MA_t]
```

Implementation adaptation:

```text
state = [downsampled_lidar, DG, DO, AO, MA]
```

Action:

```text
action = [v, w]
```

Default scaling:

```yaml
max_linear_velocity: 0.22
max_angular_velocity: 2.84
allow_reverse: false
```

Reward:

```text
G_t(s, a) =
    R0,            if DG = 0
    R4,            if DO = 0
    R1 * R2 + R3,  if DG != 0 and DO != 0
```

Implementation adaptation:

```python
if DG <= goal_tolerance:
    reward = R0
elif DO <= collision_distance:
    reward = R4
else:
    reward = R1 * R2 + R3
```

Terms:

```text
R1 = lambda_heading * (pi / 3 - abs(MA))
R2 = lambda_distance * (previous_DG / max(current_DG, eps) - 1)
R3 = lambda_obstacle if DO < min_obstacle_distance else 0
```

The paper table lists `(R0, R1) = (1000, -800)`, but based on the paper's reward definition, `-800` is the collision penalty `R4`. This implementation treats `-800` as `collision_penalty_R4`.

PER:

```text
td_error = abs(target_q - current_q)
priority = (abs(td_error) + per_epsilon) ** per_alpha
P(i) = priority / total_priority
```

Gated attention:

```python
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
attention_weights = softmax(scores)
attention_output = attention_weights @ V
transformed = layer_norm(linear(attention_output) + attention_output)
gate = sigmoid(gate_layer(transformed))
gated_output = transformed * gate
```

Soft target update:

```python
target_param = tau * source_param + (1 - tau) * target_param
```

## Package Structure

```text
turtlebot3_drl/
├── config/gap_sac.yaml
├── launch/
│   ├── train_gap_sac.launch.py
│   ├── eval_gap_sac.launch.py
│   └── drl_controller.launch.py
├── turtlebot3_drl/
│   ├── agents/
│   ├── envs/
│   ├── nodes/
│   └── utils/
└── README.md
```

## Build

```bash
cd ~/tb3_sim_ws
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 -c "import turtlebot3_drl"
```

PyTorch is required for training and policy inference:

```bash
python3 -m pip install torch
```

Do not silently replace GAP_SAC with another algorithm if PyTorch is unavailable.

## Training

Start the Gazebo baseline separately, then run:

```bash
ros2 launch turtlebot3_drl train_gap_sac.launch.py
```

Automatic curriculum training is enabled by default. It progresses through fixed
intermediate goals only after the recent success and collision gates pass:

```bash
ros2 launch turtlebot3_drl train_gap_sac.launch.py \
  episodes:=500 \
  curriculum:=true \
  start_stage:=stage_1 \
  resume_checkpoint:=/home/hug/tb3_sim_ws/runs/gap_sac_20260429_093042/checkpoint_ep_99.pt
```

Training publishes to `/cmd_vel` only if explicitly configured. Keep this disabled until the environment reset and safety behavior have been verified.

Logs and checkpoints are written under:

```text
runs/gap_sac_<timestamp>/
```

Run artifacts include `metrics.csv`, `stage_summary.csv`, `run_manifest.yaml`,
`best_checkpoint.pt`, `latest_checkpoint.pt`, and periodic `checkpoint_ep_<N>.pt`
files.

Each episode now prints a compact progress line with reward, steps, terminal reason,
minimum obstacle distance, active curriculum stage, rolling success rate, and
episode duration. This is the primary runtime signal that training is still active.

## Evaluation

```bash
ros2 run turtlebot3_drl eval_policy \
  --config $(ros2 pkg prefix turtlebot3_drl)/share/turtlebot3_drl/config/gap_sac.yaml \
  --checkpoint <path_to_checkpoint>
```

Metrics include success rate, collision rate, timeout rate, reward, path length, minimum obstacle distance, steps, and duration. A Nav2 baseline CSV can be compared later.

## Shadow-Mode Inference

```bash
ros2 launch turtlebot3_drl drl_controller.launch.py \
  policy_checkpoint:=<path_to_checkpoint>
```

Verify the shadow output:

```bash
ros2 topic echo /cmd_vel_drl
```

In normal operation the DRL controller follows user/Nav2 goals from `/goal_pose`
or the latest endpoint in `/plan`. The fixed fallback goal is only for isolated
Gazebo policy tests:

```bash
ros2 launch turtlebot3_drl drl_controller.launch.py \
  policy_checkpoint:=<path_to_checkpoint> \
  use_fixed_goal_when_missing:=true \
  fixed_goal_x:=-0.3 \
  fixed_goal_y:=0.0
```

To let the robot move in Gazebo without Nav2, direct `/cmd_vel` publishing must be
explicitly enabled:

```bash
ros2 launch turtlebot3_drl drl_controller.launch.py \
  policy_checkpoint:=<path_to_checkpoint> \
  publish_directly_to_cmd_vel:=true \
  use_fixed_goal_when_missing:=true \
  fixed_goal_x:=-0.3 \
  fixed_goal_y:=0.0
```

Default safety behavior:

- no scan, odom, or goal: publish zero velocity
- obstacle closer than `safety_stop_distance`: publish zero velocity
- no checkpoint loaded: publish zero velocity
- `/cmd_vel` is not used unless explicitly enabled

Optional launch arguments:

```text
config_file:=<path_to_gap_sac.yaml>
policy_checkpoint:=<path_to_checkpoint>
input_goal_topic:=/goal_pose
input_plan_topic:=/plan
output_cmd_vel_topic:=/cmd_vel_drl
publish_directly_to_cmd_vel:=false
use_fixed_goal_when_missing:=false
```

## Comparison Metrics Against Nav2

Use the same saved map, start pose, and goal set for both Nav2 and DRL:

- success rate
- collision rate
- timeout rate
- average reward
- average path length
- execution time
- minimum obstacle distance
- number of recovery or stop events

## Future Work

- Continue hard two-turn, S-curve, and winding-path curriculum training.
- Add map-aware safe pose and goal sampling for harder warehouse-style routes.
- Keep the bypass launch as the main DRL replacement validation path for now.
- Run real-robot shadow-mode validation before any direct DRL control.
- Treat native Nav2 controller plugin work as paused until the learned policy is stronger.

## Maintenance Log

### 2026-05-03

- Completed documentation update for thesis/report preparation.
- Recorded current DRL status: Gazebo training completed for local and hard-gap stages, with successful simulation-stage bypass tests.
- Recorded limitation: hard S-curve/winding routes still fail by timeout and need more curriculum training.
- Real-robot DRL deployment remains future work and must start in shadow mode.

### 2026-04-28

- Added `turtlebot3_drl` package skeleton.
- Added GAP_SAC defaults in `config/gap_sac.yaml`.
- Added paper-derived state, action, reward, PER, gated attention, and soft target update implementation.
- Added Gazebo RL environment scaffold.
- Added shadow-mode DRL controller publishing `/cmd_vel_drl` by default.
- Added training, evaluation, export, and launch entrypoints.
- Added checkpoint loading path for shadow-mode inference through `config_file` and `policy_checkpoint`.
- Added Gazebo reset cooldown, per-step timing, and per-episode console progress for training stability.
- Defaulted training to CPU to avoid CUDA driver issues on development laptops.
- Added `episodes:=...` passthrough in `train_gap_sac.launch.py` for short training/debug runs.
- Added deterministic Gazebo entity reset through `/set_entity_state` for `turtlebot3_waffle_pi`.
- Moved the default DRL training start pose away from the central obstacle and added reset collision retries.
- Lowered early Gazebo training velocity limits as a ROS/Gazebo adaptation while the policy is random.
- Hardened prioritized replay sampling so partially filled buffers cannot return uninitialized transitions.
- Added `initial_DG`, `final_DG`, `min_DG`, and `goal_progress` to training metrics.
- Added final checkpoint saving for short debug runs and reduced debug episode length to 250 steps.
- Switched the default training goal to a short stage-1 curriculum target before attempting longer routes.
- Added stage-1 reward shaping as ROS/Gazebo adaptations: relaxed goal tolerance, progress reward, near-goal bonus, softer obstacle penalty, step penalty, and timeout penalty.
- Added automatic curriculum training with staged fixed goals, promotion gates, stage summaries, run manifests, best/latest checkpoints, and full optimizer-state checkpoint metadata.

The DRL local planner follows the GAP_SAC formulation from the reference paper.

State:
s_t = [LiDAR_t, DG_t, DO_t, AO_t, MA_t]

where:
DG is the Euclidean distance to the goal,
DO is the distance to the nearest obstacle,
AO is the angle to the nearest obstacle,
MA is the heading angle required to face the target.

Reward:
G_t(s, a) =
    R0,                  if DG = 0
    R4,                  if DO = 0
    R1 × R2 + R3,        if DG ≠ 0 and DO ≠ 0

Heading reward:
R1 = λ1 × (π/3 − |α|)

Distance reward:
R2 = λ2 × (Dk / Dk+1 − 1)

Obstacle penalty:
R3 = λ3, if DO < minobs
R3 = 0,  if DO ≥ minobs

TD-error:
|ζi| = |Rt + γ min_i Qπ(s_{t+1}, a_{t+1}) − Q(s, a)|

Priority:
P(i) = δi / Σδi
δi = (|ζi + ε|)^α

Attention:
scores_ij = Qi Kj^T / sqrt(dk)
attention_weights = exp(scores_ij) / Σ exp(scores_ik)

Soft target update:
θtarget ← τθsource + (1 − τ)θtarget
