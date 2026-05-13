# AI Model Training and NS-3 Inference

## Overview
This document describes how the `ns3_migration_v2_gpu.zip` model is generated, what AI code is used, and how the trained model is used for inference with the NS-3 disaster recovery simulation.

The AI solution is based on Stable Baselines3 PPO and a custom Gym environment that communicates with the NS-3 simulation through POSIX shared memory at `/dev/shm/ns3ai-shm`.

## Key Files
- `train_dr.py` — training script that generates `ns3_migration_v2_gpu.zip`
- `train_dr_v2.py` — later normalized/refinement training script for `ns3_migration_v3_normalized`
- `dr_env.py` — Gym environment used for offline inference and evaluation
- `test_model.py` — inference runner that loads the trained model and drives the simulation
- `compare_results.py` — test harness comparing AI inference against fixed-rate baseline
- `dr_sim.cc` — NS-3 simulation entry point that reads commands from shared memory
- `shm_types.h` / `shm_types.py` — shared memory structure definitions used by both C++ and Python

## How `ns3_migration_v2_gpu.zip` Is Generated

### Training entry point
The model is produced by running `train_dr.py`.

In `train_dr.py`, the script:
1. Defines `NS3MigrationEnv`, a custom Gym environment that:
   - attaches to shared memory at `/dev/shm/ns3ai-shm`
   - reads `queue_depth`, `dropped_reqs`, `total_finished`, and a cursor from shared memory
   - issues migration batches by writing `cmd_start_id`, `cmd_end_id`, and `trigger = 1`
2. Constructs a PPO agent:
   - policy: `MlpPolicy`
   - device: `cuda` if available, otherwise `cpu`
   - learning_rate: `3e-4`
   - batch_size: `64`
   - n_steps: `2048`
3. Calls `model.learn(total_timesteps=150000, tb_log_name="migration_v2_optimized", reset_num_timesteps=False)`
4. Saves the trained policy with:
   - `model.save("ns3_migration_v2_gpu")`

Because Stable Baselines3 appends `.zip` when saving, the resulting file is `ns3_migration_v2_gpu.zip`.

### Observation and action spaces used during training
- `action_space = gym.spaces.Discrete(32)`
  - This encodes a batch size from 1 to 32 nodes per decision.
- `observation_space = gym.spaces.Box(low=0, high=1e9, shape=(4,), dtype=np.float32)`
  - Observations are:
    1. logical cursor position
    2. queue depth
    3. dropped request count
    4. total finished count

### Reward function in `train_dr.py`
The PPO agent is rewarded based on:
- positive progress from batch size
- bonus for newly finished migrations
- penalty for dropped requests using `np.log1p(new_drops) * 5.0`
- penalty when `queue_depth > 20000`

This reward encourages the agent to maximize throughput while preventing the DR core queue from becoming overloaded.

## How the AI Model Is Used for Inference in NS-3

### Shared memory handshake
The AI and NS-3 communicate through a shared memory region defined in `shm_types.h` and mapped in Python by `shm_types.py`.

Shared memory fields used for control:
- `cmd_start_id` — start index of the current batch
- `cmd_end_id` — end index of the current batch
- `trigger` — set to `1` by Python to tell NS-3 to process the batch

NS-3 polling behavior is implemented in `dr_sim.cc`:
- When `shm->trigger == 1`, NS-3 reads `cmd_start_id` and `cmd_end_id`
- It schedules attach requests for the requested nodes
- It resets `shm->trigger = 0` after the batch is accepted

### Inference runner
The actual inference is performed by `test_model.py`.

`test_model.py` does the following:
1. Builds a `DrMigrationEnv` environment with an existing shared memory object
2. Loads the trained model with `PPO.load("ns3_migration_v3_normalized")`
3. Resets the environment and enters a loop where it:
   - calls `model.predict(obs, deterministic=True)`
   - writes the action into shared memory by invoking `env.step(action)`
   - waits for NS-3 to clear the trigger
   - updates the logical cursor and observes the next state
4. After the main loop, it waits up to 20 seconds for the queue to drain and records final metrics

Note: `test_model.py` currently loads the `ns3_migration_v3_normalized` model. If you want to use `ns3_migration_v2_gpu.zip`, replace the load path with `PPO.load("ns3_migration_v2_gpu")`.

### Environment used for inference
`DrMigrationEnv` in `dr_env.py` defines inference-time observations and reward logic for offline compatibility. It is built to work with shared memory directly, not requiring Gym training loops during inference.

Observation features are:
- normalized queue depth
- average RRC limit of the next few nodes
- remaining nodes ratio
- completed nodes ratio

Actions represent batch sizes from `1` to `128`.

## Running Inference in the Project

### 1. Start NS-3 simulation
Run the NS-3 binary that creates and initializes `/dev/shm/ns3ai-shm`. Examples in this repository include `dr_sim.cc` and `dr_sim_compare.cc`.

### 2. Run the AI inference runner
From the project root:

```bash
python3 test_model.py
```

Or use the comparison harness:

```bash
python3 compare_results.py
```

`compare_results.py` will:
- reset shared memory state
- run a fixed-rate baseline test
- run the AI inference test
- generate a dashboard image `scalability_report_v2.png`

## Notes and Best Practices
- The model generation path is confirmed in `train_dr.py`.
- The inference loop relies on NS-3 clearing `trigger` back to `0`.
- `ns3_migration_v2_gpu.zip` is the trained PPO model saved by Stable Baselines3.
- If you need to use the v2 model for inference, update `test_model.py` or any loader to use the file name without `.zip`.

## Recommended file references
- Training: `train_dr.py`
- Inference environment: `dr_env.py`
- Inference execution: `test_model.py`
- Comparison harness: `compare_results.py`
- NS-3 bridge: `dr_sim.cc`, `dr_sim_compare.cc`, `shm_types.h`
