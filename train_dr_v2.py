import mmap
import ctypes
import os
import time
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- 1. MEMORY STRUCTURES (Matches shm_types.h) ---
class ShmNode(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("name", ctypes.c_char * 32),
        ("rrc_limit", ctypes.c_uint32),
        ("status", ctypes.c_uint8),
        ("processed_ue", ctypes.c_uint32)
    ]

class ShmCore(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("queue_depth", ctypes.c_uint32),
        ("dropped_reqs", ctypes.c_uint32),
        ("total_finished", ctypes.c_uint32)
    ]

class DrSharedMemory(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("node_count", ctypes.c_uint32),
        ("core", ShmCore),
        ("cmd_start_id", ctypes.c_uint32),
        ("cmd_end_id", ctypes.c_uint32),
        ("trigger", ctypes.c_uint8),
        ("wave_type", ctypes.c_uint8),
        ("nodes", ShmNode * 15000) # Increased to handle full dataset
    ]

SHM_PATH = "/dev/shm/ns3ai-shm"

# --- 2. REFINED ENVIRONMENT ---
class NS3MigrationEnv(gym.Env):
    def __init__(self, is_training=True):
        super().__init__()
        self.node_count = 13828 # Updated to your full dataset size
        self.cursor = 0
        self.mme_capacity = 5000.0 # Match your new C++ capacity for normalization
        
        # Increase Action Space to 128 to allow max throughput
        self.action_space = gym.spaces.Discrete(128)

        # Observation Space is now strictly 0.0 to 1.0 (Normalized)
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(4,), dtype=np.float32
        )

        print(f"Connecting to SHM at {SHM_PATH}...")
        while not os.path.exists(SHM_PATH):
            time.sleep(1)
        
        self.f = open(SHM_PATH, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), ctypes.sizeof(DrSharedMemory))
        self.shm = DrSharedMemory.from_buffer(self.mm)

    def _get_obs(self):
        """
        Refined Observations: Everything is scaled 0.0 - 1.0
        This allows the neural network to 'see' small changes in congestion.
        """
        norm_q = float(self.shm.core.queue_depth) / self.mme_capacity
        norm_cursor = float(self.cursor) / self.node_count
        norm_finished = float(self.shm.core.total_finished) / self.node_count
        
        # Next batch weight normalization (assuming max RRC around 250)
        batch_weight = np.mean([
            self.shm.nodes[i].rrc_limit for i in range(self.cursor, min(self.cursor + 10, self.node_count))
        ]) if self.cursor < self.node_count else 0
        norm_weight = float(batch_weight) / 250.0

        return np.array([
            norm_q, 
            norm_weight, 
            norm_cursor, 
            norm_finished
        ], dtype=np.float32)

    def step(self, action):
        batch_size = int(action) + 1
        start = self.cursor
        end = min(start + batch_size, self.node_count)

        # Snapshots
        initial_drops = self.shm.core.dropped_reqs
        initial_finished = self.shm.core.total_finished

        # Trigger
        self.shm.cmd_start_id = start
        self.shm.cmd_end_id = end
        self.shm.trigger = 1 

        # Wait for C++
        wait_start = time.time()
        while self.shm.trigger != 0:
            time.sleep(0.0001) 
            if time.time() - wait_start > 2.0:
                self.shm.trigger = 0
                break
        
        self.cursor = end
        
        # Metrics
        new_drops = self.shm.core.dropped_reqs - initial_drops
        newly_finished = self.shm.core.total_finished - initial_finished
        current_q_norm = float(self.shm.core.queue_depth) / self.mme_capacity

        # --- REFINED REWARD DESIGN ---
        # 1. Throughput Bonus: Reward larger successful batches
        reward = (batch_size / 128.0) * 2.0 
        
        # 2. Completion Bonus: High priority on nodes hitting 'STATUS_COMPLETED'
        reward += (newly_finished / 10.0) 
        
        # 3. Aggressive Drop Penalty: Scaled to discourage micro-bursts
        if new_drops > 0:
            reward -= (np.log1p(new_drops) * 5.0)
            
        # 4. Congestion Penalty: If queue > 80%, apply heavy friction
        if current_q_norm > 0.8:
            reward -= 10.0

        terminated = bool(self.cursor >= self.node_count)
        return self._get_obs(), float(reward), terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor = 0
        return self._get_obs(), {}

# --- 3. REFINEMENT TRAINING (TRANSFER LEARNING) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "./ppo_ns3_refinement/"
    os.makedirs(log_dir, exist_ok=True)

    env = NS3MigrationEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # --- THE FIX: ALWAYS START FRESH FOR NORMALIZED SCALING ---
    print("--- Starting Fresh Normalized Training (Scale: 0.0 - 1.0) ---")
    
    model = PPO(
        "MlpPolicy", 
        env, 
        device=device, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=3e-4,  # Standard LR for stable convergence
        batch_size=64,
        n_steps=2048
    )

    # Increase steps to 150,000 to ensure it masters the new scale
    print("--- Starting Training (150,000 steps) ---")
    try:
        model.learn(
            total_timesteps=150000, 
            tb_log_name="fresh_normalized_v3",
            reset_num_timesteps=True # Reset to start from 0 on the new scale
        )
    except KeyboardInterrupt:
        print("\nSaving current progress...")

    model.save("ns3_migration_v3_normalized")
    print("--- Training Complete. Model saved as 'ns3_migration_v3_normalized' ---")

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     log_dir = "./ppo_ns3_refinement/"
#     os.makedirs(log_dir, exist_ok=True)

#     env = NS3MigrationEnv()
#     env = Monitor(env)
#     env = DummyVecEnv([lambda: env])

#     # Check if we have an existing model to refine
#     model_path = "ns3_migration_v2_gpu.zip"
    
#     if os.path.exists(model_path):
#         print(f"--- Loading existing model {model_path} for Refinement ---")
#         # We load the old brain and link it to the new normalized environment
#         model = PPO.load(model_path, env=env, device=device)
#     else:
#         print("--- No existing model found. Starting fresh normalized training ---")
#         model = PPO("MlpPolicy", env, device=device, verbose=1, tensorboard_log=log_dir)

#     print("--- Starting Refinement (50,000 steps) ---")
#     try:
#         model.learn(
#             total_timesteps=50000, 
#             tb_log_name="refinement_v3_normalized",
#             reset_num_timesteps=False
#         )
#     except KeyboardInterrupt:
#         print("\nSaving progress...")

#     model.save("ns3_migration_v3_normalized")
#     print("--- Refinement Complete. Model saved as 'ns3_migration_v3_normalized' ---")