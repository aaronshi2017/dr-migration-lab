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

# --- 1. MEMORY STRUCTURES (Matches shm_types.h exactly) ---

class ShmNode(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("name", ctypes.c_char * 32),
        ("rrc_limit", ctypes.c_uint32),
        ("status", ctypes.c_uint8),      # 0:Idle, 1:Migrating, 2:Finished
        ("processed_ue", ctypes.c_uint32)
    ]

class ShmCore(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("queue_depth", ctypes.c_uint32),
        ("dropped_reqs", ctypes.c_uint32),
        ("total_finished", ctypes.c_uint32) # Matches the new .h field
    ]

class DrSharedMemory(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("node_count", ctypes.c_uint32),
        ("core", ShmCore),
        ("cmd_start_id", ctypes.c_uint32),
        ("cmd_end_id", ctypes.c_uint32),
        ("trigger", ctypes.c_uint8),     # Match uint8_t in C++
        ("wave_type", ctypes.c_uint8),   # Match uint8_t in C++
        ("nodes", ShmNode * 5000)        # Fixed buffer of 5000
    ]

SHM_PATH = "/dev/shm/ns3ai-shm"

# --- 2. REINFORCEMENT LEARNING ENVIRONMENT ---

class NS3MigrationEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.node_count = 5000
        self.cursor = 0
        # Action: Batch size (0-127) -> 1 to 128 nodes
        self.action_space = gym.spaces.Discrete(32)

      
        print(f"Waiting for {SHM_PATH}...")
        while not os.path.exists(SHM_PATH):
            time.sleep(1)
        
        # Open and map the shared memory
        self.f = open(SHM_PATH, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), ctypes.sizeof(DrSharedMemory))
        self.shm = DrSharedMemory.from_buffer(self.mm)

        # Observation: [Cursor, Queue Depth, Dropped Reqs]
        self.observation_space = gym.spaces.Box(
            low=0, high=1e9, shape=(4,), dtype=np.float32
        )

    def _get_obs(self):
        # Access nested core struct variables
        return np.array([
            float(self.cursor),
            float(self.shm.core.queue_depth),
            float(self.shm.core.dropped_reqs),
            float(self.shm.core.total_finished) # Fast access!
        ], dtype=np.float32)


    def step(self, action):
            # 1. Action Mapping (Discrete 0-127 -> 1 to 128 nodes)
            batch_size = int(action) + 1
            start = self.cursor
            end = min(start + batch_size, self.node_count)

            # 2. Snapshot state before the batch
            initial_drops = self.shm.core.dropped_reqs
            initial_finished = self.shm.core.total_finished # NEW: Use the global counter
            initial_q = self.shm.core.queue_depth

            # 3. Trigger C++ Migration
            self.shm.cmd_start_id = start
            self.shm.cmd_end_id = end
            self.shm.trigger = 1 

            # 4. Optimized Handshake with Timeout
            max_wait = 2.0  # Reduced to 2s for tighter training
            wait_start = time.time()
            while self.shm.trigger != 0:
                # Busy-wait with a tiny sleep to save CPU while staying fast
                time.sleep(0.0001) 
                if time.time() - wait_start > max_wait:
                    print(f"!!! HANG DETECTED at node {start}. Resetting trigger.")
                    self.shm.trigger = 0
                    break
            
            # Update logical cursor
            self.cursor = end
            
            # 5. Calculate Delta Results
            new_drops = self.shm.core.dropped_reqs - initial_drops
            newly_finished = self.shm.core.total_finished - initial_finished
            current_q = self.shm.core.queue_depth

            # 6. REWARD DESIGN (Split 50/50 Start vs. Completion)
            # Bonus for successfully starting migrations (Progress)
            reward = (batch_size / self.node_count) * 50.0 
            
            # Bonus for nodes actually FINISHING (Handles the 1.8s delay)
            # This teaches the AI that migrations take time to resolve.
            reward += (newly_finished / self.node_count) * 50.0
            
            # Penalty for drops (Log-scaled so one bad batch doesn't break the brain)
            if new_drops > 0:
                reward -= np.log1p(new_drops) * 5.0
                
            # Penalty for "Dangerous" Queue Levels
            # (Capacity is 24,000; penalizing starts at 20,000)
            if current_q > 20000:
                reward -= 2.0

            # 7. Check for Episode End
            terminated = bool(self.cursor >= self.node_count)
            truncated = False
            
            return self._get_obs(), float(reward), terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor = 0
        # Wait for C++ to be ready if necessary
        return self._get_obs(), {}

# --- 3. RUN TRAINING ---

# if __name__ == "__main__":
#     # Ensure CUDA is used
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"--- System: {device} ---")

#     # Wrap environment
#     env = NS3MigrationEnv()
#     env = Monitor(env)
#     env = DummyVecEnv([lambda: env])

#     # PPO Setup
#     model = PPO(
#         "MlpPolicy", 
#         env, 
#         device=device, 
#         verbose=1,
#         learning_rate=3e-4,
#         batch_size=64,
#         n_steps=2048
#     )

#     print("--- Starting Training ---")
#     model.learn(total_timesteps=150000)
#     model.save("ns3_migration_v2_gpu")
#     print("Training Complete. Model saved.")

# --- 3. RUN TRAINING ---

# --- 3. RUN TRAINING ---

if __name__ == "__main__":
    # 1. GPU/CUDA Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- System: {device} ---")

    # 2. TensorBoard Directory Setup
    # This creates a folder to store the diagram data
    log_dir = "./ppo_ns3_logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 3. Environment Initialization
    # We wrap it in Monitor to track rewards/lengths for the diagrams
    env = NS3MigrationEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # 4. PPO Model Configuration
    # 'tensorboard_log' tells PPO where to send the live data
    model = PPO(
        "MlpPolicy", 
        env, 
        device=device, 
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        tensorboard_log=log_dir  # Link to our diagram folder
    )

    print("--- Starting Training ---")
    
    # 5. The Learning Loop
    # 'tb_log_name' identifies this specific run in the browser
    try:
        model.learn(
            total_timesteps=150000, 
            tb_log_name="migration_v2_optimized",
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current progress...")

    # 6. Save the trained brain
    model.save("ns3_migration_v2_gpu")
    print("--- Training Complete. Model saved as 'ns3_migration_v2_gpu' ---")