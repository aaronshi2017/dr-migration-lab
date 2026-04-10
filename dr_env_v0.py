import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from shm_types import DrSharedMemory  # <--- Crucial: Reference the memory structure

class DrMigrationEnv(gym.Env):
    def __init__(self, shm_obj):
        super(DrMigrationEnv, self).__init__()
        self.shm = shm_obj
        
        # 1. Action: Batch size (1 to 50 nodes at once)
        self.action_space = spaces.Discrete(128) 
        
        # 2. Observation: [Current Queue, Avg RRC of next batch, Remaining Nodes]
        # We increase the 'high' range to accommodate 15,000+ nodes

        self.observation_space = spaces.Box(
            low=0, high=1e9, shape=(4,), dtype=np.float32
        )
        self.cursor = 0

    def _get_obs(self):
            current_node_count = self.shm.node_count
            
            # 1. Normalize Queue Depth (0 to 1.0)
            # Based on your MME Capacity of 24,000
            normalized_q = float(self.shm.core.queue_depth) / 24000.0
            
            # 2. Normalize Next Batch Weight (0 to 1.0)
            # Assuming max RRC_limit per node is around 250 (adjust if higher)
            raw_weight = np.mean([
                self.shm.nodes[i].rrc_limit for i in range(self.cursor, min(self.cursor + 10, current_node_count))
            ]) if self.cursor < current_node_count else 0
            normalized_weight = float(raw_weight) / 250.0 
            
            # 3. Normalize Progress (Remaining Nodes) (0 to 1.0)
            remaining_nodes = current_node_count - self.cursor
            normalized_remaining = float(remaining_nodes) / 13828.0
            
            # 4. Normalize Success Count (0 to 1.0)
            normalized_success = float(self.shm.core.total_finished) / 13828.0

            # RETURN 4 VALUES: Perfectly scaled for the PPO "Brain"
            return np.array([
                normalized_q, 
                normalized_weight, 
                normalized_remaining,
                normalized_success
            ], dtype=np.float32)

    def step(self, action):
        batch_size = int(action) + 1
        start = self.cursor
        # Dynamically calculate end point based on SHM node_count
        current_node_count = self.shm.node_count
        end = min(start + batch_size - 1, current_node_count - 1)
        
        # Snapshot drops for reward calculation
        initial_drops = self.shm.core.dropped_reqs
        
        # Send Command to C++
        self.shm.cmd_start_id = start
        self.shm.cmd_end_id = end
        self.shm.trigger = 1 
        
        # Wait for C++ to finish processing the batch
        # If your C++ script clears 'trigger' to 0 when done, use a while loop here instead
        # Inside step(self, action) in dr_env.py
        # Wait for C++ to finish (Handshake)
        start_wait = time.time()
        while self.shm.trigger != 0:
            time.sleep(0.001) # Tiny sleep to save CPU
            if time.time() - start_wait > 5.0: # 5 second timeout
                print("TIMEOUT: C++ is not responding!")
                self.shm.trigger = 0 # Force reset to prevent hang
                break

        self.cursor = end + 1
        
        # Reward Logic
        new_drops = self.shm.core.dropped_reqs - initial_drops
        reward = (batch_size * 2) - (new_drops * 10)
        
        if self.shm.core.queue_depth > 22000:
            reward -= 20 
            
        done = self.cursor >= current_node_count
        truncated = False
        
        return self._get_obs(), float(reward), done, truncated, {}

    def reset(self, seed=None, options=None):
        # Handle Gymnasium's newer seed/options requirements
        super().reset(seed=seed)
        
        # Reset the logical cursor
        self.cursor = 0
        
        # Return observation AND an empty info dictionary (Unpacking fix)
        return self._get_obs(), {}