import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from shm_types import DrSharedMemory

class DrMigrationEnv(gym.Env):
    def __init__(self, shm_obj):
        super(DrMigrationEnv, self).__init__()
        self.shm = shm_obj
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Box(low=0, high=1e9, shape=(4,), dtype=np.float32)
        self.cursor = 0

    def _get_obs(self):
        current_node_count = self.shm.node_count
        normalized_q = float(self.shm.core.queue_depth) / 24000.0
        raw_weight = np.mean([
            self.shm.nodes[i].rrc_limit for i in range(self.cursor, min(self.cursor + 10, current_node_count))
        ]) if self.cursor < current_node_count else 0
        normalized_weight = float(raw_weight) / 250.0
        remaining_nodes = current_node_count - self.cursor
        normalized_remaining = float(remaining_nodes) / 13828.0
        normalized_success = float(self.shm.core.total_finished) / 13828.0
        return np.array([normalized_q, normalized_weight, normalized_remaining, normalized_success], dtype=np.float32)

    def step(self, action):
        batch_size = int(action) + 1
        start = self.cursor
        current_node_count = self.shm.node_count
        end = min(start + batch_size - 1, current_node_count - 1)
        initial_drops = self.shm.core.dropped_reqs

        # Send command to C++
        self.shm.cmd_start_id = start
        self.shm.cmd_end_id = end
        self.shm.trigger = 1

        # RESTORED HANDSHAKE: Wait for C++ to acknowledge (set trigger back to 0)
        start_wait = time.time()
        while self.shm.trigger != 0:
            time.sleep(0.005)
            if time.time() - start_wait > 10.0:
                print(f'TIMEOUT: C++ not responding at cursor {self.cursor}!')
                self.shm.trigger = 0
                break

        new_drops = self.shm.core.dropped_reqs - initial_drops
        q_depth_ratio = float(self.shm.core.queue_depth) / 24000.0
        throughput_reward = (batch_size / 64.0) * 2.0
        success_reward = max(0, (batch_size - new_drops)) * 0.5
        queue_penalty = (q_depth_ratio ** 2) * 50.0 if q_depth_ratio > 0.7 else 0
        drop_penalty = new_drops * 25.0
        reward = throughput_reward + success_reward - queue_penalty - drop_penalty

        self.cursor = end + 1
        done = self.cursor >= current_node_count
        return self._get_obs(), float(reward), done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cursor = 0
        return self._get_obs(), {}
