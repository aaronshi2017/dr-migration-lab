# This file contains a test function that runs a fixed batch size test on the DrMigrationEnv environment.
import time
from tqdm import tqdm
from dr_env import DrMigrationEnv

def run_fixed_test(shm_obj, batch_size=32, node_limit=13828):
    # 1. Initialize Environment
    env = DrMigrationEnv(shm_obj)
    
    # 2. Reset
    obs, info = env.reset()
    terminated = False
    
    # Stats
    start_wall_time = time.time()
    max_q_observed = 0
    last_cursor = 0

    # 3. Progress Bar
    # FIX: leave=True ensures the bar stays on screen when finished
    # Change this line:
    pbar = tqdm(total=node_limit, desc=f"Fixed Script ({batch_size})", unit="node", leave=True)

    try:
        fixed_interval = 2.0  # Hardcoded "Dumb" interval
        # Force clear the stats so the AI starts with a 'clean' network observation
        shm_obj.core.dropped_reqs = 0
        shm_obj.core.queue_depth = 0
        # Tell C++ to reset its internal state if necessary
        shm_obj.trigger = 0
    
        while not terminated and env.cursor < node_limit:
            action = max(0, batch_size - 1)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Blindly wait 2 seconds before the next batch
            time.sleep(fixed_interval) 
            
            pbar.update(env.cursor - last_cursor)
            last_cursor = env.cursor

            # --- ADD THIS DEBUG PRINT ---
            if env.cursor % 500 == 0: # Print every 100 nodes to avoid flooding
                print(f"[DEBUG] Cursor: {env.cursor} | Finished: {shm_obj.core.total_finished} | Drops: {shm_obj.core.dropped_reqs}")
            
             # Monitor Queue Depth            
            if shm_obj.core.queue_depth > max_q_observed:
                max_q_observed = shm_obj.core.queue_depth

        pbar.close()

        # --- THE "DRAIN" PERIOD ---
        print("\n[INFO] Waiting for simulation to drain queue...")
        for _ in range(5): # Wait 5 seconds
            time.sleep(1)
            print(f"  ... Still processing: {shm_obj.core.total_finished}/{node_limit} finished")
            if shm_obj.core.total_finished >= node_limit:
                break

        total_time = time.time() - start_wall_time
        actual_finished = shm_obj.core.total_finished
        success_percent = (actual_finished / node_limit) * 100

        return {
            "success_rate": min(success_percent, 100.0),
            "time": total_time,
            "drops": shm_obj.core.dropped_reqs,
            "max_q": max_q_observed,
            "total_nodes": node_limit
        }

    except Exception as e:
        pbar.close()
        print(f"Error during Fixed test: {e}")
        return None