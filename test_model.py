# This file contains a test function that runs an inference test using a trained PPO model on the DrMigrationEnv environment.
import time
import torch
from tqdm import tqdm
from stable_baselines3 import PPO
from dr_env import DrMigrationEnv

def run_inference(shm_obj, node_limit=13828):
    # 1. Initialize Environment with the shared memory object
    env = DrMigrationEnv(shm_obj)
    
    # 2. Load the Trained Model
    # Ensure this file exists in your /app directory
    model = PPO.load("ns3_migration_v3_normalized")

    # 3. Reset (Fixes the 'unpack NoneType' error)
    obs, info = env.reset()
    terminated = False
    
    # Stats for the final report
    start_wall_time = time.time()
    max_q_observed = 0
    last_cursor = 0

    # 4. Progress Bar
    # FIX: leave=True ensures the bar stays on screen when finished
    pbar = tqdm(total=node_limit, desc="AI Orchestrator", unit="node", leave=True)

    try:
            # Reset counters before loop
            shm_obj.core.dropped_reqs = 0
            shm_obj.core.queue_depth = 0
            shm_obj.trigger = 0

            while not terminated and env.cursor < node_limit:
                action, _states = model.predict(obs, deterministic=True)
                
                if env.cursor % 100 == 0:
                    batch_size = action + 1
                    # obs[0] is key: if this is > 1.0, your normalization is wrong!
                    print(f"[AI DECISION] Node: {env.cursor} | Batch: {batch_size} | Queue Obs: {obs[0]:.4f}")
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                time.sleep(0.01) 

                # Correctly update progress bar based on cursor movement
                pbar.update(env.cursor - last_cursor)
                last_cursor = env.cursor

                if shm_obj.core.queue_depth > max_q_observed:
                    max_q_observed = shm_obj.core.queue_depth

            # --- MOVED OUTSIDE THE WHILE LOOP ---
            pbar.close()

            print("\n[INFO] Waiting for simulation to drain queue...")
            # Increased to 10 seconds to ensure high RRC_limit nodes can finish
            for i in range(20): 
                time.sleep(1)
                actual_finished = shm_obj.core.total_finished
                print(f"  ... Day {i+1}/20 | Finished: {actual_finished}/{node_limit}")
                if actual_finished >= node_limit:
                    break

            total_time = time.time() - start_wall_time
            success_percent = (shm_obj.core.total_finished / node_limit) * 100

            return {
                "success_rate": min(success_percent, 100.0),
                "time": total_time,
                "drops": shm_obj.core.dropped_reqs,
                "max_q": max_q_observed,
                "total_nodes": node_limit
            }

    except Exception as e:
        pbar.close()
        print(f"Error during AI test: {e}")
        return None