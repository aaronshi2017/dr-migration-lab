# V1 version This script compares the results of the AI orchestrator and the fixed baseline for both a 5,000 node test and a full dataset test. It reads the shared memory to get the dropped request counts and success rates, then generates a comprehensive dashboard with three rows: Success Rate, Completion Time, and Dropped Requests. Each row compares the AI rApp against the Fixed Script for both test scenarios. The final report is saved as 'scalability_report_v2.png'.
import os
import time
import pandas as pd
import ctypes
import mmap
import matplotlib.pyplot as plt
from shm_types import DrSharedMemory

# Import your test logic
from test_model import run_inference
from test_fixed import run_fixed_test

def get_shm_obj():
    shm_path = "/dev/shm/ns3ai-shm"
    if not os.path.exists(shm_path):
        print(f"\033[1;31m[ERROR]\033[0m {shm_path} not found!")
        return None
    fd = os.open(shm_path, os.O_RDWR)
    shm_size = ctypes.sizeof(DrSharedMemory)
    shm_mmap = mmap.mmap(fd, shm_size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE | mmap.PROT_READ)
    return DrSharedMemory.from_buffer(shm_mmap)

def reset_counters(shm_obj):
    """Resets SHM counters and ensures C++ queue is cleared."""
    shm_obj.core.dropped_reqs = 0
    shm_obj.core.total_finished = 0
    shm_obj.core.queue_depth = 0  # CRITICAL: AI observes this!
    shm_obj.trigger = 0
    shm_obj.cmd_start_id = 0
    shm_obj.cmd_end_id = 0
    
    for i in range(shm_obj.node_count):
        shm_obj.nodes[i].processed_ue = 0
        shm_obj.nodes[i].status = 0
    print("[SYSTEM] Shared Memory and C++ Queue Reset Successfully.")
    
def main():
    shm_obj = get_shm_obj()
    if shm_obj is None: return

    try:
        df = pd.read_csv('node_max.csv')
        full_size = len(df)
    except:
        print("Error: node_max.csv not found.")
        return
    
    print(f"--- STARTING COMPARISON (Full Dataset: {full_size} nodes) ---")

    # --- TEST 1: 5,000 NODES ---
    print("\n[1/4] Running Fixed Baseline (5k)...")
    reset_counters(shm_obj)
    fixed_5k = run_fixed_test(shm_obj, batch_size=32, node_limit=5000)
    if fixed_5k is not None:
        fixed_5k['drops'] = float(shm_obj.core.dropped_reqs)
        fixed_5k['success_rate'] = (shm_obj.core.total_finished / 5000) * 100
    else:
        print("!! Fixed test failed to return results. Check test_fixed.py !!")
        fixed_5k = {'success_rate': 0, 'time': 0, 'drops': 0} # Fallback
    # # Grab the stats while C++ is still in this specific test state
    # fixed_5k['drops'] = float(shm_obj.core.dropped_reqs)
    # fixed_5k['success_rate'] = (shm_obj.core.total_finished / 5000) * 100

    print("\n[2/4] Running AI Orchestrator (5k)...")
    reset_counters(shm_obj)
    ai_5k = run_inference(shm_obj, node_limit=5000)
    ai_5k['drops'] = float(shm_obj.core.dropped_reqs)
    ai_5k['success_rate'] = (shm_obj.core.total_finished / 5000) * 100

    # --- TEST 2: FULL DATASET ---
    print(f"\n[3/4] Running Fixed Baseline (Full: {full_size})...")
    reset_counters(shm_obj)
    fixed_full = run_fixed_test(shm_obj, batch_size=64, node_limit=full_size)
    fixed_full['drops'] = shm_obj.core.dropped_reqs

    print(f"\n[4/4] Running AI Orchestrator (Full: {full_size})...")
    reset_counters(shm_obj)
    ai_full = run_inference(shm_obj, node_limit=full_size)
    ai_full['drops'] = shm_obj.core.dropped_reqs

    # --- GENERATE PLOTS ---
    generate_triple_dashboard(ai_5k, fixed_5k, ai_full, fixed_full, full_size)

def generate_triple_dashboard(ai_5k, fixed_5k, ai_full, fixed_full, full_size):
    # 3 Rows now: Success, Time, and Drops
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Ericsson rApp Analysis: Reliability & Scalability', fontsize=22, fontweight='bold')

    # ROW 1: SUCCESS RATE
    plot_bar(axes[0,0], "Success Rate % (5k)", [ai_5k['success_rate'], fixed_5k['success_rate']], "Percent")
    plot_bar(axes[0,1], f"Success Rate % (Full: {full_size})", [ai_full['success_rate'], fixed_full['success_rate']], "Percent")

    # ROW 2: COMPLETION TIME
    plot_bar(axes[1,0], "Completion Time (5k)", [ai_5k['time'], fixed_5k['time']], "Seconds")
    plot_bar(axes[1,1], f"Completion Time (Full: {full_size})", [ai_full['time'], fixed_full['time']], "Seconds")

    # ROW 3: DROP RATES
    plot_bar(axes[2,0], "Dropped Requests (5k)", [ai_5k['drops'], fixed_5k['drops']], "Count (Lower is Better)")
    plot_bar(axes[2,1], f"Dropped Requests (Full: {full_size})", [ai_full['drops'], fixed_full['drops']], "Count (Lower is Better)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('scalability_report_v2.png', dpi=300)
    print(f"\n[FINISH] Report saved as 'scalability_report_v2.png'")

def plot_bar(ax, title, values, ylabel):
    labels = ['AI rApp', 'Fixed Script']
    # Dynamic Colors: Success=Green/Red, Time=Blue/Orange, Drops=Orange/DarkRed
    if "Success" in title: colors = ['#2ecc71', '#e74c3c']
    elif "Dropped" in title: colors = ['#f39c12', '#c0392b']
    else: colors = ['#3498db', '#9b59b6']
    
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    
    limit = 110 if "Success" in title else max(values) * 1.4 if any(v > 0 for v in values) else 10
    ax.set_ylim(0, limit)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

if __name__ == "__main__":
    main()