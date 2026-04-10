AI-Driven Disaster Recovery (DR) Orchestrator for LTE/5G Networks
🌐 Project Overview
This project simulates a critical Disaster Recovery (DR) scenario for LTE/5G infrastructure. When a primary network core fails, thousands of LTE RBS (Radio Base Station) nodes must migrate to a Disaster Recovery Core.

The challenge is the "Thundering Herd" effect: if 13,838 nodes migrate simultaneously, the surge of signaling crashes the DR Core. This project implements a Digital Twin (DT) using Reinforcement Learning to intelligently control the migration speed, ensuring maximum throughput without overloading the system.

🛠️ Key Technical Features
Massive Scalability: Verified with a dataset of 13,838 nodes, each handling associated UE signaling.
Zero-Copy IPC: High-performance POSIX Shared Memory (/dev/shm) bridge between C++ (NS-3) and Python (AI).
Performance: Achieved an 8.7x reduction in recovery time compared to manual fixed-speed scripts.
Containerized Architecture: Hosted via Docker on WSL for modular deployment.
🏗️ System Architecture
The system operates as a closed-loop Digital Twin:

NS-3 Environment (C++): Models the DR Core queue depth and signaling latency.
AI Orchestrator (Python): A DRL agent that "senses" queue health and issues optimal batch-size commands.
The SHM Bridge: A memory segment allowing nanosecond-speed data exchange between processes.
📊 Performance Comparison (13,838 Nodes)
Metric	Fixed-Rate Script (Manual)	AI Digital Twin (rApp)
Migration Time	452.21 seconds	51.62 seconds
Core Stability	Critical Overflows	Proactively Stabilized
Signaling Drops	~6.1 Million	~0.9 Million (84% Reduction)
🚀 Demo Procedure
Follow these steps to run the simulation within the Docker/WSL environment.

1. Initialize the Container
Ensure your WSL terminal is running the Docker container. Open two separate terminal tabs.

2. Terminal 1: The DR Core (NS-3)
Enter the container and launch the environment first to initialize the Shared Memory.

docker exec -it <container_id> bash
cd /opt/ns-3
./ns3 run dr_sim_compare
Wait until the log shows: [SUCCESS] Loaded 13838 nodes. Waiting for AI agent...

3. Terminal 2: The Intelligence (Python)
In the second terminal tab, enter the same container and launch the AI Orchestrator.

Bash
docker exec -it <container_id> bash
cd /app
python3 compare_results.py
4. Observe and Verify
Real-Time Adaptation: Watch the C++ logs. Note how the AI reduces batch sizes as queue_depth approaches the 5,000 capacity limit.

Results: Once finished, the script saves a visual dashboard: scalability_report_v2.png.

📂 Repository Structure
scratch/dr_sim_compare.cc: NS-3 simulation logic.

app/compare_results.py: DRL orchestrator and visualization logic.

shm_types.h: Shared C++/Python data structure definitions.

node_max.csv: RBS mobility dataset (13,838 nodes).


### Pro-Tips for GitHub:
1.  **The Header Image:** Upload your `scalability_report_v2.png` to the repository, then add this line at the very top of your README to make it look great:
    `![Performance Dashboard](scalability_report_v2.png)`
2.  **Raw Mode:** If you ever need to see how other people do their READMEs, click the **"Raw"** button on any GitHub file to see the unformatted text.
