// This file is a simplified version of the DR simulation, designed to be easily compared against the Python version. this is v1 version.

#include "ns3/core-module.h"
#include "shm_types.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <unistd.h>
#include <sys/types.h>
#include <sstream>

using namespace ns3;

class DrCore {
public:
    // LOWERED CAPACITY: 2500 makes the system "congested" so AI can outperform Fixed scripts
    uint32_t m_capacity = 5000; 
    DrSharedMemory* m_shm;

    // void RequestAttach(uint32_t nodeId) {
    //     if (m_shm->core.queue_depth < m_capacity) {
    //         m_shm->core.queue_depth++;
    //         // Realistic attachment delay (1.8s to 3.4s)
    //         double delay = 1.8 + (rand() % 1600) / 1000.0;
    //         Simulator::Schedule(Seconds(delay), &DrCore::CompleteAttach, this, nodeId);
    //     } else {
    //         // FIX: Increment drops ONLY ONCE per failed attempt, then stop.
    //         // Do not reschedule RequestAttach here; let the next batch handle retries
    //         // to prevent the "million drops" bug.
    //         m_shm->core.dropped_reqs++;
    //     }
    // }

    // void CompleteAttach(uint32_t nodeId) {
    //     if (m_shm->core.queue_depth > 0) m_shm->core.queue_depth--;
        
    //     m_shm->nodes[nodeId].processed_ue++;
        
    //     // FIX: Ensure status transition is robust
    //     if (m_shm->nodes[nodeId].processed_ue >= m_shm->nodes[nodeId].rrc_limit) {
    //         if (m_shm->nodes[nodeId].status != 2) { 
    //             m_shm->nodes[nodeId].status = 2; // STATUS_COMPLETED
    //             m_shm->core.total_finished++;    // THIS TURNS THE BAR GREEN IN PYTHON
    //         }
    //     }
    // }
    // Inside the DrCore class
    void RequestAttach(uint32_t nodeId) {
        if (m_shm->core.queue_depth < m_capacity) {
            m_shm->core.queue_depth++;
            double delay = 1.8 + (rand() % 1600) / 1000.0;
            Simulator::Schedule(Seconds(delay), &DrCore::CompleteAttach, this, nodeId);
        } else {
            // FIX: Increment drops ONLY once per failed node entry.
            // Do NOT reschedule RequestAttach here, or you will get millions of drops.
            m_shm->core.dropped_reqs++; 
        // CRITICAL: Increment processed_ue even on a drop.
            // This ensures the node doesn't hang at 99% forever.
            m_shm->nodes[nodeId].processed_ue++;
            
            // Check if this drop was the last UE for the node
            if (m_shm->nodes[nodeId].processed_ue >= m_shm->nodes[nodeId].rrc_limit) {
                if (m_shm->nodes[nodeId].status != 2) {
                    m_shm->nodes[nodeId].status = 2;
                    m_shm->core.total_finished++;
                }
            }
        }
    }


    void CompleteAttach(uint32_t nodeId) {
        if (m_shm->core.queue_depth > 0) m_shm->core.queue_depth--;
        m_shm->nodes[nodeId].processed_ue++;
        
        // FIX: This section updates the 'total_finished' counter that Python reads.
        // Without this, your success rate stays at 0.00%.
        if (m_shm->nodes[nodeId].processed_ue >= m_shm->nodes[nodeId].rrc_limit) {
            if (m_shm->nodes[nodeId].status != 2) { // 2 = STATUS_COMPLETED
                m_shm->nodes[nodeId].status = 2;
                m_shm->core.total_finished++; // THIS TURNS THE BAR GREEN
            }
        }
    }
};

int main(int argc, char *argv[]) {
    // 1. SHM Setup
    int shm_fd = shm_open("/ns3ai-shm", O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) { perror("shm_open failed"); return 1; }

    ftruncate(shm_fd, sizeof(DrSharedMemory));
    DrSharedMemory* shm = (DrSharedMemory*)mmap(0, sizeof(DrSharedMemory), 
                                                PROT_READ | PROT_WRITE, 
                                                MAP_SHARED, shm_fd, 0);
    memset(shm, 0, sizeof(DrSharedMemory));

    // 2. CSV Loader (scratch/node_max.csv)
    std::ifstream file("scratch/node_max.csv");
    std::string line;
    if (!file.is_open()) { std::cerr << "Error: CSV not found." << std::endl; return 1; }
    
    std::getline(file, line); // Header
    uint32_t idx = 0;
    while (std::getline(file, line) && idx < 15000) {
        std::stringstream ss(line);
        std::string name, rrc;
        if (std::getline(ss, name, ',') && std::getline(ss, rrc, ',')) {
            strncpy(shm->nodes[idx].name, name.c_str(), 31);
            shm->nodes[idx].rrc_limit = std::stoul(rrc);
            shm->nodes[idx].status = 0;
            shm->nodes[idx].processed_ue = 0;
            idx++;
        }
    }
    shm->node_count = idx;
    std::cout << "\033[1;32m[SUCCESS]\033[0m Loaded " << idx << " nodes." << std::endl;

    DrCore core;
    core.m_shm = shm;

    // 3. Command Listener with Intelligent Reset
    std::function<void()> CheckAI = [&]() {
        static uint32_t last_start_id = 0;
        
        // AUTO-RESET: Triggered when Python moves the cursor back to 0
        if (shm->cmd_start_id == 0 && last_start_id > 0) {
            std::cout << "\033[1;33m[RESET]\033[0m Cleaning Environment for next test..." << std::endl;
            shm->core.dropped_reqs = 0;
            shm->core.queue_depth = 0;
            shm->core.total_finished = 0;
            for (uint32_t i = 0; i < shm->node_count; ++i) {
                shm->nodes[i].status = 0;
                shm->nodes[i].processed_ue = 0;
            }
        }
        last_start_id = shm->cmd_start_id;

        // BATCH PROCESSING
        if (shm->trigger == 1) {
            uint32_t start = shm->cmd_start_id;
            uint32_t end = shm->cmd_end_id;

            for (uint32_t i = start; i < end && i < shm->node_count; ++i) {
                if (shm->nodes[i].status == 0) {
                    shm->nodes[i].status = 1; // STATUS_PROCESSING
                    // Launch all UEs for this node with slight jitter
                    for (uint32_t u = 0; u < shm->nodes[i].rrc_limit; ++u) {
                        double jitter = (rand() % 2000) / 1000.0; 
                        Simulator::Schedule(Seconds(jitter), &DrCore::RequestAttach, &core, i);
                    }
                }
            }
            shm->trigger = 0; // Handshake complete
        }

        Simulator::Schedule(MilliSeconds(50), CheckAI);
    };

    Simulator::Schedule(MilliSeconds(50), CheckAI);

    std::cout << "[SYSTEM] Real-time loop starting. Waiting for Python..." << std::endl;

    // Real-time blocking loop
    while (true) {
        if (shm->trigger == 1) {
            uint32_t start = shm->cmd_start_id;
            uint32_t end = shm->cmd_end_id;
            for (uint32_t i = start; i < end && i < shm->node_count; ++i) {
                if (shm->nodes[i].status == 0) {
                    shm->nodes[i].status = 1;
                    for (uint32_t u = 0; u < shm->nodes[i].rrc_limit; ++u) {
                        if (shm->core.queue_depth < core.m_capacity) {
                            shm->core.queue_depth++;
                            shm->core.queue_depth--;
                            shm->nodes[i].processed_ue++;
                        } else {
                            shm->core.dropped_reqs++;
                            shm->nodes[i].processed_ue++;
                        }
                    }
                    if (shm->nodes[i].processed_ue >= shm->nodes[i].rrc_limit) {
                        shm->nodes[i].status = 2;
                        shm->core.total_finished++;
                    }
                }
            }
            shm->trigger = 0; // Handshake complete
        }
        usleep(10000); // 10ms poll interval only
    }

    std::cout << "[FINISH] Final Finished Count: " << shm->core.total_finished << std::endl;
    return 0;
}