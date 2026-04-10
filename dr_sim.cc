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


using namespace ns3;

class DrCore {
public:
    uint32_t m_capacity = 24000; // Total MME TPS capacity
    DrSharedMemory* m_shm;

    void RequestAttach(uint32_t nodeId) {
        if (m_shm->core.queue_depth < m_capacity) {
            m_shm->core.queue_depth++;
            // Measured delay 1.8s to 3.4s
            double delay = 1.8 + (rand() % 1600) / 1000.0;
            Simulator::Schedule(Seconds(delay), &DrCore::CompleteAttach, this, nodeId);
        } else {
            m_shm->core.dropped_reqs++;
            // T3411 Retry after 10s
            Simulator::Schedule(Seconds(10.0), &DrCore::RequestAttach, this, nodeId);
        }
    }

    void CompleteAttach(uint32_t nodeId) {
        m_shm->core.queue_depth--;
        m_shm->nodes[nodeId].processed_ue++;
        
        // If all UEs for this node finished, mark node as Finished (status 2)
        if (m_shm->nodes[nodeId].processed_ue >= m_shm->nodes[nodeId].rrc_limit) {
            if (m_shm->nodes[nodeId].status != 2) { 
                m_shm->nodes[nodeId].status = 2;
                m_shm->core.total_finished++; // <--- VERY IMPORTANT
            }
        }
    }
};

// ... (includes and DrCore class remain the same)

int main(int argc, char *argv[]) {
    // 1. SHM Setup
    int shm_fd = shm_open("ns3ai-shm", O_CREAT | O_RDWR, 0666);
    ftruncate(shm_fd, sizeof(DrSharedMemory));
    DrSharedMemory* shm = (DrSharedMemory*)mmap(0, sizeof(DrSharedMemory), PROT_WRITE, MAP_SHARED, shm_fd, 0);
    memset(shm, 0, sizeof(DrSharedMemory));

    // 2. Optimized CSV Loader (Only one loop!)
    std::ifstream file("scratch/node_max.csv");
    std::string line;
    if (!std::getline(file, line)) return 1; // Skip header

    uint32_t idx = 0;
    while (std::getline(file, line) && idx < 5000) {
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
    std::cout << "Successfully loaded " << idx << " nodes." << std::endl;

    DrCore core;
    core.m_shm = shm;

    // 3. Command Listener with AUTO-RESET
    std::function<void()> CheckAI = [&]() {
        static uint32_t last_start_id = 0;

        // --- DETECTION OF PYTHON RESET ---
        // If the Python cursor jumps back to 0, reset the C++ environment
        if (shm->cmd_start_id < last_start_id) {
            std::cout << "\033[1;33m[SYSTEM]\033[0m Python Reset Detected. Clearing Node Status..." << std::endl;
            for (uint32_t i = 0; i < shm->node_count; ++i) {
                shm->nodes[i].status = 0;
                shm->nodes[i].processed_ue = 0;
            }
            shm->core.dropped_reqs = 0;
            shm->core.queue_depth = 0;
            shm->core.total_finished = 0; // <--- Reset this too
        }
        last_start_id = shm->cmd_start_id;

        if (shm->trigger == 1) {
            uint32_t start = shm->cmd_start_id;
            uint32_t end = shm->cmd_end_id;

            // Safety: Ensure we don't go out of bounds
            if (start < shm->node_count) {
                for (uint32_t i = start; i < end && i < shm->node_count; ++i) {
                    if (shm->nodes[i].status != 0) continue; 
                    
                    shm->nodes[i].status = 1; 
                    // for (uint32_t u = 0; u < shm->nodes[i].rrc_limit; ++u) {
                    //     Simulator::ScheduleNow(&DrCore::RequestAttach, &core, i);
                    // }
                    for (uint32_t u = 0; u < shm->nodes[i].rrc_limit; ++u) {
                        // Spread the UEs out over 1.0 to 5.0 seconds 
                        // This is much faster than the 60s jitter from before, but safer than "Now"
                        double small_jitter = (rand() % 4000) / 1000.0 + 1.0; 
                        Simulator::Schedule(Seconds(small_jitter), &DrCore::RequestAttach, &core, i);
                    }
                }
            }
    
            // ALWAYS reset the trigger last
            shm->trigger = 0; 
            std::cout << "[AI Done] Batch " << start << " to " << end << " processed." << std::endl;
        }

        Simulator::Schedule(Seconds(4), CheckAI);
    };

    Simulator::Schedule(Seconds(4), CheckAI);
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}