#ifndef SHM_TYPES_H
#define SHM_TYPES_H

#include <stdint.h>

struct ShmNode {
    char name[32];          // Node Name from Excel
    uint32_t rrc_limit;     // Max UE from Excel
    uint8_t status;         // 0:Idle, 1:Migrating, 2:Finished
    uint32_t processed_ue;  // Counter to track progress per node
} __attribute__((packed));

struct ShmCore {
    uint32_t queue_depth;   // Active handshakes (1.8s-3.4s)
    uint32_t dropped_reqs;  // Overload counter
    uint32_t total_finished; // <--- NEW: Fast counter for AI
} __attribute__((packed));

struct DrSharedMemory {
    uint32_t node_count;    // Actual number of nodes loaded from CSV
    ShmCore core;
    
    // AI Control
    uint32_t cmd_start_id;  
    uint32_t cmd_end_id;    
    uint8_t trigger;        // 1: Start
    uint8_t wave_type;      // 0: Squared, 1: Staggered
    
    ShmNode nodes[15000];    // Buffer for up to 5000 entries
} __attribute__((packed));

#endif
