import ctypes

class ShmNode(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("name", ctypes.c_char * 32),
                ("rrc_limit", ctypes.c_uint32),
                ("status", ctypes.c_uint8),
                ("processed_ue", ctypes.c_uint32)]

class ShmCore(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("queue_depth", ctypes.c_uint32),
                ("dropped_reqs", ctypes.c_uint32),
                ("total_finished", ctypes.c_uint32)]

class DrSharedMemory(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("node_count", ctypes.c_uint32),
                ("core", ShmCore),
                ("cmd_start_id", ctypes.c_uint32),
                ("cmd_end_id", ctypes.c_uint32),
                ("trigger", ctypes.c_uint8),
                ("wave_type", ctypes.c_uint8),
                ("nodes", ShmNode * 15000)]