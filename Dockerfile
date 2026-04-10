# Use the NVIDIA TensorFlow base
FROM nvcr.io/nvidia/tensorflow:24.01-tf2-py3

# 1. Install system dependencies
RUN apt-get update && apt-get install -y \
    g++ python3 python3-dev cmake ninja-build git \
    libboost-all-dev libsqlite3-dev libxml2-dev \
    libgtk-3-dev pybind11-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade core python tools
RUN pip install --upgrade pip setuptools wheel

# 3. Install required Python libraries
# Note: we use a standard install for gymnasium and cppyy
RUN pip install jupyterlab gymnasium "cppyy>=3.1.2"

# 4. Clone and Build ns-3
WORKDIR /opt
RUN git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3

WORKDIR /opt/ns-3
RUN git clone https://github.com/hust-diangroup/ns3-ai.git contrib/ai

# 5. Configure and Build ns-3 with Python bindings
RUN ./ns3 configure --disable-examples --disable-tests \
    -- -DPython3_EXECUTABLE=$(which python3)
RUN ./ns3 build

# 6. Fix ns3-ai Python Module
# We set the PYTHONPATH so Python can find the ns3ai_utils.py file directly
ENV PYTHONPATH="/opt/ns-3/contrib/ai/python_utils:${PYTHONPATH}"

WORKDIR /opt/ns-3/contrib/ai/python_utils
# Perform a standard install (no -e flag) to ensure metadata is created
RUN pip install .

# 7. Final Workspace Setup
WORKDIR /app
EXPOSE 8888

CMD ["bash"]
