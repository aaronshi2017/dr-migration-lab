FROM dr-lab-final:v1

# Ensure permissions and paths are correct for the new folder
WORKDIR /opt/ns-3

# We force the removal of old links and create fresh ones pointing to the /app volume
RUN rm -f scratch/dr_sim_compare.cc scratch/shm_types.h && \
    ln -s /app/dr_sim_compare.cc scratch/dr_sim_compare.cc && \
    ln -s /app/shm_types.h scratch/shm_types.h

# Copy the data file directly into scratch so ns-3 finds it immediately
# Note: Ensure node_max.csv is in your dr-migration-labv1 folder
COPY node_max.csv /opt/ns-3/scratch/node_max.csv

# Re-configure to ensure the build system registers the new scratch links
RUN ./ns3 configure --disable-examples --disable-tests

WORKDIR /app
