FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.01-py3-alps3
ENV MAX_JOBS=8

# setup.py installs flash_attn_interface top-level; TE 2.11 imports it as
# flash_attn_3.flash_attn_interface, so we copy it into that namespace.
ARG FA3_COMMIT=59df2f9bdf029f8e837ca79e1c0fe8c5d32dea25
RUN git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && \
    cd /tmp/flash-attention && git checkout ${FA3_COMMIT} && \
    git rev-parse HEAD && \
    cd hopper && python setup.py install && \
    py=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    mkdir -p "$py/flash_attn_3" && \
    cp flash_attn_interface.py "$py/flash_attn_3/flash_attn_interface.py" && \
    cd / && rm -rf /tmp/flash-attention

# Verify the exact symbols TE 2.11 imports
RUN python -c "from flash_attn_3.flash_attn_interface import flash_attn_func; print('FA3 OK')"