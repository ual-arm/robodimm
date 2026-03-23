FROM mambaorg/micromamba:1.5-jammy

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app

# Copy application files
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app

# Ensure directories exist for volumes
RUN mkdir -p /app/meshes /app/saved_configs /app/saved_libraries /app/saved_programs /app/station

# Expose port
EXPOSE 8000

# Activate environment and run with uvicorn
# Note: "micromamba run -n base" ensures the environment is active
CMD ["micromamba", "run", "-n", "base", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
