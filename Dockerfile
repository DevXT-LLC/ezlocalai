ARG LLAMACPP_IMAGE="full-cuda"
ARG BASE_IMAGE="ghcr.io/ggerganov/llama.cpp:${LLAMACPP_IMAGE}"
FROM ${BASE_IMAGE}

# Set the working directory
WORKDIR /app

# Copy the entrypoint script to the container
COPY entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script as the entrypoint for the container
ENTRYPOINT ["/app/entrypoint.sh"]
