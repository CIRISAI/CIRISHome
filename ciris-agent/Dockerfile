# CIRIS Agent Home Assistant Addon Dockerfile
#
# This Dockerfile builds a Home Assistant addon that runs the CIRIS Agent.
# The base image (from build.yaml) includes S6-overlay and bashio.
#
# Key requirements for SUPERVISOR_TOKEN to work:
# 1. config.yaml must have hassio_api: true
# 2. run.sh must use shebang: #!/usr/bin/with-contenv bashio
# 3. Wheel must be pure Python (py3-none-any), not manylinux

ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install system dependencies
# hadolint ignore=DL3018
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy and install CIRIS Agent wheel
# NOTE: The wheel MUST be py3-none-any (pure Python)
# Alpine uses musllinux, so manylinux wheels will NOT work
COPY ciris_agent-*.whl /tmp/
# hadolint ignore=DL3013
RUN pip3 install --no-cache-dir --break-system-packages /tmp/ciris_agent-*.whl && \
    rm /tmp/ciris_agent-*.whl

# Copy web assets to gui_static where the agent expects them
# The agent looks for gui_static in the ciris_engine package directory
COPY www/ /tmp/www/
RUN CIRIS_PKG=$(python3 -c "import ciris_engine; print(ciris_engine.__path__[0])") && \
    mkdir -p "$CIRIS_PKG/gui_static" && \
    cp -r /tmp/www/* "$CIRIS_PKG/gui_static/" && \
    rm -rf /tmp/www && \
    echo "GUI static assets installed to: $CIRIS_PKG/gui_static"

# Environment
ENV PYTHONUNBUFFERED=1

# Copy run script
# IMPORTANT: run.sh uses #!/usr/bin/with-contenv bashio to get SUPERVISOR_TOKEN
COPY run.sh /run.sh
RUN chmod +x /run.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8099/v1/system/health || exit 1

EXPOSE 8099

# The run.sh script launches ciris-agent
CMD ["/run.sh"]
