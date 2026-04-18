#!/usr/bin/with-contenv bashio
# CIRIS Agent startup script for Home Assistant Addon
#
# IMPORTANT: The shebang above (#!/usr/bin/with-contenv bashio) is REQUIRED
# for SUPERVISOR_TOKEN to be available. Do NOT use /command/with-contenv.
#
# The with-contenv wrapper ensures environment variables injected by the
# HA Supervisor (like SUPERVISOR_TOKEN) are available to child processes.

set -e

# Set CIRIS_HOME to addon data directory (persistent storage)
# /root is forbidden as a system directory, /data is the HA addon persistent storage
export CIRIS_HOME=/data/ciris

# Ensure directory exists
mkdir -p "$CIRIS_HOME"

# Enable sidebar panel via Supervisor API
# This ensures CIRIS appears in the HA sidebar automatically
if [ -n "$SUPERVISOR_TOKEN" ]; then
    curl -sf -X POST \
        -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"ingress_panel": true}' \
        http://supervisor/addons/self/options >/dev/null 2>&1 || true
fi

# Launch CIRIS Agent with API adapter
# The agent will auto-detect HA mode via SUPERVISOR_TOKEN
exec ciris-agent --adapter api --host 0.0.0.0 --port 8099
