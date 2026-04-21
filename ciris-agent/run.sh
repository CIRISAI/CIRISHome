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

# Track if HA needs restart (set by component install)
NEEDS_HA_RESTART=false

# Auto-install CIRIS conversation agent custom component
# This eliminates the need for manual HACS or copy steps
if [ -d "/app/custom_components/ciris" ]; then
    # Get version from manifest.json
    BUNDLED_VERSION=$(grep -o '"version": *"[^"]*"' /app/custom_components/ciris/manifest.json 2>/dev/null | cut -d'"' -f4 || echo "unknown")
    INSTALLED_VERSION=""
    if [ -f /config/custom_components/ciris/manifest.json ]; then
        INSTALLED_VERSION=$(grep -o '"version": *"[^"]*"' /config/custom_components/ciris/manifest.json | cut -d'"' -f4)
    fi

    if [ "$BUNDLED_VERSION" != "$INSTALLED_VERSION" ]; then
        bashio::log.info "Installing CIRIS conversation agent v${BUNDLED_VERSION}..."

        # Create custom_components directory if needed
        mkdir -p /config/custom_components

        # Remove old installation to ensure clean update
        rm -rf /config/custom_components/ciris

        # Copy bundled component
        cp -r /app/custom_components/ciris /config/custom_components/

        bashio::log.info "CIRIS conversation agent v${BUNDLED_VERSION} installed."
        bashio::log.warning "Please restart Home Assistant to enable the CIRIS integration."
        NEEDS_HA_RESTART=true
    else
        bashio::log.info "CIRIS conversation agent v${BUNDLED_VERSION} already installed."
    fi
else
    bashio::log.warning "Custom component not bundled in addon image."
fi

# Enable sidebar panel via Supervisor API
# This ensures CIRIS appears in the HA sidebar automatically
if [ -n "$SUPERVISOR_TOKEN" ]; then
    curl -sf -X POST \
        -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"ingress_panel": true}' \
        http://supervisor/addons/self/options >/dev/null 2>&1 || true
fi

# Auto-configure CIRIS integration in background (after agent starts)
# This eliminates manual "Add Integration" steps
# Runs in background because the agent must be healthy before config flow validation works
if [ -n "$SUPERVISOR_TOKEN" ] && [ "$NEEDS_HA_RESTART" != "true" ]; then
    (
        # Wait for agent to be healthy (up to 60 seconds)
        bashio::log.info "Waiting for agent to start for auto-config..."
        for i in $(seq 1 30); do
            if curl -sf http://localhost:8099/v1/system/health >/dev/null 2>&1; then
                bashio::log.info "Agent is healthy, proceeding with auto-config"
                break
            fi
            sleep 2
        done

        # Check if CIRIS integration already exists
        EXISTING=$(curl -sf \
            -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
            "http://supervisor/core/api/config/config_entries/entry" 2>/dev/null | \
            grep -o '"domain":"ciris"' || true)

        if [ -z "$EXISTING" ]; then
            bashio::log.info "Auto-configuring CIRIS integration..."

            # Start config flow
            FLOW_RESULT=$(curl -sf -X POST \
                -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
                -H "Content-Type: application/json" \
                -d '{"handler": "ciris"}' \
                "http://supervisor/core/api/config/config_entries/flow" 2>/dev/null || echo "")

            if [ -n "$FLOW_RESULT" ]; then
                FLOW_ID=$(echo "$FLOW_RESULT" | grep -o '"flow_id":"[^"]*"' | cut -d'"' -f4)

                if [ -n "$FLOW_ID" ]; then
                    # Complete the flow with default settings
                    COMPLETE_RESULT=$(curl -sf -X POST \
                        -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
                        -H "Content-Type: application/json" \
                        -d '{"api_url": "http://local-ciris_agent:8099", "name": "CIRIS"}' \
                        "http://supervisor/core/api/config/config_entries/flow/$FLOW_ID" 2>/dev/null || echo "")

                    if echo "$COMPLETE_RESULT" | grep -q '"type":"create_entry"'; then
                        bashio::log.info "CIRIS integration auto-configured successfully!"
                    else
                        bashio::log.warning "Config flow did not create entry (may need HA restart first)"
                    fi
                fi
            else
                bashio::log.warning "Could not start config flow (HA may need restart first)"
            fi
        else
            bashio::log.info "CIRIS integration already configured."
        fi
    ) &
elif [ -n "$SUPERVISOR_TOKEN" ] && [ "$NEEDS_HA_RESTART" = "true" ]; then
    bashio::log.info "Skipping auto-config (HA restart needed to load new component)"
    bashio::log.info "After HA restart, restart this addon to auto-configure integration"
fi

# Launch CIRIS Agent with API adapter
# The agent will auto-detect HA mode via SUPERVISOR_TOKEN
exec ciris-agent --adapter api --host 0.0.0.0 --port 8099
