#!/command/with-contenv bashio
# shellcheck shell=bash

bashio::log.info "Starting CIRIS Web UI..."
bashio::log.info "Supervisor token: $(if [ -n "$SUPERVISOR_TOKEN" ]; then echo 'present'; else echo 'missing'; fi)"

cd /app || exit 1
exec python3 server.py
