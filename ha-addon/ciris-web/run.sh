#!/usr/bin/with-contenv bashio

bashio::log.info "Starting CIRIS Web UI..."
bashio::log.info "Supervisor token: $(if [ -n "$SUPERVISOR_TOKEN" ]; then echo 'present'; else echo 'missing'; fi)"

cd /app
exec python3 server.py
