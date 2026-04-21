#!/bin/bash
# deploy-addon.sh
# Deploys/updates the CIRIS Agent addon on Home Assistant
#
# Usage: ./scripts/deploy-addon.sh [HA_HOST] [--fresh]
#
# Options:
#   --fresh    Uninstall and reinstall addon (forces complete rebuild)
#
# =============================================================================
# LESSONS LEARNED (from debugging session):
# =============================================================================
#
# 1. ADDON SLUG NAMING:
#    - Local addon slug uses underscore (ciris_agent) for HA compatibility
#    - Directory name doesn't have to match slug
#    - Supervisor discovers local addons as "local_<slug>"
#
# 2. SUPERVISOR_TOKEN INJECTION:
#    - Requires hassio_api: true in config.yaml
#    - Also set hassio_role: default for proper API access
#    - Token is injected by Supervisor into container environment
#    - Use #!/usr/bin/with-contenv bashio shebang (NOT /command/with-contenv)
#
# 3. DOCKER CACHING ISSUES:
#    - "ha addons rebuild" uses Docker layer cache
#    - Changes to run.sh may NOT be picked up by rebuild
#    - Use --fresh flag to force complete uninstall/reinstall
#    - Even uninstall/reinstall may use cached layers
#    - Consider bumping version in config.yaml to force rebuild
#
# 4. DEBUGGING TIPS:
#    - Echo/bashio::log in run.sh may not appear in addon logs
#    - "ha addons exec" command may not work correctly
#    - Check supervisor logs: ha supervisor logs | grep ciris
#    - Verify config.yaml syntax: python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"
#
# 5. WHEEL REQUIREMENTS:
#    - Must be pure Python wheel (py3-none-any)
#    - HA uses Alpine (musllinux), manylinux wheels won't install
#    - Deploy wheel first with deploy-agent-wheel.sh
#
# 6. CONFIG.YAML STRUCTURE:
#    - All string values should be quoted
#    - armv7 is deprecated in newer HA versions
#    - init: false for simple scripts
#    - map: share:rw to access /share directory
#
# 7. DOCKER CACHE BUSTING:
#    - HA Supervisor aggressively caches Docker layers
#    - Even uninstall/reinstall may use cached layers
#    - Use BUILD_TIMESTAMP arg in build.yaml to force rebuild
#    - This script auto-sets timestamp on each deploy
#
# 8. CIRISVERIFY BINARY (MUSL):
#    - HA addons use Alpine Linux (musl libc), NOT glibc
#    - The glibc build will fail with "symbol not found" errors
#    - Rust's musl target doesn't support cdylib directly
#    - SOLUTION: Build staticlib then link into .so with zig cc:
#      1. cargo build --release --target aarch64-unknown-linux-musl (produces .a)
#      2. zig cc -target aarch64-linux-musl -shared \
#           -o libciris_verify_ffi.so \
#           -Wl,--whole-archive libciris_verify_ffi.a -Wl,--no-whole-archive \
#           -lunwind   # <-- This resolves _Unwind_* symbols!
#    - The -lunwind flag is critical for backtrace/unwinding support on musl
#
# 9. KOTLIN WASM PRODUCTION BUILDS BROKEN (KT-69154):
#    - Kotlin WASM production builds fail with:
#      "WebAssembly.instantiate(): Import #1 'js_code' 'kotlin.wasm.internal.throwJsError':
#       function import requires a callable"
#    - This is a known Kotlin bug where the JS wrapper gets out of sync with WASM module
#    - WORKAROUND: Use developmentExecutable builds ONLY
#    - Development builds are larger (~11MB gzipped) but functional
#    - Track: https://youtrack.jetbrains.com/issue/KT-69154
#    - DO NOT use productionExecutable until this bug is fixed upstream
#
# =============================================================================

set -e

# Configuration
HA_HOST="${1:-192.168.50.243}"
HA_USER="root"
ADDON_PATH="/addons/ciris_agent"

# SSH ControlMaster - reuses single connection (saves ~0.3s per SSH call)
SSH_CONTROL_PATH="/tmp/ssh-ciris-%r@%h:%p"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=${SSH_CONTROL_PATH} -o ControlPersist=120"
SSH_CMD="ssh ${SSH_OPTS} ${HA_USER}@${HA_HOST}"
SCP_CMD="scp ${SSH_OPTS} -q"

# Cleanup SSH control socket on exit
cleanup_ssh() {
    ssh -O exit -o ControlPath="${SSH_CONTROL_PATH}" "${HA_USER}@${HA_HOST}" 2>/dev/null || true
}
trap cleanup_ssh EXIT
ADDON_SLUG="local_ciris_agent"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_ADDON_DIR="${REPO_ROOT}/ciris-agent"
VERSION="6.2.0"

# Parse arguments
FRESH_INSTALL=false
for arg in "$@"; do
    case $arg in
        --fresh)
            FRESH_INSTALL=true
            shift
            ;;
        192.168.*)
            # Skip IP addresses
            ;;
        *)
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Check SSH connectivity (also establishes ControlMaster connection)
log_step "Checking SSH connectivity to $HA_HOST..."
if ! $SSH_CMD -o ConnectTimeout=5 "echo 'SSH OK'" &>/dev/null; then
    log_error "Cannot connect to ${HA_USER}@${HA_HOST}"
    exit 1
fi

# Check local addon directory exists
if [ ! -d "$LOCAL_ADDON_DIR" ]; then
    log_error "Local addon directory not found: $LOCAL_ADDON_DIR"
    exit 1
fi

log_info "Deploying CIRIS Agent addon v${VERSION} to HA at $HA_HOST"

# Step 1: Ensure addon directory exists on HA
log_step "Creating addon directory on HA..."
$SSH_CMD "mkdir -p ${ADDON_PATH}"

# Step 2: Copy addon files (excluding wheel - that's handled by deploy-agent-wheel.sh)
log_step "Copying addon files..."
for file in Dockerfile build.yaml; do
    if [ -f "${LOCAL_ADDON_DIR}/${file}" ]; then
        $SCP_CMD "${LOCAL_ADDON_DIR}/${file}" "${HA_USER}@${HA_HOST}:${ADDON_PATH}/"
    fi
done

# Set BUILD_TIMESTAMP in build.yaml to bust Docker cache
BUILD_TS=$(date +%s)
log_info "Setting BUILD_TIMESTAMP=${BUILD_TS} for cache invalidation"
$SSH_CMD "sed -i 's/BUILD_TIMESTAMP:.*/BUILD_TIMESTAMP: \"${BUILD_TS}\"/' ${ADDON_PATH}/build.yaml"

# Copy CIRISVerify binary (lib/)
if [ -d "${LOCAL_ADDON_DIR}/lib" ]; then
    log_step "Copying CIRISVerify binary..."
    $SSH_CMD "mkdir -p ${ADDON_PATH}/lib"
    $SCP_CMD -r "${LOCAL_ADDON_DIR}/lib/"* "${HA_USER}@${HA_HOST}:${ADDON_PATH}/lib/"
    log_info "CIRISVerify binary copied"
fi

# Copy CIRIS conversation agent custom component (eliminates need for HACS)
if [ -d "${LOCAL_ADDON_DIR}/custom_components/ciris" ]; then
    log_step "Copying CIRIS conversation agent..."
    $SSH_CMD "mkdir -p ${ADDON_PATH}/custom_components"
    $SCP_CMD -r "${LOCAL_ADDON_DIR}/custom_components/ciris" "${HA_USER}@${HA_HOST}:${ADDON_PATH}/custom_components/"
    log_info "CIRIS conversation agent copied (will be installed to HA on addon start)"
fi

# Copy www from Compose Multiplatform web build output
# Priority: 1) upstream CIRISAgent KMP 2.x, 2) local mobile-web conversion
#
# IMPORTANT: ONLY use developmentExecutable builds!
# Kotlin WASM production builds have bug KT-69154 that causes:
#   "WebAssembly.instantiate(): Import #1 'js_code' 'kotlin.wasm.internal.throwJsError':
#    function import requires a callable"
# Development builds work correctly. Track KT-69154 for upstream fix.
# https://youtrack.jetbrains.com/issue/KT-69154
COMPOSE_WEB_BUILD=""

# Check upstream CIRISAgent first (native KMP 2.x) - DEVELOPMENT ONLY
UPSTREAM_WEB="${CIRIS_AGENT_DIR}/mobile/webApp/build/dist/wasmJs/developmentExecutable"
if [ -d "$UPSTREAM_WEB" ]; then
    COMPOSE_WEB_BUILD="$UPSTREAM_WEB"
    log_info "Using upstream CIRISAgent KMP 2.x web build (development)"
fi

# Fall back to local mobile-web (downstream conversion) - DEVELOPMENT ONLY
if [ -z "$COMPOSE_WEB_BUILD" ]; then
    LOCAL_WEB="${REPO_ROOT}/mobile-web/webApp/build/dist/wasmJs/developmentExecutable"
    if [ -d "$LOCAL_WEB" ]; then
        COMPOSE_WEB_BUILD="$LOCAL_WEB"
        log_info "Using local mobile-web conversion build (development)"
    fi
fi

if [ -n "$COMPOSE_WEB_BUILD" ] && [ -d "$COMPOSE_WEB_BUILD" ]; then
    log_step "Copying Compose web build from ${COMPOSE_WEB_BUILD}..."
    $SSH_CMD "rm -rf ${ADDON_PATH}/www && mkdir -p ${ADDON_PATH}/www"
    $SCP_CMD -r "${COMPOSE_WEB_BUILD}/"* "${HA_USER}@${HA_HOST}:${ADDON_PATH}/www/" 2>/dev/null || true
    log_info "Web build copied ($(ls ${COMPOSE_WEB_BUILD}/*.wasm 2>/dev/null | wc -l) wasm files)"
else
    log_warn "Compose web build not found"
    log_warn "Build upstream: cd ${CIRIS_AGENT_DIR}/mobile && ./gradlew :webApp:wasmJsBrowserDevelopmentExecutable"
    log_warn "Or local:       cd ${REPO_ROOT}/mobile-web && ./gradlew :webApp:wasmJsBrowserDevelopmentExecutable"
    log_warn "NOTE: Production builds broken due to Kotlin WASM bug KT-69154"
fi

# Copy wheel file - prefer upstream CIRISAgent build, fall back to local
CIRIS_AGENT_DIR="${CIRIS_AGENT_DIR:-$(cd "$REPO_ROOT/.." && pwd)/CIRISAgent}"
UPSTREAM_WHEEL=$(ls "${CIRIS_AGENT_DIR}/dist"/ciris_agent-*.whl 2>/dev/null | sort -V | tail -1)
LOCAL_WHEEL=$(ls "${LOCAL_ADDON_DIR}"/*.whl 2>/dev/null | head -1)

# Check what's already on HA
REMOTE_WHEEL=$($SSH_CMD "ls ${ADDON_PATH}/ciris_agent-*.whl 2>/dev/null | sort -V | tail -1" || echo "")
REMOTE_VERSION=""
if [ -n "$REMOTE_WHEEL" ]; then
    REMOTE_VERSION=$(basename "$REMOTE_WHEEL" | sed 's/ciris_agent-\([^-]*\)-.*/\1/')
fi

if [ -n "$UPSTREAM_WHEEL" ] && [ -f "$UPSTREAM_WHEEL" ]; then
    UPSTREAM_VERSION=$(basename "$UPSTREAM_WHEEL" | sed 's/ciris_agent-\([^-]*\)-.*/\1/')
    if [ "$UPSTREAM_VERSION" != "$REMOTE_VERSION" ]; then
        log_step "Copying upstream wheel: $(basename $UPSTREAM_WHEEL)..."
        # Remove old wheels first
        $SSH_CMD "rm -f ${ADDON_PATH}/ciris_agent-*.whl"
        $SCP_CMD "$UPSTREAM_WHEEL" "${HA_USER}@${HA_HOST}:${ADDON_PATH}/"
    else
        log_info "Upstream wheel v${UPSTREAM_VERSION} already deployed, skipping"
    fi
elif [ -n "$LOCAL_WHEEL" ] && [ -f "$LOCAL_WHEEL" ]; then
    LOCAL_VERSION=$(basename "$LOCAL_WHEEL" | sed 's/ciris_agent-\([^-]*\)-.*/\1/')
    if [ "$LOCAL_VERSION" != "$REMOTE_VERSION" ]; then
        log_step "Copying local wheel: $(basename $LOCAL_WHEEL)..."
        $SSH_CMD "rm -f ${ADDON_PATH}/ciris_agent-*.whl"
        $SCP_CMD "$LOCAL_WHEEL" "${HA_USER}@${HA_HOST}:${ADDON_PATH}/"
    else
        log_info "Local wheel v${LOCAL_VERSION} already deployed, skipping"
    fi
else
    if [ -n "$REMOTE_WHEEL" ]; then
        log_info "Using existing wheel on HA: $(basename $REMOTE_WHEEL)"
    else
        log_warn "No wheel file found (checked upstream: ${CIRIS_AGENT_DIR}/dist/, local: ${LOCAL_ADDON_DIR}/)"
    fi
fi

# Step 3: Generate config.yaml with correct settings
log_step "Generating config.yaml..."
$SSH_CMD "cat > ${ADDON_PATH}/config.yaml << 'CONFIGEOF'
name: \"CIRIS Agent\"
description: \"CIRIS AI Agent for Home Assistant - Multi-modal AI home automation\"
version: \"${VERSION}\"
slug: \"ciris_agent\"
url: \"https://github.com/CIRISAI/CIRISHome\"
arch:
  - aarch64
  - amd64
init: false
ingress: true
ingress_port: 8099
ingress_stream: true
ingress_panel: true
panel_icon: mdi:robot
panel_title: CIRIS
homeassistant_api: true
hassio_api: true
hassio_role: default
auth_api: true
startup: services
map:
  - share:rw
  - config:rw
ports:
  8099/tcp: null
ports_description:
  8099/tcp: \"CIRIS Agent Web UI (internal via ingress)\"
options: {}
schema: {}
CONFIGEOF"

# Step 4: Generate run.sh with correct shebang for SUPERVISOR_TOKEN
# CRITICAL: Must use /usr/bin/with-contenv, NOT /command/with-contenv
log_step "Generating run.sh..."
$SSH_CMD "cat > ${ADDON_PATH}/run.sh << 'RUNEOF'
#!/usr/bin/with-contenv bashio
# CIRIS Agent startup script for Home Assistant Addon
#
# IMPORTANT: The shebang above (#!/usr/bin/with-contenv bashio) is REQUIRED
# for SUPERVISOR_TOKEN to be available. Do NOT use /command/with-contenv.

set -e

# Set CIRIS_HOME and CIRIS_CONFIG_DIR to addon data directory (persistent storage)
# /root is forbidden as a system directory, /data is HA addon persistent storage
# CIRIS_CONFIG_DIR is needed for first_run.py to find the .env file in INSTALLED mode
export CIRIS_HOME=/data/ciris
export CIRIS_CONFIG_DIR=/data/ciris

# Ensure directory exists
mkdir -p \"\$CIRIS_HOME\"
mkdir -p \"\$CIRIS_HOME/logs\"
mkdir -p /share/ciris_logs

# Copy any existing logs/env to /share for debugging (run in background)
(while true; do
    cp \"\$CIRIS_HOME/logs/latest.log\" /share/ciris_logs/latest.log 2>/dev/null || true
    cp \"\$CIRIS_HOME/logs/incidents_latest.log\" /share/ciris_logs/incidents.log 2>/dev/null || true
    cp \"\$CIRIS_HOME/.env\" /share/ciris_logs/env.txt 2>/dev/null || true
    sleep 5
done) &

# Track if HA needs restart (set by component install)
NEEDS_HA_RESTART=false

# Install CIRIS conversation agent custom component (eliminates need for HACS)
# This copies the bundled custom_components/ciris to HA's config directory
echo \"[CIRIS STARTUP] Installing CIRIS conversation agent...\" >> /share/ciris_logs/startup.log
if [ -d /app/custom_components/ciris ]; then
    mkdir -p /config/custom_components
    # Get versions for comparison
    BUNDLED_VERSION=\$(grep -o '\"version\": *\"[^\"]*\"' /app/custom_components/ciris/manifest.json | cut -d'\"' -f4)
    INSTALLED_VERSION=\"\"
    if [ -f /config/custom_components/ciris/manifest.json ]; then
        INSTALLED_VERSION=\$(grep -o '\"version\": *\"[^\"]*\"' /config/custom_components/ciris/manifest.json | cut -d'\"' -f4)
    fi
    echo \"[CIRIS STARTUP] Bundled version: \$BUNDLED_VERSION, Installed version: \$INSTALLED_VERSION\" >> /share/ciris_logs/startup.log
    # Always update if versions differ or not installed
    if [ \"\$BUNDLED_VERSION\" != \"\$INSTALLED_VERSION\" ]; then
        rm -rf /config/custom_components/ciris
        cp -r /app/custom_components/ciris /config/custom_components/
        echo \"[CIRIS STARTUP] CIRIS conversation agent v\$BUNDLED_VERSION installed to /config/custom_components/ciris\" >> /share/ciris_logs/startup.log
        echo \"[CIRIS STARTUP] NOTE: HA restart may be required to load new component version\" >> /share/ciris_logs/startup.log
        NEEDS_HA_RESTART=true
    else
        echo \"[CIRIS STARTUP] CIRIS conversation agent v\$INSTALLED_VERSION already up to date\" >> /share/ciris_logs/startup.log
    fi
else
    echo \"[CIRIS STARTUP] WARNING: custom_components/ciris not found in addon\" >> /share/ciris_logs/startup.log
fi

# Enable sidebar panel via Supervisor API
# This ensures CIRIS appears in the HA sidebar automatically
echo \"[CIRIS STARTUP] Checking SUPERVISOR_TOKEN...\" >> /share/ciris_logs/startup.log
if [ -n \"\$SUPERVISOR_TOKEN\" ]; then
    echo \"[CIRIS STARTUP] SUPERVISOR_TOKEN present (length: \${#SUPERVISOR_TOKEN}), enabling ingress panel...\" >> /share/ciris_logs/startup.log
    PANEL_RESULT=\$(curl -s -w \" HTTP_%{http_code}\" -X POST \\
        -H \"Authorization: Bearer \$SUPERVISOR_TOKEN\" \\
        -H \"Content-Type: application/json\" \\
        -d '{\"ingress_panel\": true}' \\
        http://supervisor/addons/self/options 2>&1)
    echo \"[CIRIS STARTUP] Panel enable result: \$PANEL_RESULT\" >> /share/ciris_logs/startup.log
else
    echo \"[CIRIS STARTUP] WARNING: SUPERVISOR_TOKEN not set, cannot enable panel\" >> /share/ciris_logs/startup.log
fi

# Auto-configure CIRIS integration in background (after agent starts)
# This eliminates manual "Add Integration" steps
# Runs in background because the agent must be healthy before config flow validation works
# Skip if HA restart is needed (component not loaded yet)
if [ -n \"\$SUPERVISOR_TOKEN\" ] && [ \"\$NEEDS_HA_RESTART\" != \"true\" ]; then
    (
        # Wait for agent to be healthy (up to 60 seconds)
        echo \"[CIRIS AUTO-CONFIG] Waiting for agent to start...\" >> /share/ciris_logs/startup.log
        for i in \$(seq 1 30); do
            if curl -sf http://localhost:8099/v1/system/health >/dev/null 2>&1; then
                echo \"[CIRIS AUTO-CONFIG] Agent is healthy, proceeding with auto-config\" >> /share/ciris_logs/startup.log
                break
            fi
            sleep 2
        done

        # Check if CIRIS integration already exists
        EXISTING=\$(curl -sf \\
            -H \"Authorization: Bearer \$SUPERVISOR_TOKEN\" \\
            \"http://supervisor/core/api/config/config_entries/entry\" 2>/dev/null | \\
            grep -o '\"domain\":\"ciris\"' || true)

        if [ -z \"\$EXISTING\" ]; then
            echo \"[CIRIS AUTO-CONFIG] CIRIS integration not found, attempting auto-configure...\" >> /share/ciris_logs/startup.log

            # Start config flow
            FLOW_RESULT=\$(curl -sf -X POST \\
                -H \"Authorization: Bearer \$SUPERVISOR_TOKEN\" \\
                -H \"Content-Type: application/json\" \\
                -d '{\"handler\": \"ciris\"}' \\
                \"http://supervisor/core/api/config/config_entries/flow\" 2>/dev/null || echo \"\")

            echo \"[CIRIS AUTO-CONFIG] Flow init result: \$FLOW_RESULT\" >> /share/ciris_logs/startup.log

            if [ -n \"\$FLOW_RESULT\" ]; then
                FLOW_ID=\$(echo \"\$FLOW_RESULT\" | grep -o '\"flow_id\":\"[^\"]*\"' | cut -d'\"' -f4)

                if [ -n \"\$FLOW_ID\" ]; then
                    echo \"[CIRIS AUTO-CONFIG] Completing flow \$FLOW_ID with defaults...\" >> /share/ciris_logs/startup.log

                    # Complete the flow with default settings
                    COMPLETE_RESULT=\$(curl -sf -X POST \\
                        -H \"Authorization: Bearer \$SUPERVISOR_TOKEN\" \\
                        -H \"Content-Type: application/json\" \\
                        -d '{\"api_url\": \"http://local-ciris_agent:8099\", \"name\": \"CIRIS\"}' \\
                        \"http://supervisor/core/api/config/config_entries/flow/\$FLOW_ID\" 2>/dev/null || echo \"\")

                    echo \"[CIRIS AUTO-CONFIG] Flow complete result: \$COMPLETE_RESULT\" >> /share/ciris_logs/startup.log

                    if echo \"\$COMPLETE_RESULT\" | grep -q '\"type\":\"create_entry\"'; then
                        echo \"[CIRIS AUTO-CONFIG] SUCCESS: CIRIS integration auto-configured!\" >> /share/ciris_logs/startup.log
                    else
                        echo \"[CIRIS AUTO-CONFIG] Config flow did not create entry (may need HA restart first)\" >> /share/ciris_logs/startup.log
                    fi
                fi
            fi
        else
            echo \"[CIRIS AUTO-CONFIG] CIRIS integration already configured.\" >> /share/ciris_logs/startup.log
        fi
    ) &
elif [ -n \"\$SUPERVISOR_TOKEN\" ] && [ \"\$NEEDS_HA_RESTART\" = \"true\" ]; then
    echo \"[CIRIS AUTO-CONFIG] Skipping auto-config (HA restart needed to load new component)\" >> /share/ciris_logs/startup.log
    echo \"[CIRIS AUTO-CONFIG] After HA restart, restart this addon to auto-configure integration\" >> /share/ciris_logs/startup.log
fi

# Launch CIRIS Agent with API adapter
# The agent will auto-detect HA mode via SUPERVISOR_TOKEN
exec ciris-agent --adapter api --host 0.0.0.0 --port 8099
RUNEOF
chmod +x ${ADDON_PATH}/run.sh"

# Step 5: Set correct permissions
log_step "Setting file permissions..."
$SSH_CMD "chmod 755 ${ADDON_PATH} && chmod 644 ${ADDON_PATH}/*.yaml ${ADDON_PATH}/Dockerfile 2>/dev/null || true && chmod 755 ${ADDON_PATH}/run.sh"

# Step 6: Reload addon store
log_step "Reloading addon store..."
$SSH_CMD "ha addons reload" >/dev/null 2>&1
sleep 3

# Step 7: Check if addon is discovered
log_step "Checking addon discovery..."
ADDON_INFO=$($SSH_CMD "ha addons info ${ADDON_SLUG} 2>&1" || echo "NOT_FOUND")

if [[ "$ADDON_INFO" == *"NOT_FOUND"* ]] || [[ "$ADDON_INFO" == *"Addon not found"* ]]; then
    log_error "Addon not discovered. Check config.yaml syntax."
    log_info "Verify: ssh ${HA_USER}@${HA_HOST} 'cat ${ADDON_PATH}/config.yaml'"
    exit 1
fi

log_info "Addon discovered: ${ADDON_SLUG}"

# Helper: Wait for addon to reach a specific state
wait_for_addon_state() {
    local target_state="$1"
    local max_wait="$2"
    local elapsed=0
    while [ $elapsed -lt $max_wait ]; do
        STATE=$($SSH_CMD "ha addons info ${ADDON_SLUG} 2>&1 | grep 'state:' | awk '{print \$2}'" || echo "unknown")
        if [ "$STATE" = "$target_state" ]; then
            return 0
        fi
        # If we're waiting for "started" but it's "stopped", addon exists - we can start it
        if [ "$target_state" = "started" ] && [ "$STATE" = "stopped" ]; then
            return 1  # Signal caller to start it
        fi
        sleep 3
        elapsed=$((elapsed + 3))
        echo -n "."
    done
    return 2  # Timeout
}

# Step 8: Fresh install or rebuild
if [ "$FRESH_INSTALL" = true ]; then
    log_step "Performing fresh install (uninstall + install)..."
    log_warn "This will remove existing addon data and force Docker rebuild"
    $SSH_CMD "ha addons uninstall ${ADDON_SLUG} 2>/dev/null || true"
    sleep 3

    log_info "Installing addon (this may take several minutes for Docker build)..."
    $SSH_CMD "ha addons install ${ADDON_SLUG}" &
    INSTALL_PID=$!

    # Wait for SSH command to complete (up to 5 min)
    TIMEOUT=300
    ELAPSED=0
    while kill -0 $INSTALL_PID 2>/dev/null; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        echo -n "."
        if [ $ELAPSED -ge $TIMEOUT ]; then
            log_warn "Install SSH timeout, polling addon state..."
            break
        fi
    done
    echo ""

    # Poll until addon exists (stopped state means install complete)
    log_info "Waiting for install to complete..."
    for i in {1..60}; do
        STATE=$($SSH_CMD "ha addons info ${ADDON_SLUG} 2>&1 | grep 'state:' | awk '{print \$2}'" || echo "unknown")
        if [ "$STATE" = "stopped" ] || [ "$STATE" = "started" ]; then
            log_info "Install complete (state: $STATE)"
            break
        fi
        sleep 3
        echo -n "."
    done
    echo ""
else
    # Check if already installed
    INSTALLED=$($SSH_CMD "ha addons info ${ADDON_SLUG} 2>&1 | grep -c 'state: started\|state: stopped'" || echo "0")

    if [ "$INSTALLED" -gt 0 ]; then
        log_step "Rebuilding existing addon..."
        log_warn "Note: Docker rebuild may use cached layers. Use --fresh if run.sh changes aren't picked up."
        $SSH_CMD "ha addons rebuild ${ADDON_SLUG}" 2>&1 | head -5 || true
        # Rebuild auto-starts, wait for it
        log_info "Waiting for rebuild..."
        wait_for_addon_state "started" 120 || true
        echo ""
    else
        log_step "Installing addon (this may take several minutes for Docker build)..."
        $SSH_CMD "ha addons install ${ADDON_SLUG}" &
        INSTALL_PID=$!

        TIMEOUT=300
        ELAPSED=0
        while kill -0 $INSTALL_PID 2>/dev/null; do
            sleep 5
            ELAPSED=$((ELAPSED + 5))
            echo -n "."
            if [ $ELAPSED -ge $TIMEOUT ]; then
                log_warn "Install taking longer than expected, polling state..."
                break
            fi
        done
        echo ""

        # Poll until stopped (install complete)
        for i in {1..60}; do
            STATE=$($SSH_CMD "ha addons info ${ADDON_SLUG} 2>&1 | grep 'state:' | awk '{print \$2}'" || echo "unknown")
            [ "$STATE" = "stopped" ] || [ "$STATE" = "started" ] && break
            sleep 3
        done
    fi
fi

# Step 9: Start addon if not running (synchronous, not background)
log_step "Checking addon state..."
ADDON_STATE=$($SSH_CMD "ha addons info ${ADDON_SLUG} 2>&1 | grep 'state:' | awk '{print \$2}'" || echo "unknown")

if [ "$ADDON_STATE" != "started" ]; then
    log_step "Starting addon..."
    # Run synchronously with timeout, capture output for debugging
    $SSH_CMD "timeout 60 ha addons start ${ADDON_SLUG}" 2>&1 || log_warn "Start command timed out, addon may still be starting"
fi

# Step 10: Wait for addon to be fully started (shorter, we already waited above)
log_step "Waiting for addon to be ready..."
wait_for_addon_state "started" 60
if [ $? -eq 0 ]; then
    log_info "Addon started successfully"
else
    log_warn "Addon may still be starting, check logs"
fi

# Step 11: Sidebar panel is now self-enabled by the addon on startup via run.sh
# The addon calls the Supervisor API to enable ingress_panel automatically

# Step 12: Show final status
echo ""
log_step "Final addon status:"
$SSH_CMD "ha addons info ${ADDON_SLUG} 2>&1 | grep -E 'state|version|ingress_url'"

echo ""
log_info "Deployment complete!"
echo ""
echo "Useful commands:"
echo "  View logs:     ssh ${HA_USER}@${HA_HOST} 'ha addons logs ${ADDON_SLUG}'"
echo "  Restart:       ssh ${HA_USER}@${HA_HOST} 'ha addons restart ${ADDON_SLUG}'"
echo "  Rebuild:       ssh ${HA_USER}@${HA_HOST} 'ha addons rebuild ${ADDON_SLUG}'"
echo "  Fresh deploy:  $0 ${HA_HOST} --fresh"
echo ""
echo "Troubleshooting SUPERVISOR_TOKEN:"
echo "  1. Check supervisor logs: ssh ${HA_USER}@${HA_HOST} 'ha supervisor logs | grep ciris'"
echo "  2. Verify config: ssh ${HA_USER}@${HA_HOST} 'cat ${ADDON_PATH}/config.yaml | grep hassio'"
echo "  3. If token missing after rebuild, try --fresh flag"
