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
#    - Local addon slug MUST use underscore (ciris_agent), NOT hyphen
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
# =============================================================================

set -e

# Configuration
HA_HOST="${1:-192.168.50.243}"
HA_USER="root"
ADDON_PATH="/addons/ciris_agent"
ADDON_SLUG="local_ciris_agent"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
LOCAL_ADDON_DIR="${REPO_ROOT}/ciris-agent"
VERSION="6.0.7"

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

# Check SSH connectivity
log_step "Checking SSH connectivity to $HA_HOST..."
if ! ssh -o ConnectTimeout=5 "${HA_USER}@${HA_HOST}" "echo 'SSH OK'" &>/dev/null; then
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
ssh "${HA_USER}@${HA_HOST}" "mkdir -p ${ADDON_PATH}"

# Step 2: Copy addon files (excluding wheel - that's handled by deploy-agent-wheel.sh)
log_step "Copying addon files..."
for file in Dockerfile build.yaml; do
    if [ -f "${LOCAL_ADDON_DIR}/${file}" ]; then
        scp -q "${LOCAL_ADDON_DIR}/${file}" "${HA_USER}@${HA_HOST}:${ADDON_PATH}/"
    fi
done

# Copy www from Compose Multiplatform web build output
COMPOSE_WEB_BUILD="${REPO_ROOT}/mobile-web/webApp/build/dist/wasmJs/developmentExecutable"
if [ -d "${COMPOSE_WEB_BUILD}" ]; then
    log_step "Copying Compose web build (this may take a moment)..."
    ssh "${HA_USER}@${HA_HOST}" "mkdir -p ${ADDON_PATH}/www"
    scp -qr "${COMPOSE_WEB_BUILD}/"* "${HA_USER}@${HA_HOST}:${ADDON_PATH}/www/" 2>/dev/null || true
else
    log_warn "Compose web build not found at ${COMPOSE_WEB_BUILD}"
    log_warn "Run: cd mobile-web && ./gradlew wasmJsBrowserDevelopmentExecutableDistribution"
fi

# Step 3: Generate config.yaml with correct settings
log_step "Generating config.yaml..."
ssh "${HA_USER}@${HA_HOST}" "cat > ${ADDON_PATH}/config.yaml << 'CONFIGEOF'
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
ssh "${HA_USER}@${HA_HOST}" "cat > ${ADDON_PATH}/run.sh << 'RUNEOF'
#!/usr/bin/with-contenv bashio
# CIRIS Agent startup script for Home Assistant Addon
#
# IMPORTANT: The shebang above (#!/usr/bin/with-contenv bashio) is REQUIRED
# for SUPERVISOR_TOKEN to be available. Do NOT use /command/with-contenv.

set -e

# Set CIRIS_HOME to addon data directory (persistent storage)
# /root is forbidden as a system directory, /data is HA addon persistent storage
export CIRIS_HOME=/data/ciris

# Ensure directory exists
mkdir -p \"\$CIRIS_HOME\"

# Enable sidebar panel via Supervisor API
# This ensures CIRIS appears in the HA sidebar automatically
if [ -n \"\$SUPERVISOR_TOKEN\" ]; then
    curl -sf -X POST \\
        -H \"Authorization: Bearer \$SUPERVISOR_TOKEN\" \\
        -H \"Content-Type: application/json\" \\
        -d '{\"ingress_panel\": true}' \\
        http://supervisor/addons/self/options >/dev/null 2>&1 || true
fi

# Launch CIRIS Agent with API adapter
# The agent will auto-detect HA mode via SUPERVISOR_TOKEN
exec ciris-agent --adapter api --host 0.0.0.0 --port 8099
RUNEOF
chmod +x ${ADDON_PATH}/run.sh"

# Step 5: Set correct permissions
log_step "Setting file permissions..."
ssh "${HA_USER}@${HA_HOST}" "chmod 755 ${ADDON_PATH} && chmod 644 ${ADDON_PATH}/*.yaml ${ADDON_PATH}/Dockerfile 2>/dev/null || true && chmod 755 ${ADDON_PATH}/run.sh"

# Step 6: Reload addon store
log_step "Reloading addon store..."
ssh "${HA_USER}@${HA_HOST}" "ha addons reload" >/dev/null 2>&1
sleep 3

# Step 7: Check if addon is discovered
log_step "Checking addon discovery..."
ADDON_INFO=$(ssh "${HA_USER}@${HA_HOST}" "ha addons info ${ADDON_SLUG} 2>&1" || echo "NOT_FOUND")

if [[ "$ADDON_INFO" == *"NOT_FOUND"* ]] || [[ "$ADDON_INFO" == *"Addon not found"* ]]; then
    log_error "Addon not discovered. Check config.yaml syntax."
    log_info "Verify: ssh ${HA_USER}@${HA_HOST} 'cat ${ADDON_PATH}/config.yaml'"
    exit 1
fi

log_info "Addon discovered: ${ADDON_SLUG}"

# Step 8: Fresh install or rebuild
if [ "$FRESH_INSTALL" = true ]; then
    log_step "Performing fresh install (uninstall + install)..."
    log_warn "This will remove existing addon data and force Docker rebuild"
    ssh "${HA_USER}@${HA_HOST}" "ha addons uninstall ${ADDON_SLUG} 2>/dev/null || true"
    sleep 5

    log_info "Installing addon (this may take several minutes for Docker build)..."
    # Run install in background and monitor
    ssh "${HA_USER}@${HA_HOST}" "ha addons install ${ADDON_SLUG}" &
    INSTALL_PID=$!

    TIMEOUT=300
    ELAPSED=0
    while kill -0 $INSTALL_PID 2>/dev/null; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if [ $ELAPSED -ge $TIMEOUT ]; then
            log_warn "Install taking longer than expected, continuing in background..."
            break
        fi
        echo -n "."
    done
    echo ""
else
    # Check if already installed
    INSTALLED=$(ssh "${HA_USER}@${HA_HOST}" "ha addons info ${ADDON_SLUG} 2>&1 | grep -c 'state: started\|state: stopped'" || echo "0")

    if [ "$INSTALLED" -gt 0 ]; then
        log_step "Rebuilding existing addon..."
        log_warn "Note: Docker rebuild may use cached layers. Use --fresh if run.sh changes aren't picked up."
        ssh "${HA_USER}@${HA_HOST}" "ha addons rebuild ${ADDON_SLUG}" >/dev/null 2>&1 || true
    else
        log_step "Installing addon (this may take several minutes for Docker build)..."
        ssh "${HA_USER}@${HA_HOST}" "ha addons install ${ADDON_SLUG}" &
        INSTALL_PID=$!

        TIMEOUT=300
        ELAPSED=0
        while kill -0 $INSTALL_PID 2>/dev/null; do
            sleep 5
            ELAPSED=$((ELAPSED + 5))
            if [ $ELAPSED -ge $TIMEOUT ]; then
                log_warn "Install taking longer than expected..."
                break
            fi
            echo -n "."
        done
        echo ""
    fi
fi

sleep 5

# Step 9: Start addon if not running
log_step "Checking addon state..."
ADDON_STATE=$(ssh "${HA_USER}@${HA_HOST}" "ha addons info ${ADDON_SLUG} 2>&1 | grep 'state:'" || echo "state: unknown")

if [[ "$ADDON_STATE" != *"started"* ]]; then
    log_step "Starting addon..."
    ssh "${HA_USER}@${HA_HOST}" "ha addons start ${ADDON_SLUG}" &>/dev/null &
    sleep 10
fi

# Step 10: Wait for addon to be fully started
log_step "Waiting for addon to be ready..."
for i in {1..12}; do
    STATE=$(ssh "${HA_USER}@${HA_HOST}" "ha addons info ${ADDON_SLUG} 2>&1 | grep 'state:' | awk '{print \$2}'" || echo "unknown")
    if [ "$STATE" = "started" ]; then
        break
    fi
    sleep 5
done

# Step 11: Sidebar panel is now self-enabled by the addon on startup via run.sh
# The addon calls the Supervisor API to enable ingress_panel automatically

# Step 12: Show final status
echo ""
log_step "Final addon status:"
ssh "${HA_USER}@${HA_HOST}" "ha addons info ${ADDON_SLUG} 2>&1 | grep -E 'state|version|ingress_url'"

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
