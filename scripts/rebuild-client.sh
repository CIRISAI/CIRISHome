#!/bin/bash
# rebuild-client.sh
# Completely rebuilds the CIRIS web client (WASM) from scratch
#
# Usage: ./scripts/rebuild-client.sh [--deploy] [--production]
#   --deploy      Also deploy to Home Assistant after building
#   --production  Build production (optimized) instead of development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MOBILE_WEB_DIR="$REPO_ROOT/mobile-web"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${CYAN}[STEP]${NC} $1"; }

# Parse arguments
DEPLOY=false
PRODUCTION=false
for arg in "$@"; do
    case $arg in
        --deploy)
            DEPLOY=true
            ;;
        --production)
            PRODUCTION=true
            ;;
    esac
done

# Check mobile-web directory exists
if [ ! -d "$MOBILE_WEB_DIR" ]; then
    log_error "mobile-web directory not found: $MOBILE_WEB_DIR"
    exit 1
fi

log_info "Building WASM client from: $MOBILE_WEB_DIR"

cd "$MOBILE_WEB_DIR"

# Step 1: Clean previous builds completely
log_step "Cleaning previous builds..."
rm -rf shared/build webApp/build .gradle/caches/kotlin-data-container 2>/dev/null || true
./gradlew clean 2>/dev/null || true

# Step 2: Build the WASM client
if [ "$PRODUCTION" = true ]; then
    log_step "Building production WASM (optimized, slower)..."
    BUILD_TASK=":webApp:wasmJsBrowserDistribution"
    BUILD_DIR="webApp/build/dist/wasmJs/productionExecutable"
else
    log_step "Building development WASM (faster)..."
    BUILD_TASK=":webApp:wasmJsBrowserDevelopmentExecutableDistribution"
    BUILD_DIR="webApp/build/dist/wasmJs/developmentExecutable"
fi

./gradlew $BUILD_TASK 2>&1 | tail -30

# Step 3: Verify build output
if [ ! -d "$BUILD_DIR" ]; then
    log_error "Build failed - output directory not found: $BUILD_DIR"
    exit 1
fi

WASM_COUNT=$(ls "$BUILD_DIR"/*.wasm 2>/dev/null | wc -l)
if [ "$WASM_COUNT" -eq 0 ]; then
    log_error "Build failed - no WASM files found in $BUILD_DIR"
    exit 1
fi

log_info "Build successful: $WASM_COUNT WASM files"
log_info "Output: $BUILD_DIR"

# Step 4: List build contents
log_step "Build contents:"
ls -la "$BUILD_DIR" | head -15

# Step 5: Deploy if requested
if [ "$DEPLOY" = true ]; then
    log_step "Deploying to Home Assistant..."

    HA_HOST="${HA_HOST:-192.168.50.243}"
    HA_USER="root"
    ADDON_PATH="/addons/ciris_agent"

    # SSH ControlMaster for faster operations
    SSH_CONTROL_PATH="/tmp/ssh-ciris-%r@%h:%p"
    SSH_OPTS="-o ControlMaster=auto -o ControlPath=${SSH_CONTROL_PATH} -o ControlPersist=60"
    SSH_CMD="ssh ${SSH_OPTS} ${HA_USER}@${HA_HOST}"
    SCP_CMD="scp ${SSH_OPTS} -q"

    # Cleanup SSH control socket on exit
    cleanup_ssh() {
        ssh -O exit -o ControlPath="${SSH_CONTROL_PATH}" "${HA_USER}@${HA_HOST}" 2>/dev/null || true
    }
    trap cleanup_ssh EXIT

    log_info "Copying to $HA_HOST..."
    $SSH_CMD "rm -rf ${ADDON_PATH}/www && mkdir -p ${ADDON_PATH}/www"
    $SCP_CMD -r "${BUILD_DIR}/"* "${HA_USER}@${HA_HOST}:${ADDON_PATH}/www/"

    log_info "Deployed successfully to ${ADDON_PATH}/www/"

    # Verify deployment
    REMOTE_WASM=$($SSH_CMD "ls ${ADDON_PATH}/www/*.wasm 2>/dev/null | wc -l")
    log_info "Remote WASM files: $REMOTE_WASM"
fi

echo ""
log_info "Client rebuild complete!"
echo ""
echo "Build type: $([ "$PRODUCTION" = true ] && echo "Production" || echo "Development")"
echo "Output: $BUILD_DIR"
echo ""
if [ "$DEPLOY" = false ]; then
    echo "To deploy to Home Assistant:"
    echo "  $0 --deploy"
    echo ""
    echo "Or manually:"
    echo "  scp -r $BUILD_DIR/* root@192.168.50.243:/addons/ciris_agent/www/"
fi
