#!/bin/bash
# deploy-agent-wheel.sh
# Builds and deploys CIRISAgent wheel to Home Assistant addon directory
# Also fetches the correct CIRISVerify musl binary for Alpine Linux
# Supports KMP 2.x with optional WASM web app build
#
# Usage: ./scripts/deploy-agent-wheel.sh [HA_HOST] [--skip-verify] [--build-web]
#
# Options:
#   --skip-verify  Skip downloading CIRISVerify binary (use existing)
#   --build-web    Build and deploy WASM web app (requires KMP 2.x upstream)
#
# Environment variables:
#   CIRIS_AGENT_DIR    - Path to CIRISAgent directory (default: ../CIRISAgent)
#   CIRIS_VERIFY_VER   - CIRISVerify version to download (default: auto-detect latest)
#
# =============================================================================
# LESSONS LEARNED:
# =============================================================================
#
# 1. WHEEL PLATFORM COMPATIBILITY:
#    - Must build pure Python wheel (py3-none-any), NOT manylinux
#    - HA uses Alpine Linux which uses musllinux, not glibc
#    - manylinux wheels (built on glibc systems) won't install on Alpine
#    - Error: "is not a supported wheel on this platform"
#
# 2. PLATFORM-SPECIFIC FILES:
#    - Any .jar, .so, .dll, or binary files will cause manylinux wheel
#    - Must temporarily remove these files before building
#    - The CIRIS desktop app JAR file is a common culprit
#
# 3. BUILD PROCESS:
#    - Use python3 -m build --wheel for reproducible builds
#    - Clean dist/, build/, *.egg-info before building
#    - Verify wheel filename contains "py3-none-any"
#
# 4. DEPLOYMENT:
#    - Remove old wheels before copying new ones
#    - Wheel goes to /addons/ciris_agent/ on HA
#    - After deploying wheel, run deploy-addon.sh --fresh to rebuild
#
# 5. CIRISVERIFY MUSL BINARY:
#    - HA addons use Alpine Linux (musl libc), NOT glibc
#    - Must download the musl-specific binary from GitHub releases
#    - Asset name: ciris-verify-vX.Y.Z-linux-arm64-musl.tar.gz
#    - The -lunwind flag is required for _Unwind_* symbol resolution
#
# 6. KMP 2.x UPSTREAM (2025+):
#    - CIRISAgent migrated to Kotlin 2.0.21 + Compose 1.7.1
#    - Native wasmJs target for web deployment
#    - AGP 8.5.2 (compatible with Chaquopy 17.0.0)
#    - Build web: ./gradlew :webApp:wasmJsBrowserDistribution
#    - Web output: mobile/webApp/build/dist/wasmJs/productionExecutable/
#    - Replaces separate CIRISHome/mobile-web conversion process
#
# =============================================================================

set -e

# Configuration defaults
HA_HOST="192.168.50.243"
HA_USER="root"
CIRIS_AGENT_DIR="${CIRIS_AGENT_DIR:-../CIRISAgent}"
ADDON_PATH="/addons/ciris_agent"
CIRIS_VERIFY_REPO="CIRISAI/CIRISVerify"
CIRIS_HOME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse arguments
SKIP_VERIFY=false
BUILD_WEB=false
for arg in "$@"; do
    case $arg in
        --skip-verify)
            SKIP_VERIFY=true
            ;;
        --build-web)
            BUILD_WEB=true
            ;;
        192.168.*|10.*|172.*)
            HA_HOST="$arg"
            ;;
        *)
            # Check if it looks like a hostname
            if [[ "$arg" =~ ^[a-zA-Z] ]] && [[ ! "$arg" =~ ^-- ]]; then
                HA_HOST="$arg"
            fi
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

# Resolve relative path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$CIRIS_AGENT_DIR" == ../* ]]; then
    CIRIS_AGENT_DIR="$(cd "$SCRIPT_DIR" && cd "$CIRIS_AGENT_DIR" 2>/dev/null && pwd)" || true
fi

# Check prerequisites
if [ ! -d "$CIRIS_AGENT_DIR" ]; then
    log_error "CIRISAgent directory not found at: $CIRIS_AGENT_DIR"
    log_info "Set CIRIS_AGENT_DIR environment variable or ensure ../CIRISAgent exists relative to CIRISHome"
    exit 1
fi

# Check for python build module
if ! python3 -c "import build" 2>/dev/null; then
    log_error "Python 'build' module not found. Install with: pip install build"
    exit 1
fi

cd "$CIRIS_AGENT_DIR"
log_info "Working in: $(pwd)"

# Check SSH connectivity
log_step "Checking SSH connectivity to $HA_HOST..."
if ! ssh -o ConnectTimeout=5 "${HA_USER}@${HA_HOST}" "echo 'SSH OK'" &>/dev/null; then
    log_error "Cannot connect to ${HA_USER}@${HA_HOST}"
    exit 1
fi

# =============================================================================
# CIRISVerify musl binary download
# =============================================================================
download_cirisverify_musl() {
    local lib_dir="${CIRIS_HOME_DIR}/ciris-agent/lib"
    local target_file="${lib_dir}/libciris_verify.so"

    # Get latest version if not specified
    local version="${CIRIS_VERIFY_VER:-}"
    if [ -z "$version" ]; then
        log_step "Fetching latest CIRISVerify version..."
        version=$(gh release view --repo "$CIRIS_VERIFY_REPO" --json tagName -q '.tagName' 2>/dev/null || echo "")
        if [ -z "$version" ]; then
            log_error "Cannot fetch CIRISVerify version. Is 'gh' CLI installed and authenticated?"
            return 1
        fi
    fi
    log_info "CIRISVerify version: $version"

    # Determine architecture (HA Yellow is aarch64, also support x86_64)
    local arch
    arch=$(ssh "${HA_USER}@${HA_HOST}" "uname -m" 2>/dev/null || echo "aarch64")
    local asset_name
    case "$arch" in
        aarch64|arm64)
            asset_name="ciris-verify-${version}-linux-arm64-musl.tar.gz"
            ;;
        x86_64)
            # Note: x86_64 musl may not be available; fall back to glibc if needed
            asset_name="ciris-verify-${version}-linux-arm64-musl.tar.gz"
            log_warn "x86_64 musl binary may not be available; using arm64-musl"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            return 1
            ;;
    esac

    log_step "Downloading CIRISVerify musl binary: $asset_name"

    # Create temp directory for download
    local tmp_dir
    tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" RETURN

    # Download using gh CLI
    if ! gh release download "$version" \
        --repo "$CIRIS_VERIFY_REPO" \
        --pattern "$asset_name" \
        --dir "$tmp_dir" 2>/dev/null; then
        log_error "Failed to download $asset_name from $CIRIS_VERIFY_REPO"
        return 1
    fi

    # Extract the tarball
    log_info "Extracting binary..."
    mkdir -p "$lib_dir"
    tar -xzf "${tmp_dir}/${asset_name}" -C "$tmp_dir"

    # Find and copy the .so file
    local so_file
    so_file=$(find "$tmp_dir" -name "libciris_verify*.so" -o -name "libciris_verify_ffi.so" | head -1)
    if [ -z "$so_file" ]; then
        log_error "No .so file found in tarball"
        return 1
    fi

    cp "$so_file" "$target_file"
    chmod 755 "$target_file"

    local size
    size=$(du -h "$target_file" | cut -f1)
    log_info "CIRISVerify musl binary installed: $target_file ($size)"

    return 0
}

# Download CIRISVerify musl binary unless skipped
if [ "$SKIP_VERIFY" = false ]; then
    log_step "Fetching CIRISVerify musl binary for Alpine Linux..."
    if command -v gh &>/dev/null; then
        if download_cirisverify_musl; then
            log_info "CIRISVerify musl binary ready"
        else
            log_warn "Failed to download CIRISVerify. Using existing binary if available."
        fi
    else
        log_warn "GitHub CLI (gh) not found. Skipping CIRISVerify download."
        log_info "Install with: brew install gh  OR  apt install gh"
        log_info "Or use --skip-verify to use existing binary"
    fi
else
    log_info "Skipping CIRISVerify download (--skip-verify)"
fi

# List of platform-specific files that would create manylinux wheel
PLATFORM_FILES=(
    "ciris_engine/desktop_app/CIRIS-linux-x64-2.0.0.jar"
    # Add other platform-specific files here as needed
)

# Move platform-specific files temporarily
MOVED_FILES=()
for file in "${PLATFORM_FILES[@]}"; do
    if [ -f "$file" ]; then
        log_warn "Found platform-specific file: $file"
        log_info "Moving to /tmp temporarily..."
        mv "$file" "/tmp/$(basename $file)"
        MOVED_FILES+=("$file")
    fi
done

# Clean previous builds
log_step "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

# Build pure Python wheel
log_step "Building pure Python wheel..."
python3 -m build --wheel

# Restore moved files
for file in "${MOVED_FILES[@]}"; do
    log_info "Restoring: $file"
    mv "/tmp/$(basename $file)" "$file"
done

# Find the built wheel
WHEEL_FILE=$(ls dist/ciris_agent-*-py3-none-any.whl 2>/dev/null | head -1)

if [ -z "$WHEEL_FILE" ]; then
    log_error "No pure Python wheel found!"
    log_error "The build may have created a platform-specific wheel instead."
    echo ""
    echo "Files in dist/:"
    ls -la dist/
    echo ""
    log_info "Check for platform-specific files in the source tree:"
    find . -name "*.so" -o -name "*.jar" -o -name "*.dll" 2>/dev/null | head -10
    exit 1
fi

log_info "Built wheel: $WHEEL_FILE"
WHEEL_SIZE=$(du -h "$WHEEL_FILE" | cut -f1)
log_info "Wheel size: $WHEEL_SIZE"

# Extract version from wheel filename
WHEEL_VERSION=$(basename "$WHEEL_FILE" | sed 's/ciris_agent-\([^-]*\)-.*/\1/')
log_info "Wheel version: $WHEEL_VERSION"

# Verify it's a pure Python wheel
if [[ "$WHEEL_FILE" != *"py3-none-any"* ]]; then
    log_error "Wheel is not pure Python (py3-none-any)."
    log_error "HA uses Alpine (musllinux) and cannot install manylinux wheels."
    exit 1
fi

# Deploy to HA
log_step "Deploying wheel to HA at $HA_HOST..."
ssh "${HA_USER}@${HA_HOST}" "mkdir -p ${ADDON_PATH}"

# Remove old wheels first
log_info "Removing old wheels..."
ssh "${HA_USER}@${HA_HOST}" "rm -f ${ADDON_PATH}/ciris_agent-*.whl"

# Copy new wheel
log_info "Copying wheel to HA..."
scp -q "$WHEEL_FILE" "${HA_USER}@${HA_HOST}:${ADDON_PATH}/"

# Verify deployment
DEPLOYED_WHEEL=$(ssh "${HA_USER}@${HA_HOST}" "ls ${ADDON_PATH}/ciris_agent-*.whl 2>/dev/null" || echo "")
if [ -z "$DEPLOYED_WHEEL" ]; then
    log_error "Wheel deployment failed!"
    exit 1
fi

echo ""
log_info "Wheel deployed successfully!"
echo ""
echo "  Location: ${HA_HOST}:${ADDON_PATH}/$(basename $WHEEL_FILE)"
echo "  Version:  $WHEEL_VERSION"
echo "  Size:     $WHEEL_SIZE"
echo ""

# =============================================================================
# WASM Web App Build (KMP 2.x)
# =============================================================================
if [ "$BUILD_WEB" = true ]; then
    log_step "Building WASM web app from KMP 2.x upstream..."

    MOBILE_DIR="${CIRIS_AGENT_DIR}/mobile"
    WEBAPP_DIR="${MOBILE_DIR}/webApp"
    WASM_OUTPUT="${WEBAPP_DIR}/build/dist/wasmJs/productionExecutable"

    # Check if webApp module exists
    if [ ! -d "$WEBAPP_DIR" ]; then
        log_warn "webApp module not found at $WEBAPP_DIR"
        log_info "Run migration first: cd ${MOBILE_DIR} && ./scripts/migrate-to-kmp2.sh"
        log_info "Skipping web build..."
    else
        cd "$MOBILE_DIR"

        # Build production WASM
        log_info "Building wasmJsBrowserDistribution (this may take a few minutes)..."
        if ./gradlew :webApp:wasmJsBrowserDistribution --quiet 2>&1; then
            log_ok "WASM build complete"

            # Check output
            if [ -d "$WASM_OUTPUT" ]; then
                WASM_SIZE=$(du -sh "$WASM_OUTPUT" | cut -f1)
                log_info "WASM output: $WASM_OUTPUT ($WASM_SIZE)"

                # Deploy to HA addon www directory
                log_step "Deploying WASM app to HA..."
                WWW_PATH="${ADDON_PATH}/www"

                ssh "${HA_USER}@${HA_HOST}" "rm -rf ${WWW_PATH} && mkdir -p ${WWW_PATH}"
                scp -rq "${WASM_OUTPUT}/"* "${HA_USER}@${HA_HOST}:${WWW_PATH}/"

                log_ok "WASM app deployed to ${HA_HOST}:${WWW_PATH}/"
            else
                log_warn "WASM output directory not found: $WASM_OUTPUT"
            fi
        else
            log_error "WASM build failed"
            log_info "Check build errors with: cd ${MOBILE_DIR} && ./gradlew :webApp:wasmJsBrowserDistribution"
        fi

        cd "$CIRIS_AGENT_DIR"
    fi
fi

# =============================================================================
# Sync CIRISHome mobile-web from upstream (if needed)
# =============================================================================
sync_mobile_web() {
    local src_mobile="${CIRIS_AGENT_DIR}/mobile"
    local dst_mobile="${CIRIS_HOME_DIR}/mobile-web"

    log_step "Syncing mobile-web from KMP 2.x upstream..."

    # Check if upstream has wasmJs target
    if grep -q "wasmJs" "${src_mobile}/shared/build.gradle.kts" 2>/dev/null; then
        log_info "Upstream has native wasmJs - syncing commonMain only"

        # Sync only commonMain (preserve wasmJsMain implementations)
        rsync -av --delete \
            --exclude='build/' \
            --exclude='.gradle/' \
            "${src_mobile}/shared/src/commonMain/" \
            "${dst_mobile}/shared/src/commonMain/" > /dev/null 2>&1 || true

        log_ok "commonMain synced"
    else
        log_info "Upstream doesn't have wasmJs yet - using full rebuild"
        log_info "Run: ./scripts/rebuild-mobile-web.sh to convert"
    fi
}

# Check if mobile-web sync is needed
if [ -d "${CIRIS_HOME_DIR}/mobile-web" ]; then
    # Check if upstream is newer than local
    UPSTREAM_MOD=$(stat -c %Y "${CIRIS_AGENT_DIR}/mobile/shared/src/commonMain" 2>/dev/null || echo "0")
    LOCAL_MOD=$(stat -c %Y "${CIRIS_HOME_DIR}/mobile-web/shared/src/commonMain" 2>/dev/null || echo "0")

    if [ "$UPSTREAM_MOD" -gt "$LOCAL_MOD" ]; then
        log_info "Upstream mobile code is newer - consider syncing"
        log_info "Run: ./scripts/rebuild-mobile-web.sh"
    fi
fi

log_info "Next steps:"
echo "  1. Deploy addon:  ./scripts/deploy-addon.sh $HA_HOST --fresh"
echo "  2. Or rebuild:    ssh ${HA_USER}@${HA_HOST} 'ha addons rebuild local_ciris_agent'"
echo ""
log_warn "Note: Use --fresh flag with deploy-addon.sh to ensure Docker picks up the new wheel"
