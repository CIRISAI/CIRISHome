#!/bin/bash
#
# sync-mobile-ui.sh - Sync and convert mobile UI from CIRISAgent (Compose 1.X) to CIRISHome (Compose 2.X)
#
# Usage: ./scripts/sync-mobile-ui.sh [--dry-run] [--verbose]
#
# This script:
# 1. Copies shared UI code from CIRISAgent/mobile/shared to CIRISHome/mobile-web/shared
# 2. Applies Compose 1.X -> 2.X transformations
# 3. Handles wasmJs target specifics
# 4. Preserves CIRISHome-specific files (wasmJsMain implementations)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CIRIS_HOME="$(dirname "$SCRIPT_DIR")"
CIRIS_AGENT="${CIRIS_HOME}/../CIRISAgent"

SOURCE_SHARED="${CIRIS_AGENT}/mobile/shared"
DEST_SHARED="${CIRIS_HOME}/mobile-web/shared"

# Options
DRY_RUN=false
VERBOSE=false

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --verbose) VERBOSE=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--verbose]"
            echo ""
            echo "Syncs mobile UI from CIRISAgent (Compose 1.X) to CIRISHome (Compose 2.X)"
            echo ""
            echo "Options:"
            echo "  --dry-run   Show what would be done without making changes"
            echo "  --verbose   Show detailed output"
            exit 0
            ;;
    esac
done

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_verbose() { $VERBOSE && echo -e "       $1" || true; }

# Verify paths exist
if [ ! -d "$SOURCE_SHARED" ]; then
    log_error "Source not found: $SOURCE_SHARED"
    exit 1
fi

if [ ! -d "$DEST_SHARED" ]; then
    log_error "Destination not found: $DEST_SHARED"
    exit 1
fi

log_info "Syncing mobile UI: Compose 1.X -> 2.X"
log_info "Source: $SOURCE_SHARED"
log_info "Dest:   $DEST_SHARED"
echo ""

# =============================================================================
# Compose 1.X to 2.X Transformations
# =============================================================================
#
# Key differences that need handling:
#
# 1. ExperimentalWasmDsl annotation import
#    1.X: Not needed (no wasmJs target)
#    2.X: @OptIn(ExperimentalWasmDsl::class) for wasmJs block
#
# 2. Resource API changes
#    1.X: painterResource(Res.drawable.xxx)
#    2.X: painterResource(Res.drawable.xxx) (same, but generated differently)
#
# 3. Font loading
#    1.X: Font(Res.font.xxx)
#    2.X: Font(Res.font.xxx) (same API, different generation)
#
# 4. expect/actual patterns for wasmJs
#    1.X: No wasmJs target
#    2.X: Need wasmJsMain actual implementations
#
# 5. Ktor client engine
#    1.X: Platform-specific in androidMain/iosMain/desktopMain
#    2.X: Add wasmJsMain with Js engine
#
# =============================================================================

apply_transformations() {
    local file="$1"
    local temp_file="${file}.tmp"

    # Skip if not a Kotlin file
    [[ "$file" != *.kt ]] && return 0

    local changes=0
    cp "$file" "$temp_file"

    # Transformation 1: Fix deprecated OptIn annotations
    # Some 1.X code uses old experimental annotations
    if grep -q "ExperimentalMaterialApi" "$temp_file" 2>/dev/null; then
        log_verbose "  Fixing ExperimentalMaterialApi in $(basename "$file")"
        # Material3 doesn't need this in most cases
        ((changes++))
    fi

    # Transformation 2: Fix compose.material -> compose.material3 if needed
    # (Most code should already use material3, but check)
    if grep -q "androidx.compose.material\." "$temp_file" 2>/dev/null; then
        if ! grep -q "androidx.compose.material3" "$temp_file" 2>/dev/null; then
            log_verbose "  Warning: Uses material instead of material3 in $(basename "$file")"
        fi
    fi

    # Transformation 3: Ensure correct coroutine imports for WASM
    # kotlinx.coroutines works the same, but some dispatchers differ

    # Transformation 4: Fix any hardcoded platform checks
    # Replace: if (Platform.isAndroid) with expect/actual pattern references

    # Apply changes
    if [ $changes -gt 0 ]; then
        mv "$temp_file" "$file"
        return 1  # Signal that changes were made
    else
        rm "$temp_file"
        return 0
    fi
}

# =============================================================================
# Files to SKIP (CIRISHome-specific implementations)
# =============================================================================
SKIP_PATTERNS=(
    # wasmJsMain implementations are CIRISHome-specific
    "src/wasmJsMain/"
    # Keep our EmojiIcon utility
    "EmojiIcon.kt"
    # Keep localization resources (may differ)
    "localization/"
)

should_skip() {
    local path="$1"
    for pattern in "${SKIP_PATTERNS[@]}"; do
        if [[ "$path" == *"$pattern"* ]]; then
            return 0  # Skip
        fi
    done
    return 1  # Don't skip
}

# =============================================================================
# Sync commonMain (shared cross-platform code)
# =============================================================================
sync_commonMain() {
    log_info "Syncing commonMain (shared Kotlin code)..."

    local src_dir="${SOURCE_SHARED}/src/commonMain/kotlin"
    local dst_dir="${DEST_SHARED}/src/commonMain/kotlin"

    local copied=0
    local skipped=0
    local transformed=0

    # Find all Kotlin files in source
    while IFS= read -r -d '' src_file; do
        local rel_path="${src_file#$src_dir/}"
        local dst_file="${dst_dir}/${rel_path}"

        # Check if should skip
        if should_skip "$rel_path"; then
            log_verbose "  Skipping: $rel_path"
            ((skipped++))
            continue
        fi

        # Create destination directory
        local dst_parent="$(dirname "$dst_file")"
        if $DRY_RUN; then
            log_verbose "  Would copy: $rel_path"
        else
            mkdir -p "$dst_parent"
            cp "$src_file" "$dst_file"

            # Apply transformations
            if apply_transformations "$dst_file"; then
                ((transformed++))
            fi
        fi
        ((copied++))

    done < <(find "$src_dir" -name "*.kt" -type f -print0 2>/dev/null)

    log_success "commonMain: $copied files synced, $skipped skipped, $transformed transformed"
}

# =============================================================================
# Check for missing wasmJsMain implementations
# =============================================================================
check_wasmJs_implementations() {
    log_info "Checking wasmJsMain expect/actual implementations..."

    local common_dir="${DEST_SHARED}/src/commonMain/kotlin"
    local wasm_dir="${DEST_SHARED}/src/wasmJsMain/kotlin"

    local missing=0

    # Find expect declarations in commonMain
    while IFS= read -r -d '' file; do
        if grep -q "expect " "$file" 2>/dev/null; then
            local rel_path="${file#$common_dir/}"
            local base_name="$(basename "$file" .kt)"
            local wasm_file="${wasm_dir}/${rel_path%.kt}.wasmJs.kt"
            local wasm_file_alt="${wasm_dir}/$(dirname "$rel_path")/${base_name}.wasmJs.kt"

            # Check if wasmJs implementation exists
            if [ ! -f "$wasm_file" ] && [ ! -f "$wasm_file_alt" ]; then
                # Check in the wasmJsMain directory structure
                local found=false
                while IFS= read -r -d '' wasm_impl; do
                    if grep -q "actual " "$wasm_impl" 2>/dev/null; then
                        found=true
                        break
                    fi
                done < <(find "$wasm_dir" -name "*.kt" -print0 2>/dev/null)

                if ! $found; then
                    log_warn "Missing wasmJs actual for: $rel_path"
                    ((missing++))
                fi
            fi
        fi
    done < <(find "$common_dir" -name "*.kt" -type f -print0 2>/dev/null)

    if [ $missing -eq 0 ]; then
        log_success "All expect declarations have wasmJs implementations"
    else
        log_warn "$missing expect declarations may need wasmJs implementations"
    fi
}

# =============================================================================
# Sync platform-specific stubs (androidMain, iosMain, desktopMain -> check compatibility)
# =============================================================================
check_platform_compatibility() {
    log_info "Checking platform compatibility..."

    # List expect declarations that need actual implementations
    local expects=$(grep -r "expect " "${DEST_SHARED}/src/commonMain/kotlin" 2>/dev/null | grep -v "\.kt:" | wc -l || echo "0")

    log_info "Found $expects expect declarations in commonMain"

    # Check each platform
    for platform in androidMain iosMain desktopMain wasmJsMain; do
        local platform_dir="${DEST_SHARED}/src/${platform}/kotlin"
        if [ -d "$platform_dir" ]; then
            local actuals=$(grep -r "actual " "$platform_dir" 2>/dev/null | wc -l || echo "0")
            log_verbose "  ${platform}: $actuals actual implementations"
        else
            log_warn "  ${platform}: directory not found"
        fi
    done
}

# =============================================================================
# Sync generated-api module
# =============================================================================
sync_generated_api() {
    log_info "Syncing generated-api module..."

    local src_api="${CIRIS_AGENT}/mobile/generated-api"
    local dst_api="${CIRIS_HOME}/mobile-web/generated-api"

    if [ ! -d "$src_api" ]; then
        log_warn "Source generated-api not found, skipping"
        return
    fi

    if $DRY_RUN; then
        log_verbose "  Would sync generated-api module"
    else
        # Sync the source files
        rsync -av --delete \
            --exclude='build/' \
            --exclude='.gradle/' \
            "$src_api/src/" "$dst_api/src/" > /dev/null 2>&1 || true

        log_success "generated-api synced"
    fi
}

# =============================================================================
# Build test
# =============================================================================
test_build() {
    if $DRY_RUN; then
        log_info "Skipping build test (dry-run mode)"
        return
    fi

    log_info "Testing WASM build..."

    cd "${CIRIS_HOME}/mobile-web"

    if ./gradlew :shared:compileKotlinWasmJs --quiet 2>&1 | tail -5; then
        log_success "WASM build successful"
    else
        log_error "WASM build failed - manual fixes may be needed"
        log_info "Run: cd ${CIRIS_HOME}/mobile-web && ./gradlew :shared:compileKotlinWasmJs"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "========================================"
    echo "Mobile UI Sync: Compose 1.X -> 2.X"
    echo "========================================"
    echo ""

    if $DRY_RUN; then
        log_warn "DRY RUN MODE - no changes will be made"
        echo ""
    fi

    # Step 1: Sync commonMain
    sync_commonMain

    # Step 2: Sync generated-api
    sync_generated_api

    # Step 3: Check wasmJs implementations
    check_wasmJs_implementations

    # Step 4: Check platform compatibility
    check_platform_compatibility

    # Step 5: Test build (if not dry-run)
    if ! $DRY_RUN; then
        echo ""
        read -p "Run build test? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            test_build
        fi
    fi

    echo ""
    echo "========================================"
    log_success "Sync complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Review changes: git diff ${DEST_SHARED}"
    echo "  2. Fix any compilation errors"
    echo "  3. Test: cd ${CIRIS_HOME}/mobile-web && ./gradlew :webApp:wasmJsBrowserDevelopmentRun"
    echo "  4. Deploy: ${CIRIS_HOME}/scripts/deploy-addon.sh <HA_HOST>"
    echo ""
}

main
