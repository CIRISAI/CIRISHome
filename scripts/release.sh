#!/bin/bash
# Release script for CIRIS Agent addon
# Usage: ./scripts/release.sh 5.1.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$REPO_ROOT/ciris-agent/config.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

error() { echo -e "${RED}ERROR: $1${NC}" >&2; exit 1; }
info() { echo -e "${GREEN}$1${NC}"; }
warn() { echo -e "${YELLOW}$1${NC}"; }

# Check arguments
NEW_VERSION="${1:-}"
if [ -z "$NEW_VERSION" ]; then
    CURRENT_VERSION=$(grep '^version:' "$CONFIG_FILE" | sed 's/version: *"\([^"]*\)"/\1/')
    echo "Current version: $CURRENT_VERSION"
    echo ""
    echo "Usage: $0 <new-version>"
    echo "Example: $0 5.1.0"
    exit 1
fi

# Validate version format (semver)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    error "Invalid version format. Use semver (e.g., 5.1.0)"
fi

# Check for uncommitted changes
if ! git diff --quiet HEAD; then
    error "Uncommitted changes detected. Please commit or stash them first."
fi

# Get current version
CURRENT_VERSION=$(grep '^version:' "$CONFIG_FILE" | sed 's/version: *"\([^"]*\)"/\1/')
info "Upgrading from v$CURRENT_VERSION to v$NEW_VERSION"

# Update version in config.yaml
sed -i "s/^version: .*/version: \"$NEW_VERSION\"/" "$CONFIG_FILE"

# Verify the change
UPDATED_VERSION=$(grep '^version:' "$CONFIG_FILE" | sed 's/version: *"\([^"]*\)"/\1/')
if [ "$UPDATED_VERSION" != "$NEW_VERSION" ]; then
    error "Failed to update version in config.yaml"
fi

# Commit and tag
info "Committing version bump..."
git add "$CONFIG_FILE"
git commit -m "Release v$NEW_VERSION

- Bump version from $CURRENT_VERSION to $NEW_VERSION
- Users can refresh in HA to get the update"

info "Creating tag v$NEW_VERSION..."
git tag -a "v$NEW_VERSION" -m "Release v$NEW_VERSION"

# Push
info "Pushing to origin..."
git push origin main --tags

echo ""
info "=========================================="
info " Released v$NEW_VERSION successfully!"
info "=========================================="
echo ""
info "Users can now:"
info "  1. Go to Settings > Add-ons > CIRIS Agent"
info "  2. Click 'Check for updates' or wait for auto-refresh"
info "  3. Click 'Update' when available"
