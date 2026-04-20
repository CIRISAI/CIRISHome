# QuickSetup HA Addon Mode Fix - Lessons Learned

## Problem Summary

The QuickSetup screen in Home Assistant addon mode incorrectly showed:

- Green "CIRIS AI Services Active" banner (should be blue "Bring Your Own Key")
- "Google sign-in is required" error message
- Disabled "Next" button

## Root Cause

**Inconsistent BYOK mode detection logic between Welcome and QuickSetup screens.**

### The Two Key Screens

1. **WelcomeStep** (works correctly):

   ```kotlin
   val isCirisMode = state.setupMode == SetupMode.CIRIS_PROXY
   // Blue banner shown when !isCirisMode (anything NOT CIRIS_PROXY = BYOK)
   ```

2. **QuickSetupStep** (was broken):
   ```kotlin
   // Original (broken) - required explicit BYOK mode
   val isBYOKMode = state.setupMode == SetupMode.BYOK || state.isHAAddonMode
   ```

### Why It Failed

In HA addon mode:

1. `setHAAddonMode(true)` is called, which sets `setupMode = SetupMode.BYOK`
2. BUT multiple places in the QuickSetupStep code checked `isBYOKMode` using different logic
3. Some places used `state.setupMode == SetupMode.CIRIS_PROXY` to determine CIRIS mode
4. Others used different variables, creating inconsistency

### The Multi-Location Bug

The QuickSetupStep had BYOK mode checks in multiple locations:

- **Header badge** (~line 2617): Controls badge color/text
- **Mode info card** (~line 2660-2675): Shows "CIRIS AI Services Active" vs "Bring Your Own Key"
- **LLM config section** (~line 2862-2877): Shows required vs optional

Not all locations were using the same `isBYOKMode` variable!

## Fix Applied

1. **Unified logic at top of QuickSetupStep**:

   ```kotlin
   val isCirisMode = state.setupMode == SetupMode.CIRIS_PROXY
   val isBYOKMode = !isCirisMode
   val effectiveBYOKMode = true // FORCE TRUE FOR TESTING (remove after verified)
   ```

2. **Updated ALL locations to use `effectiveBYOKMode`**:
   - Header badge
   - Mode info card (icon, text, description)
   - LLM config section subtitle
   - LLM config description text

3. **Added console debug logging**:
   ```kotlin
   println("[QuickSetup] ========== BUILD 2026-04-20-B ==========")
   println("[QuickSetup] state.setupMode=${state.setupMode}")
   println("[QuickSetup] state.isHAAddonMode=${state.isHAAddonMode}")
   println("[QuickSetup] isCirisMode=$isCirisMode")
   println("[QuickSetup] isBYOKMode=$isBYOKMode")
   println("[QuickSetup] effectiveBYOKMode=$effectiveBYOKMode (FORCED TRUE)")
   ```

## Files Modified

- `mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SetupScreen.kt`
  - QuickSetupStep function - unified BYOK mode logic
  - Added debug logging

- `mobile-web/webApp/src/wasmJsMain/kotlin/ai/ciris/web/Main.kt`
  - Updated build marker for cache-busting

## Build & Deploy Process

### WASM Build Commands

```bash
# Clean rebuild (production)
cd /home/emoore/CIRISHome/mobile-web
./gradlew --stop
rm -rf .gradle build
./gradlew :webApp:wasmJsBrowserDistribution --rerun-tasks --no-build-cache

# Development build
./gradlew :webApp:wasmJsBrowserDevelopmentExecutableDistribution --rerun-tasks
```

### Deploy to HA Addon

```bash
cd /home/emoore/CIRISHome
./scripts/deploy-addon.sh 192.168.50.243
```

### Build Output Locations

- **Production**: `mobile-web/webApp/build/dist/wasmJs/productionExecutable/`
- **Development**: `mobile-web/webApp/build/dist/wasmJs/developmentExecutable/`
- **Addon www**: `ciris-agent/www/` (copied during deploy)

## Key Debugging Insights

### 1. WASM Build Caching

Gradle caches aggressively. Always use `--rerun-tasks --no-build-cache` when debugging:

```bash
./gradlew :webApp:wasmJsBrowserDistribution --rerun-tasks --no-build-cache
```

### 2. Verify Build Timestamps

Check timestamps match your edit times:

```bash
ls -la mobile-web/webApp/build/dist/wasmJs/productionExecutable/
```

### 3. Console Debug Logging

Add console.log output via `println()` in Kotlin/WASM:

```kotlin
println("[DEBUG] Variable value: $myVar")
```

View in browser DevTools console.

### 4. Multiple Code Locations

When fixing UI issues, search for ALL usages of the variable:

```bash
grep -n "isBYOKMode" mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SetupScreen.kt
```

### 5. docker-compose.web.yml

The local docker compose uses `developmentExecutable`:

```yaml
volumes:
  - ./mobile-web/webApp/build/dist/wasmJs/developmentExecutable:/usr/share/nginx/html:ro
```

But `deploy-addon.sh` uses `productionExecutable` for HA addon deployment.

## State Flow in HA Addon Mode

```
Main.kt (WASM entry)
  ↓ detectHAMode() returns true
  ↓ passes isHAAddonMode=true to CIRISApp

CIRISApp.kt
  ↓ Creates SetupViewModel
  ↓ Calls setupViewModel.setHAAddonMode(true)

SetupViewModel.kt
  ↓ setHAAddonMode(true) sets:
  ↓   - state.isHAAddonMode = true
  ↓   - state.setupMode = SetupMode.BYOK

SetupScreen.kt (QuickSetupStep)
  ↓ Reads state via collectAsState()
  ↓ Should show BYOK mode UI
```

## Fix Verified - 2026-04-20

**Status: COMPLETE**

The fix has been verified working in HA addon mode:

- Welcome screen shows blue "Bring Your Own Key" badge
- QuickSetup screen shows blue "Bring Your Own Key Mode" card
- Validation shows "API key is required" (correct for BYOK) instead of "Google sign-in is required"
- Console logs confirm: `state.setupMode=BYOK`, `state.isHAAddonMode=true`, `isBYOKMode=true`

### Cleanup Done

1. Removed hardcoded `effectiveBYOKMode = true`
2. Set `effectiveBYOKMode = isBYOKMode` for backward compatibility
3. Removed debug console logging

### Remaining

- Sync fix to upstream CIRISAgent repo (mobile-web is a conversion of CIRISAgent/client)

## Production vs Development Build

**CRITICAL**: The production webpack build causes WASM runtime errors:

```
WebAssembly.instantiate(): Import #1 "js_code" "kotlin.wasm.internal.throwJsError": function import requires a callable
```

**Workaround**: Deploy the development build instead of production:

1. Build: `./gradlew :webApp:wasmJsBrowserDevelopmentExecutableDistribution`
2. Copy dev files to www: `scp -r mobile-web/webApp/build/dist/wasmJs/developmentExecutable/* root@HA_HOST:/addons/ciris_agent/www/`
3. Rebuild addon: `ssh root@HA_HOST 'ha addons rebuild local_ciris_agent'`

The development build is larger (3.7MB JS vs 552KB) but works correctly.

## Localization Keys

- `setup.quick_byok_active` = "Bring Your Own Key"
- `setup.quick_ciris_active` = "CIRIS AI Services Active"
- `setup.quick_byok_desc` = BYOK description
- `setup.quick_ciris_desc` = CIRIS proxy description
