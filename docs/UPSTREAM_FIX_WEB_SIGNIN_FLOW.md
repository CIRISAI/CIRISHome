# Upstream Fix: Web Sign-In Flow (WASM) - Phone vs Desktop Behavior

**Target Repository**: CIRISAgent/client/shared
**Status**: Bug Report for Upstream
**Issue**: "Web Sign-In not available" error on phone HA app when desktop browser works

## Problem Summary

When accessing CIRIS addon from phone's HA companion app:
- Setup completed successfully from desktop browser
- Phone's HA app shows "Web Sign-In not available" error
- Backend logs show authentication IS working: `[HA_INGRESS_AUTH] Authenticated user: Eric Moore`

## CRITICAL BUG FOUND

**Location**: `CIRISApp.kt` line 742

The non-first-run code path has **NO check for HA addon mode**:

```kotlin
} else if (isFirstRun == true) {
    if (isHAAddonMode) {
        // First run + HA mode - goes to Setup  <-- THIS WORKS
        currentScreen = Screen.Setup
    } else {
        currentScreen = Screen.Login
    }
} else {
    // NOT first run - NO isHAAddonMode check!  <-- BUG HERE
    // Always tries to load stored token from localStorage
    val tokenResult = secureStorage.getAccessToken()
    ...
    if (storedToken == null) {
        currentScreen = Screen.Login  // <-- Phone ends up here
    }
}
```

**Why phone fails:**
1. Desktop completed setup → stored token in desktop's localStorage
2. Phone's HA app webview = **different localStorage** (no stored token)
3. `secureStorage.getAccessToken()` returns null
4. Code goes to `Screen.Login`
5. OAuth button doesn't work because `googleSignInCallback == null` for WASM

**But backend IS authenticating!** Every request shows:
```
[HA_INGRESS_AUTH] Authenticated user: Eric Moore... (id: 4dfc6a84...)
```

The ingress token in HTTP headers works perfectly - the frontend just doesn't use it!

## Root Cause Analysis

### The WASM Build Has No Native OAuth

In `webApp/src/wasmJsMain/kotlin/ai/ciris/web/Main.kt:79`:
```kotlin
CIRISApp(
    accessToken = accessToken ?: "",
    baseUrl = baseUrl,
    googleSignInCallback = null  // <-- ALWAYS null for WASM
)
```

**This is intentional** - WASM/browser cannot do native Google Sign-In like iOS/Android.

### The Login Screen OAuth Button Check

In `shared/src/commonMain/kotlin/ai/ciris/mobile/shared/CIRISApp.kt:1022-1167`:
```kotlin
onGoogleSignIn = {
    if (googleSignInCallback != null) {
        // Native sign-in available - call it
        googleSignInCallback.onGoogleSignInRequested { result -> ... }
    } else {
        // No callback provided - show error
        platformLog(TAG, "[ERROR] googleSignInCallback is NULL")
        loginErrorMessage = "${getOAuthProviderName()} Sign-In not available"
    }
}
```

### Why Desktop Works But Phone Doesn't

**Desktop browser (works):**
1. Access HA via `http://homeassistant.local:8123/`
2. Click CIRIS in sidebar
3. HA loads addon via ingress: `/api/hassio_ingress/TOKEN/`
4. `detectHAMode()` returns `true` (path contains `/api/hassio_ingress/`)
5. `isHAAddonMode = true` → Skips Login → Goes to Setup (BYOK)
6. No OAuth needed - HA handles auth via `SUPERVISOR_TOKEN`

**Phone browser (fails):**
Several scenarios could cause failure:

#### Scenario A: Different URL Pattern on Phone
If the phone accesses HA differently (e.g., Nabu Casa cloud URL, companion app webview, or a reverse proxy), the path might NOT contain:
- `/api/hassio_ingress/`
- `/hassio/ingress/`

```kotlin
// From Main.kt:94-98
private fun detectHAMode(): Boolean {
    val path = window.location.pathname
    return path.contains("/api/hassio_ingress/") ||
           path.contains("/hassio/ingress/") ||
           window.parent != window // Embedded in iframe
}
```

#### Scenario B: Iframe Detection Fails
If the phone browser's webview handles iframes differently, `window.parent != window` might return `false` even when embedded.

#### Scenario C: Not First Run
If `isFirstRun == false` (user already exists), the code path at line 742-744 tries to authenticate:
```kotlin
} else {
    // Not first run - try to load stored token
    platformLog(TAG, "[INFO] Not first run, attempting to load token")
    ...
}
```
If token validation fails, user ends up on Login screen where OAuth doesn't work.

#### Scenario D: localStorage Inconsistency
The `ciris_ha_addon_mode` flag is stored in localStorage:
```kotlin
if (isHAAddon) {
    localStorage.setItem("ciris_ha_addon_mode", "true")
}
```
But subsequent checks use `isHAAddonMode()` from Platform.wasmJs.kt:
```kotlin
actual fun isHAAddonMode(): Boolean {
    return localStorage.getItem("ciris_ha_addon_mode") == "true"
}
```
If localStorage is cleared or the URL detection fails on phone, this flag won't be set.

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    WASM App Initialization                       │
├─────────────────────────────────────────────────────────────────┤
│  Main.kt                                                         │
│  ├─ detectHAMode()                                               │
│  │   ├─ pathname.contains("/api/hassio_ingress/") → HA mode     │
│  │   ├─ pathname.contains("/hassio/ingress/") → HA mode         │
│  │   └─ window.parent != window → HA mode (iframe)              │
│  │                                                               │
│  └─ CIRISApp(                                                    │
│       googleSignInCallback = null,  // ALWAYS null for WASM      │
│       isHAAddonMode = detectHAMode()                             │
│     )                                                            │
└─────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CIRISApp.kt Startup                           │
├─────────────────────────────────────────────────────────────────┤
│  if (isHAAddonMode) {                                            │
│      setupViewModel.setHAAddonMode(true)  // Sets BYOK mode     │
│  }                                                               │
│                                                                  │
│  LaunchedEffect: checkFirstRunStatus()                           │
│  ├─ isFirstRun == true && isHAAddonMode == true                  │
│  │   └─→ Screen.Setup (BYOK) ✓ Works!                            │
│  │                                                               │
│  ├─ isFirstRun == true && isHAAddonMode == false                 │
│  │   └─→ Screen.Login ✗ OAuth fails!                             │
│  │                                                               │
│  └─ isFirstRun == false                                          │
│      └─→ Try token validation → May fail → Screen.Login ✗       │
└─────────────────────────────────────────────────────────────────┘
```

## Debug Steps for Users

Add these console logs to diagnose the issue:

1. **Check detection values** - Look in browser console for:
   ```
   [CIRIS-Web] isHAAddon=true/false
   [CIRIS-Web] pathname=/api/hassio_ingress/TOKEN/
   ```

2. **Check CIRISApp startup** - Look for:
   ```
   [CIRISApp] [INFO] First run in HA Addon mode, skipping login
   ```
   vs
   ```
   [CIRISApp] [INFO] First run detected, navigating to Login
   ```

3. **Verify localStorage** - In browser DevTools:
   ```javascript
   localStorage.getItem("ciris_ha_addon_mode")  // Should be "true"
   ```

## THE FIX (Required)

**In `CIRISApp.kt` around line 742, add HA addon mode check to non-first-run branch:**

```kotlin
} else {
    // Not first run
    if (isHAAddonMode) {
        // HA addon mode - no stored token needed!
        // Ingress handles auth via HTTP headers automatically
        platformLog(TAG, "[INFO] Not first run in HA Addon mode - using ingress auth directly")
        interactViewModel.startPolling()
        currentScreen = Screen.Interact
    } else {
        // Normal mode - try to load stored token
        platformLog(TAG, "[INFO] Not first run, attempting to load and validate stored token")
        ...existing token loading code...
    }
}
```

**Why this works:**
1. HA ingress injects `Ingress-Token` header on all requests
2. Backend extracts user from this header (already working - see logs)
3. No client-side token storage needed in HA mode
4. Frontend just goes directly to chat screen

## Additional Fixes

### Fix 1: Add More URL Patterns (Quick Fix)

Update `detectHAMode()` to recognize more HA URL patterns:

```kotlin
private fun detectHAMode(): Boolean {
    val path = window.location.pathname
    val hostname = window.location.hostname
    val href = window.location.href

    return path.contains("/api/hassio_ingress/") ||
           path.contains("/hassio/ingress/") ||
           // Nabu Casa cloud URLs
           hostname.endsWith(".ui.nabu.casa") ||
           hostname.contains(".nabucasa.") ||
           // HA Android/iOS companion app webview
           href.contains("home-assistant://") ||
           // Check for HA-specific URL patterns
           path.startsWith("/lovelace") ||
           // Iframe check (existing)
           window.parent != window
}
```

### Fix 2: Add Web OAuth Flow (Better Fix)

Implement a redirect-based OAuth flow for WASM:

```kotlin
// In Main.kt - create a web-compatible OAuth callback
val webOAuthCallback = object : NativeSignInCallback {
    override fun onGoogleSignInRequested(callback: (NativeSignInResult) -> Unit) {
        // Redirect to Google OAuth
        val clientId = "YOUR_WEB_CLIENT_ID"
        val redirectUri = encodeURIComponent(window.location.href)
        val authUrl = "https://accounts.google.com/o/oauth2/v2/auth?" +
            "client_id=$clientId&" +
            "redirect_uri=$redirectUri&" +
            "response_type=token&" +
            "scope=email%20profile"
        window.location.href = authUrl
    }

    override fun onSilentSignInRequested(callback: (NativeSignInResult) -> Unit) {
        callback(NativeSignInResult.Cancelled)  // No silent sign-in on web
    }
}

// On page load, check for OAuth callback
val hashParams = window.location.hash  // #access_token=...
if (hashParams.contains("access_token")) {
    // Handle OAuth callback
}
```

### Fix 3: Show Helpful Error Message (UX Fix)

When OAuth isn't available, show a helpful message instead of error:

```kotlin
// In CIRISApp.kt login screen
if (googleSignInCallback == null) {
    // Show BYOK setup option instead of OAuth button
    Button(
        onClick = {
            setupViewModel.setGoogleAuthState(isAuth = false, ...)
            currentScreen = Screen.Setup
        }
    ) {
        Text("Set Up with Your Own API Key")
    }

    Text(
        text = "OAuth sign-in is not available in the web browser. " +
               "Please use the mobile app or set up with your own API key.",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}
```

### Fix 4: Force BYOK Mode for WASM (Simplest Fix)

Since WASM can never do native OAuth, always use BYOK:

```kotlin
// In Main.kt
val isHAAddon = detectHAMode()

// WASM can't do native OAuth - always use BYOK mode
// The web app is primarily for HA addon anyway
val forceByokMode = true

CIRISApp(
    ...
    isHAAddonMode = isHAAddon || forceByokMode
)
```

## Files to Modify

1. **`webApp/src/wasmJsMain/kotlin/ai/ciris/web/Main.kt`**
   - Improve `detectHAMode()` to handle more URL patterns
   - Or implement web OAuth redirect flow

2. **`shared/src/commonMain/kotlin/ai/ciris/mobile/shared/CIRISApp.kt`**
   - Login screen: Show BYOK option when OAuth unavailable
   - Add better error messaging for web users

3. **`shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/LoginScreen.kt`**
   - Add "Continue without sign-in" button that goes to BYOK setup

## Testing

1. **Desktop browser via HA ingress** - Should work (baseline)
2. **Phone browser via HA ingress** - Should work after fix
3. **Phone browser via Nabu Casa** - Should work after fix
4. **Direct web access (not HA)** - Should show BYOK option, not OAuth error

## Summary

The root cause is that:
1. **WASM cannot do native OAuth** (`googleSignInCallback = null`)
2. **HA addon mode detection may fail on phone** (different URL patterns)
3. **When both conditions hit**, user sees Login screen with non-functional OAuth button

The recommended fix is either:
- **Quick**: Improve `detectHAMode()` URL pattern matching
- **Better**: Show BYOK option when OAuth unavailable
- **Simplest**: Force BYOK mode for all WASM builds (web is primarily for HA addon)
