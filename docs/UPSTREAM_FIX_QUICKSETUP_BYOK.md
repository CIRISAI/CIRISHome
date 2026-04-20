# Upstream Fix: QuickSetup BYOK Mode Detection

**Target Repository**: CIRISAgent/client/shared
**File**: `src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SetupScreen.kt`

## Problem

In HA addon mode (and any non-OAuth mode), the QuickSetup screen incorrectly shows:

- Green "CIRIS AI Services Active" banner instead of blue "Bring Your Own Key"
- "Google sign-in is required" validation error
- Disabled "Next" button

## Root Cause

**Inconsistent BYOK mode detection between WelcomeStep and QuickSetupStep.**

### WelcomeStep (correct):

```kotlin
val isCirisMode = state.setupMode == SetupMode.CIRIS_PROXY
// Uses !isCirisMode to determine BYOK mode - anything NOT CIRIS_PROXY = BYOK
```

### QuickSetupStep (was broken):

The code had BYOK mode checks scattered across multiple locations using different variables:

- Header badge
- Mode info card
- LLM config section

Some used `isBYOKMode`, others used different logic, creating inconsistency.

## Fix

### 1. Unify BYOK mode detection at top of QuickSetupStep

```kotlin
// Around line 2598-2605 in QuickSetupStep function
// Determine if this is BYOK mode - use same logic as WelcomeStep for consistency
// Anything that is NOT explicitly CIRIS_PROXY is treated as BYOK mode
val isCirisMode = state.setupMode == SetupMode.CIRIS_PROXY
val isBYOKMode = !isCirisMode

// Use isBYOKMode for all UI decisions
val effectiveBYOKMode = isBYOKMode
```

### 2. Update ALL UI elements to use effectiveBYOKMode

Search for all usages of `isBYOKMode` in QuickSetupStep and replace with `effectiveBYOKMode`:

**Header badge (~line 2617-2630):**

```kotlin
Surface(
    shape = RoundedCornerShape(20.dp),
    color = if (effectiveBYOKMode) SetupColors.InfoLight else SetupColors.SuccessLight,
    // ...
) {
    Text(
        text = if (effectiveBYOKMode) {
            localizedString("mobile.setup_byok_badge")  // Remove emoji prefix
        } else {
            localizedString("mobile.setup_free_badge")
        },
        // ...
    )
}
```

**Mode info card (~line 2650-2685):**

```kotlin
Surface(
    color = if (effectiveBYOKMode) SetupColors.InfoLight else SetupColors.SuccessLight,
    // ...
) {
    Icon(
        imageVector = if (effectiveBYOKMode) Icons.Filled.Settings else Icons.Filled.CheckCircle,
        tint = if (effectiveBYOKMode) SetupColors.InfoDark else SetupColors.SuccessDark,
        // ...
    )
    Text(
        text = if (effectiveBYOKMode) {
            localizedString("setup.quick_byok_active")
        } else {
            localizedString("setup.quick_ciris_active")
        },
        // ...
    )
}
```

**LLM config section (~line 2860-2880):**

```kotlin
SetupCollapsibleSection(
    subtitle = when {
        // ...configured case...
        effectiveBYOKMode -> localizedString("setup.llm_config_subtitle_required")
        else -> localizedString("setup.llm_config_subtitle_optional")
    },
    // ...
) {
    Text(
        text = if (effectiveBYOKMode) {
            localizedString("mobile.setup_byok_llm_desc")
        } else {
            localizedString("mobile.setup_ciris_llm_desc")
        },
        // ...
    )
}
```

## Emoji Tofu Issue

The badge text uses emoji prefix `"🔑 "` which may not render in all environments.

**Fix option 1**: Remove emoji, use icon instead

```kotlin
// Instead of: "🔑 " + localizedString("mobile.setup_byok_badge")
// Use just: localizedString("mobile.setup_byok_badge")
// And add an Icon composable before the text
```

**Fix option 2**: Use emoji font
Ensure emoji fonts are available. In Docker/Alpine:

```dockerfile
RUN apk add --no-cache font-noto-emoji
```

## Validation Logic

The validation in `SetupState.kt` is already correct - it uses:

```kotlin
SetupStep.QUICK_SETUP -> {
    when {
        setupMode == SetupMode.CIRIS_PROXY -> {
            // CIRIS Proxy needs OAuth
            if (!isGoogleAuth || googleIdToken == null) "Google sign-in required"
        }
        else -> {
            // BYOK needs provider + API key
            when {
                llmProvider.isEmpty() -> "Select provider"
                !isKeylessProvider && llmApiKey.isEmpty() -> "API key required"
                else -> null
            }
        }
    }
}
```

## Testing

1. Build WASM: `./gradlew :webApp:wasmJsBrowserDevelopmentExecutableDistribution`
2. Deploy to HA addon
3. Open HA ingress URL
4. Verify Welcome shows blue "Bring Your Own Key"
5. Click Continue
6. Verify QuickSetup shows blue "Bring Your Own Key Mode" card
7. Verify validation shows "API key is required" (not "Google sign-in required")
8. Enter API key, verify Next button enables

## Files Changed

- `shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SetupScreen.kt`
  - QuickSetupStep function: unified BYOK mode detection
  - All UI elements updated to use `effectiveBYOKMode`
