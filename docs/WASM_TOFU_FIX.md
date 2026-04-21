# WASM "Tofu" Icon Fix - Port Guide for CIRISAgent

**Date**: April 21, 2025
**Issue**: Unicode emoji characters render as empty boxes ("tofu") in Kotlin/WASM + Skia

## Problem

WASM/Skia doesn't include font support for Unicode emoji/symbols. Characters like `❓`, `▶`, `≈`, `⚒` render as empty boxes in:
- Rising SSE bubbles from bottom-left
- Yellow timeline bar (BubbleNet collapsed view)
- Caught bubbles panel
- Skill import dialog icons
- Various UI elements with bullet points

## Solution

Replace `Text(emoji)` calls with `Icon(imageVector)` using CIRISIcons SVG-based ImageVectors.

## Files to Copy from `mobile-web/`

Copy these files to the equivalent location in CIRISAgent's mobile-web:

### 1. CIRISIcons.kt (emoji mapping functions)

**Path**: `shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/components/CIRISIcons.kt`

**Add these functions at the end of the file** (after the `CIRISIcons` object):

```kotlin
/**
 * Map emoji/symbol strings to CIRIS icons for WASM/Skia rendering.
 * SSE events and backend use Unicode symbols; these need conversion to ImageVectors.
 */
fun emojiToIcon(emoji: String): ImageVector? = when (emoji) {
    // Event type symbols
    "\u2753" -> CIRISIcons.thoughtStart       // ❓ thought_start / recall
    "\u25B6" -> CIRISIcons.speak              // ▶ snapshot_and_context / speak
    "\u2248" -> CIRISIcons.dma                // ≈ dma_results
    "\u2139" -> CIRISIcons.idma               // ℹ idma_result
    "\u26A0" -> CIRISIcons.warning            // ⚠ aspdma_result / action_result / default
    "\u2692" -> CIRISIcons.tool               // ⚒ tsaspdma_result / tool
    "\u25CE" -> CIRISIcons.conscience         // ◎ conscience_result
    // Action symbols
    "\u25CB" -> CIRISIcons.observe            // ○ observe
    "\u2716" -> CIRISIcons.reject             // ✖ reject
    "\u22EF" -> CIRISIcons.ponder             // ⋯ ponder
    "\u275A\u275A" -> CIRISIcons.defer        // ❚❚ defer
    "\u2795" -> CIRISIcons.memorize           // ➕ memorize
    "\u2796" -> CIRISIcons.forget             // ➖ forget
    "\u2714" -> CIRISIcons.taskComplete       // ✔ task_complete
    // Common fallbacks
    "\u2705" -> CIRISIcons.check              // ✅ check
    "\u274C" -> CIRISIcons.xmark              // ❌ x-mark
    // Skill dialog symbols
    "\u2756" -> CIRISIcons.identityDiamond    // ❖ identity
    "\u25A0" -> CIRISIcons.requirements       // ■ requirements
    "\u2261" -> CIRISIcons.instruct           // ≡ instructions
    "\u25C6" -> CIRISIcons.shield             // ◆ safety
    else -> null
}

/**
 * Get icon with default fallback for unrecognized emojis.
 */
fun emojiToIconOrDefault(emoji: String): ImageVector =
    emojiToIcon(emoji) ?: CIRISIcons.circle

/**
 * Get bus color for an emoji symbol.
 * Used for tinting icons in the bubble overlay.
 */
fun emojiBusColor(emoji: String): Color = when (emoji) {
    // LLM bus (purple-ish)
    "\u2753", "\u2248", "\u2139", "\u26A0", "\u25CE" -> CIRISColors.BusLLM
    // COMM bus (teal)
    "\u25B6", "\u25CB" -> CIRISColors.BusComm
    // TOOL bus (orange)
    "\u2692", "\u2714" -> CIRISColors.BusTool
    // MEMORY bus (violet)
    "\u2795", "\u2796" -> CIRISColors.BusMemory
    // WISE bus (brass)
    "\u22EF", "\u275A\u275A" -> CIRISColors.BusWise
    // RUNTIME bus (magenta)
    "\u2716" -> CIRISColors.BusRuntime
    else -> Color.Unspecified
}
```

**Required import** (add to imports at top):
```kotlin
import androidx.compose.ui.graphics.Color
```

### 2. InteractScreen.kt (bubble/timeline rendering)

**Path**: `shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/InteractScreen.kt`

**Key changes**:

Add import:
```kotlin
import ai.ciris.mobile.shared.ui.components.emojiToIconOrDefault
import ai.ciris.mobile.shared.ui.components.emojiBusColor
```

#### FullScreenFloatingBubble

Replace:
```kotlin
Text(
    text = emoji,
    fontSize = 24.sp,
    modifier = modifier...
)
```

With:
```kotlin
Icon(
    imageVector = emojiToIconOrDefault(emoji),
    contentDescription = null,
    modifier = modifier
        .size(28.dp)
        .offset(x = wobble.dp, y = offsetY)
        .alpha(alpha)
        .zIndex(100f)
        .then(tappableModifier),
    tint = emojiBusColor(emoji)
)
```

#### BubbleNet collapsed view (yellow bar)

Replace:
```kotlin
Text(event.emoji, fontSize = 14.sp)
```

With:
```kotlin
Icon(
    imageVector = emojiToIconOrDefault(event.emoji),
    contentDescription = null,
    modifier = Modifier.size(14.dp),
    tint = emojiBusColor(event.emoji)
)
```

#### CaughtBubblesPanel

Replace:
```kotlin
Text(b.emoji, fontSize = 10.sp)
```

With:
```kotlin
Icon(
    imageVector = emojiToIconOrDefault(b.emoji),
    contentDescription = null,
    modifier = Modifier.size(14.dp),
    tint = emojiBusColor(b.emoji)
)
```

### 3. SkillImportDialog.kt (WorkshopCard)

**Path**: `shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/components/SkillImportDialog.kt`

In `WorkshopCard`, replace:
```kotlin
Text(text = emoji, style = MaterialTheme.typography.titleLarge)
```

With:
```kotlin
Icon(
    imageVector = emojiToIconOrDefault(emoji),
    contentDescription = null,
    modifier = Modifier.padding(end = 12.dp).size(24.dp),
    tint = MaterialTheme.colorScheme.primary
)
```

### 4. SettingsScreen.kt (retry buttons, log messages)

**Path**: `shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SettingsScreen.kt`

Replace emoji strings with icons:
- `"🔄 Retry"` -> `Row { Icon(CIRISIcons.refresh); Text("Retry") }`
- `"🔄"` spinner text -> `Icon(CIRISIcons.refresh)`
- `"[verify] ✓"` -> `"[verify] +"`
- `"[verify] ✗"` -> `"[verify] X"`

### 5. Unicode bullet points (multiple files)

Replace `"• "` (bullet) with `"- "` throughout:
- `TrustPage.kt`
- `LLMSettingsScreen.kt`
- `SetupScreen.kt`
- `WalletPage.kt`
- `SkillStudioScreen.kt`
- `InteractViewModel.kt`

Replace `" · "` (middle dot separator) with `" - "`:
- `SettingsScreen.kt` (hardware info)
- `InteractScreen.kt` (kindLabel separators)

## Quick Copy Commands

```bash
# From CIRISHome directory
cd ~/CIRISHome

# Copy changed files to CIRISAgent mobile-web (adjust paths as needed)
AGENT_MOBILE=~/CIRISAgent/mobile-web
cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/components/CIRISIcons.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/components/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/InteractScreen.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/components/SkillImportDialog.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/components/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SettingsScreen.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/

# Additional files with bullet point fixes
cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/TrustPage.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/LLMSettingsScreen.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SetupScreen.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/WalletPage.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/SkillStudioScreen.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/ui/screens/

cp mobile-web/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/viewmodels/InteractViewModel.kt \
   $AGENT_MOBILE/shared/src/commonMain/kotlin/ai/ciris/mobile/shared/viewmodels/
```

## Testing

After copying, rebuild WASM:
```bash
cd $AGENT_MOBILE
./gradlew :webApp:wasmJsBrowserDevelopmentExecutable
```

Deploy to HA addon and verify:
1. SSE bubbles rise from bottom-left as colored icons (not boxes)
2. Yellow timeline bar shows icons (not boxes)
3. Caught bubbles show icons
4. Skill import dialog shows icons for sections

## Note on KT-69154

Production WASM builds remain broken due to Kotlin bug KT-69154. Always use `developmentExecutable` builds.
