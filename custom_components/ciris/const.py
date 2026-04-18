"""Constants for the CIRIS integration."""

DOMAIN = "ciris"

# Configuration - Parent entry (API connection)
CONF_API_URL = "api_url"
CONF_API_KEY = "api_key"  # pragma: allowlist secret
CONF_TIMEOUT = "timeout"
CONF_CHANNEL = "channel"

# Configuration - Sub-entry (Context profile)
CONF_PROFILE_NAME = "profile_name"
CONF_ROOM_TYPE = "room_type"
CONF_LANGUAGE = "language"
CONF_SAFETY_LEVEL = "safety_level"
CONF_RESPONSE_STYLE = "response_style"
CONF_CUSTOM_INSTRUCTIONS = "custom_instructions"
CONF_WAKE_WORD = "wake_word"

# Defaults
DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_TIMEOUT = 30
DEFAULT_CHANNEL = "homeassistant"
DEFAULT_LANGUAGE = "en"
DEFAULT_SAFETY_LEVEL = "family_friendly"
DEFAULT_RESPONSE_STYLE = "normal"

# Room types with descriptions
ROOM_TYPES = {
    "adult_room": "Adult space - unrestricted content, full capabilities",
    "kids_room": "Children's space - safe content, simplified responses",
    "shared_space": "Family space - balanced for all ages",
    "elderly_care": "Senior care - patient, clear, health-aware responses",
    "office": "Work space - professional, task-focused responses",
    "guest_room": "Guest space - polite, helpful, privacy-conscious",
}

# Safety levels
SAFETY_LEVELS = {
    "unrestricted": "Full capabilities, adult content allowed",
    "family_friendly": "Safe for family, no explicit content",
    "kids_safe": "Child-safe, educational focus, no violence",
    "elderly_safe": "Clear language, health topics allowed, no alarming content",
}

# Response styles
RESPONSE_STYLES = {
    "normal": "Standard conversational responses",
    "simplified": "Simple words, short sentences (elderly/kids)",
    "detailed": "Comprehensive explanations",
    "brief": "Concise, to-the-point answers",
    "formal": "Professional, formal tone",
    "playful": "Fun, engaging tone (kids)",
}

# Language options (common ones)
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "ru": "Russian",
}

# Context profile presets
CONTEXT_PRESETS = {
    "kids_room": {
        "room_type": "kids_room",
        "safety_level": "kids_safe",
        "response_style": "playful",
        "custom_instructions": (
            "You are talking to children. Use simple words, be encouraging, "
            "and make learning fun. Never discuss adult topics, violence, "
            "or anything scary. Focus on being helpful and educational."
        ),
    },
    "elderly_care": {
        "room_type": "elderly_care",
        "safety_level": "elderly_safe",
        "response_style": "simplified",
        "custom_instructions": (
            "You are assisting an elderly person. Speak clearly and patiently. "
            "Use simple language and short sentences. Be respectful and kind. "
            "If they mention health concerns, suggest consulting their doctor. "
            "Help them with technology questions patiently."
        ),
    },
    "adult_room": {
        "room_type": "adult_room",
        "safety_level": "unrestricted",
        "response_style": "normal",
        "custom_instructions": "",
    },
    "shared_space": {
        "room_type": "shared_space",
        "safety_level": "family_friendly",
        "response_style": "normal",
        "custom_instructions": (
            "This is a shared family space. Keep responses appropriate for all ages. "
            "Be helpful and friendly to everyone."
        ),
    },
    "office": {
        "room_type": "office",
        "safety_level": "family_friendly",
        "response_style": "formal",
        "custom_instructions": (
            "This is a work environment. Maintain a professional tone. "
            "Focus on productivity and task completion. Be efficient."
        ),
    },
}
