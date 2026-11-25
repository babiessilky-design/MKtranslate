from .custom import CustomCloningProvider
try:
    from .fish_audio import FishAudioProvider, FishAudioTTSProvider
except ImportError:
    FishAudioProvider = None
    FishAudioTTSProvider = None

try:
    from .gemini_tts import GeminiTTSProvider
except ImportError:
    GeminiTTSProvider = None

# Enregistrement de nos Providers actifs
PROVIDERS_REGISTRY = {
    "Custom": CustomCloningProvider(),
    "FishAudio": None,  # Lazy loaded if needed
    "Gemini": GeminiTTSProvider() if GeminiTTSProvider else None,
    "gemini": GeminiTTSProvider() if GeminiTTSProvider else None,
}

def get_provider(voice_name):
    """
    Analyse le préfixe de la voix (ex: 'Custom/Moussa_FR' -> 'Custom')
    et retourne le Provider correspondant.
    """
    if not isinstance(voice_name, str):
        return None
        
    prefix = voice_name.split("/")[0] if "/" in voice_name else "Edge"
    return PROVIDERS_REGISTRY.get(prefix, None)

