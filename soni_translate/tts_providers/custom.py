import os
from .base import BaseTTSProvider

class CustomCloningProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.providers = {}

    def get_cloning_provider(self, engine_name):
        if engine_name not in self.providers:
            if engine_name == "ChatterBox Multilingual":
                from .chatterbox import ChatterBoxProvider
                self.providers[engine_name] = ChatterBoxProvider()
            elif engine_name == "OmniVoice":
                from .omnivoice import OmniVoiceProvider
                self.providers[engine_name] = OmniVoiceProvider()
            else:
                # Moteur F5-TTS
                from .f5 import F5Provider
                self.providers[engine_name] = F5Provider()
                
        return self.providers[engine_name]

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        # On lit le moteur de clonage sélectionné par l'utilisateur (F5-TTS, ChatterBox ou OmniVoice)
        cloning_engine = os.getenv("CUSTOM_CLONING_ENGINE", "F5-TTS")
        self.logger.info(f"Custom routing: delegating generation to '{cloning_engine}'...")
        
        provider = self.get_cloning_provider(cloning_engine)
        provider.generate(text, voice, target_lang, output_file, **kwargs)
