import os
import torch
from .base import BaseTTSProvider

# Cache global de session pour garder F5-TTS chargé en VRAM
F5_MODEL_CACHE = None

class F5Provider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Charge le modèle officiel unifié F5-TTS en VRAM une seule fois."""
        global F5_MODEL_CACHE
        if F5_MODEL_CACHE is None:
            self.logger.info("Initializing official F5-TTS model (SWivid/F5-TTS)...")
            from f5_tts.api import F5TTS
            
            # F5TTS télécharge automatiquement le modèle de base et le vocodeur Vocos depuis Hugging Face
            F5_MODEL_CACHE = F5TTS(
                device=self.device,
                use_ema=True
            )
        return F5_MODEL_CACHE

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        model = self.load_model()
        
        # Le nom de la voix passée est sous la forme : "Custom/Moussa_FR"
        voice_name = voice.split("/")[-1]
        
        # Récupération de l'audio de référence et du texte associé dans ta Voice Library
        project_root = "/content/SoniTranslate"
        ref_audio = os.path.join(project_root, "voice_library", f"{voice_name}.wav")
        ref_txt_path = os.path.join(project_root, "voice_library", f"{voice_name}.txt")
        
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Reference audio not found in voice library: {ref_audio}")
            
        if os.path.exists(ref_txt_path):
            with open(ref_txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        else:
            ref_text = "Hello there." # Fallback par défaut si pas de fichier texte
            
        speed = kwargs.get("speed", 1.0)
        
        # Génération directe via l'API officielle F5-TTS
        # Elle écrit directement le fichier audio final à l'emplacement indiqué
        model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            file_wave=output_file,
            remove_silence=False,
            speed=speed
        )
