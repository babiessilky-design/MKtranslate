import os
import sys
import torch
import soundfile as sf
from .base import BaseTTSProvider

# Assurer l'accès au package omnivoice local si présent
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_omnivoice_src = os.path.join(_base_dir, "omnivoice")
if os.path.exists(_omnivoice_src) and _omnivoice_src not in sys.path:
    sys.path.insert(0, _omnivoice_src)

# Cache global de session pour garder le modèle chargé en mémoire vive
OMNIVOICE_MODEL_CACHE = None


class OmniVoiceProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Dossier de stockage des poids du modèle dans SoniTranslate
        self.local_dir = "weights/OmniVoice"

    def load_model(self):
        """Charge le modèle OmniVoice en mémoire vive une seule fois."""
        global OMNIVOICE_MODEL_CACHE
        if OMNIVOICE_MODEL_CACHE is None:
            self.logger.info("Initializing OmniVoice model from 'k2-fsa/OmniVoice'...")
            from omnivoice import OmniVoice
            
            # Création du dossier de stockage si inexistant
            os.makedirs(self.local_dir, exist_ok=True)
            
            # Téléchargement automatique depuis Hugging Face si le dossier est vide
            from huggingface_hub import snapshot_download
            if not os.listdir(self.local_dir):
                self.logger.info("Downloading OmniVoice model weights from Hugging Face...")
                snapshot_download(
                    repo_id="k2-fsa/OmniVoice",
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False
                )
            
            # Initialisation du modèle
            OMNIVOICE_MODEL_CACHE = OmniVoice.from_pretrained(
                self.local_dir,
                device_map=self.device,
                dtype=torch.float32 if self.device == 'cpu' else torch.float16
            )
        return OMNIVOICE_MODEL_CACHE

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        model = self.load_model()
        
        # Le nom de la voix passée est sous la forme : "Custom/Moussa_FR"
        voice_name = voice.split("/")[-1]
        
        # Récupération de l'audio de référence et du texte associé dans la Voice Library
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        voice_lib_dirs = [
            os.path.join(base_dir, "voice_library"),
            os.path.join(base_dir, "soni_translate", "voice_library"),
            "voice_library",
            os.path.join("soni_translate", "voice_library"),
            "/content/SoniTranslate/voice_library",
        ]
        
        ref_audio = None
        ref_txt_path = None
        for vdir in voice_lib_dirs:
            if not os.path.exists(vdir):
                continue
            for ext in [".wav", ".mp3", ".flac"]:
                cand = os.path.join(vdir, f"{voice_name}{ext}")
                if os.path.exists(cand):
                    ref_audio = cand
                    cand_txt = os.path.join(vdir, f"{voice_name}.txt")
                    if os.path.exists(cand_txt):
                        ref_txt_path = cand_txt
                    break
            if ref_audio:
                break

        if not ref_audio:
            for fallback_name in ["ref_homme_fr", "ref_femme_fr"]:
                for vdir in voice_lib_dirs:
                    cand = os.path.join(vdir, f"{fallback_name}.wav")
                    if os.path.exists(cand):
                        ref_audio = cand
                        break
                if ref_audio:
                    break

        if not ref_audio:
            raise FileNotFoundError(f"Reference audio not found in voice library: {voice_name}")
            
        if ref_txt_path and os.path.exists(ref_txt_path):
            with open(ref_txt_path, "r", encoding="utf-8") as f:
                ref_text = f.read().strip()
        else:
            ref_text = "Hello there." # Fallback par défaut si pas de fichier texte
            
        # Génération de l'audio
        # OmniVoice renvoie une liste de waveforms, on prend la première [0]
        wav = model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speed=1.0
        )
        
        # Enregistrement natif à 24 000 Hz comme défini par pyVideoTrans
        sf.write(output_file, wav[0], 24000)
