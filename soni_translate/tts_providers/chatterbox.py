import os
import sys
from pathlib import Path
import torch
import torchaudio
import soundfile as sf
from .base import BaseTTSProvider

# Assurer l'accès au package chatterbox déplacé localement
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_chatterbox_src = os.path.join(_base_dir, "chatterbox", "src")
if os.path.exists(_chatterbox_src) and _chatterbox_src not in sys.path:
    sys.path.insert(0, _chatterbox_src)

# Cache de session global pour conserver le modèle chargé en VRAM
CHATTERBOX_MODEL_CACHE = None

class ChatterBoxProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        
        # Détection automatique du meilleur processeur disponible (CUDA, MPS ou CPU)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

    def get_language_id(self, target_lang):
        """Mappe la langue de SoniTranslate vers le code ISO à 2 lettres de Chatterbox"""
        lang = str(target_lang).lower()
        if "french" in lang or "fr" in lang: return "fr"
        elif "spanish" in lang or "es" in lang: return "es"
        elif "japanese" in lang or "ja" in lang: return "ja"
        elif "chinese" in lang or "zh" in lang: return "zh"
        elif "italian" in lang or "it" in lang: return "it"
        elif "portuguese" in lang or "pt" in lang: return "pt"
        elif "hindi" in lang or "hi" in lang: return "hi"
        elif "german" in lang or "de" in lang: return "de"
        elif "danish" in lang or "da" in lang: return "da"
        elif "dutch" in lang or "nl" in lang: return "nl"
        elif "finnish" in lang or "fi" in lang: return "fi"
        elif "greek" in lang or "el" in lang: return "el"
        elif "hebrew" in lang or "he" in lang: return "he"
        elif "korean" in lang or "ko" in lang: return "ko"
        elif "malay" in lang or "ms" in lang: return "ms"
        elif "norwegian" in lang or "no" in lang: return "no"
        elif "polish" in lang or "pl" in lang: return "pl"
        elif "russian" in lang or "ru" in lang: return "ru"
        elif "swedish" in lang or "sv" in lang: return "sv"
        elif "swahili" in lang or "sw" in lang: return "sw"
        elif "turkish" in lang or "tr" in lang: return "tr"
        elif "arabic" in lang or "ar" in lang: return "ar"
        else: return "en"

    def load_model(self):
        """Charge le modèle Chatterbox en mémoire vive une seule fois."""
        global CHATTERBOX_MODEL_CACHE
        if CHATTERBOX_MODEL_CACHE is None:
            self.logger.info(f"Initializing Chatterbox Multilingual model on '{self.device}'...")
            
            # S'assurer que le chemin vers le dossier local chatterbox/src est présent
            if os.path.exists(_chatterbox_src) and _chatterbox_src not in sys.path:
                sys.path.insert(0, _chatterbox_src)

            # Garantir l'utilisation de 'eager' attention pour la compatibilité output_attentions=True
            try:
                import chatterbox.models.t3.llama_configs as lc
                if hasattr(lc, "LLAMA_520M_CONFIG_DICT"):
                    lc.LLAMA_520M_CONFIG_DICT["attn_implementation"] = "eager"
                if hasattr(lc, "LLAMA_CONFIGS") and "Llama_520M" in lc.LLAMA_CONFIGS:
                    lc.LLAMA_CONFIGS["Llama_520M"]["attn_implementation"] = "eager"
            except Exception:
                pass

            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            
            # Chargement natif du modèle Chatterbox Multilingual
            CHATTERBOX_MODEL_CACHE = ChatterboxMultilingualTTS.from_pretrained(
                device=self.device
            )

            # S'assurer que les modèles et backends instanciés ont bien eager attention
            if hasattr(CHATTERBOX_MODEL_CACHE, "t3"):
                t3_mod = CHATTERBOX_MODEL_CACHE.t3
                if hasattr(t3_mod, "cfg"):
                    t3_mod.cfg._attn_implementation = "eager"
                if hasattr(t3_mod, "tfmr") and hasattr(t3_mod.tfmr, "config"):
                    t3_mod.tfmr.config._attn_implementation = "eager"
                if hasattr(t3_mod, "patched_model") and hasattr(t3_mod.patched_model, "config"):
                    t3_mod.patched_model.config._attn_implementation = "eager"
        return CHATTERBOX_MODEL_CACHE

    def find_reference_audio(self, voice_name):
        """Localise intelligemment le fichier audio de référence pour la voix demandée."""
        candidates = [
            os.path.join(_base_dir, "voice_library", f"{voice_name}.wav"),
            os.path.join(_base_dir, "voice_library", f"{voice_name}.mp3"),
            os.path.join(_base_dir, "voice_library", f"{voice_name}.flac"),
            os.path.join(_base_dir, "soni_translate", "voice_library", f"{voice_name}.wav"),
            os.path.join(_base_dir, "soni_translate", "voice_library", f"{voice_name}.mp3"),
            os.path.join(_base_dir, "soni_translate", "voice_library", f"{voice_name}.flac"),
            os.path.join("voice_library", f"{voice_name}.wav"),
            os.path.join("voice_library", f"{voice_name}.mp3"),
            os.path.join("soni_translate", "voice_library", f"{voice_name}.wav"),
            f"/content/SoniTranslate/voice_library/{voice_name}.wav",
        ]
        for path in candidates:
            if os.path.exists(path):
                return os.path.abspath(path)

        # Fallback automatique sur l'une des voix de base du projet si disponible
        for fallback_name in ["ref_homme_fr", "ref_femme_fr"]:
            for folder in [
                os.path.join(_base_dir, "voice_library"),
                os.path.join(_base_dir, "soni_translate", "voice_library"),
                "voice_library",
                os.path.join("soni_translate", "voice_library")
            ]:
                fallback_path = os.path.join(folder, f"{fallback_name}.wav")
                if os.path.exists(fallback_path):
                    self.logger.warning(f"Voice '{voice_name}' not found, falling back to '{fallback_name}' ({fallback_path})")
                    return os.path.abspath(fallback_path)

        raise FileNotFoundError(f"Reference audio not found for voice: {voice_name}")

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        model = self.load_model()
        
        # Le nom de la voix passée est sous la forme : "Custom/ref_homme_fr" ou "ref_homme_fr"
        voice_name = voice.split("/")[-1] if "/" in voice else voice
        ref_audio = self.find_reference_audio(voice_name)
            
        lang_id = self.get_language_id(target_lang)
        
        # Sanitisation anti-hallucination : ponctuation terminale obligatoire et nettoyage
        clean_text = text.strip()
        if not clean_text or not any(c.isalnum() for c in clean_text):
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            import numpy as np
            silence_arr = np.zeros(int(model.sr * 0.4), dtype=np.float32)
            sf.write(output_file, silence_arr, model.sr)
            return

        if clean_text[-1] not in ('.', '!', '?', ':', ';', '…'):
            clean_text += '.'

        # Paramètres d'expressivité et température naturelle (0.8 par défaut)
        temperature = float(kwargs.get("temperature", os.getenv("CHATTERBOX_TEMPERATURE", 0.8)))
        exaggeration = float(kwargs.get("exaggeration", 0.5))
        cfg_weight = float(kwargs.get("cfg_weight", 0.5))
        
        # Calcul dynamique du plafond max_new_tokens (Anti-Hallucination Guard)
        num_words = max(1, len(clean_text.split()))
        seg_duration = kwargs.get("seg_duration")
        if seg_duration and float(seg_duration) > 0:
            max_from_dur = int(float(seg_duration) * 55) + 30
            max_from_words = int(num_words * 20) + 50
            max_new_tokens = min(600, max(50, min(max_from_dur, max_from_words)))
        else:
            max_new_tokens = min(600, max(50, int(num_words * 18) + 60))
        
        # --- SÉCURITÉ DE TRANSFERT D'ACCENT ---
        # Si on fait du cross-lingual (voix fr parlant anglais), on met cfg_weight à 0.0
        # pour éviter le transfert d'accent français et garder l'anglais natif
        is_cross_lingual = "fr" in voice_name.lower() and "en" in lang_id
        if is_cross_lingual:
            self.logger.info("Cross-lingual detected: setting cfg_weight to 0.0 to prevent accent transfer.")
            cfg_weight = 0.0
            
        # Génération directe via Chatterbox (renvoie un tenseur PyTorch)
        wav_tensor = model.generate(
            clean_text,
            language_id=lang_id,
            audio_prompt_path=ref_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        
        # Enregistrement natif à 24000Hz avec soundfile
        wav_tensor = wav_tensor.detach().cpu()
        if wav_tensor.ndim == 2:
            wav_np = wav_tensor.transpose(0, 1).numpy()
        else:
            wav_np = wav_tensor.numpy()

        import numpy as np
        if wav_np.size > 0:
            max_val = float(np.abs(wav_np).max())
            if max_val > 1.0:
                wav_np = wav_np / max_val * 0.98

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        try:
            sf.write(output_file, wav_np, model.sr)
        except Exception:
            # Fallback format si le codec ogg direct échoue
            wav_fallback = output_file.rsplit('.', 1)[0] + ".wav"
            sf.write(wav_fallback, wav_np, model.sr)
            if output_file.endswith('.ogg'):
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_fallback, "-c:a", "libvorbis", output_file],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                if os.path.exists(output_file):
                    try:
                        os.remove(wav_fallback)
                    except Exception:
                        pass
