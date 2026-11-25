import os
import re
import io
import time
import logging
try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None
from .base import BaseTTSProvider

logger = logging.getLogger("soni_translate")

class GeminiTTSProvider(BaseTTSProvider):
    """
    Provider TTS haute fidélité utilisant l'API Google Gemini 2.5 / 2.0 (Audio Modality).
    Génère un audio naturel, expressif et parfaitement décodé au format WAV PCM uncompressé.
    """
    def __init__(self):
        super().__init__()
        self.valid_voices = {
            "puck": "Puck",
            "charon": "Charon",
            "fenrir": "Fenrir",
            "orus": "Orus",
            "aoede": "Aoede",
            "kore": "Kore",
            "leda": "Leda",
            "zephyr": "Zephyr",
            "callirrhoe": "Callirrhoe",
            "autonoe": "Autonoe",
            "europa": "Europa",
            "despina": "Despina",
            "thalassa": "Thalassa",
            "galatea": "Galatea",
            "iapetus": "Iapetus",
            "enceladus": "Enceladus",
        }

    def normalize_voice(self, voice_str):
        """Assainit le nom de la voix et mappe vers une voix pré-construite Gemini valide."""
        if not voice_str:
            return "Puck"
        v = str(voice_str).strip()
        # Enlever le préfixe 'Gemini/' ou 'gemini/'
        if "/" in v:
            v = v.split("/")[-1].strip()
        # Enlever les annotations entre parenthèses (ex: '(Gemini Homme)')
        v_clean = re.sub(r'\(.*?\)', '', v).strip()
        v_lower = v_clean.lower()

        # Mapping intelligent
        if v_lower == "fawkes":
            return "Charon"
        if v_lower in self.valid_voices:
            return self.valid_voices[v_lower]

        # Détection du genre pour fallbacks intelligents
        if any(w in v.lower() for w in ["femme", "female", "denise", "vivienne", "eloise", "kore", "aoede", "leda", "woman"]):
            return "Aoede"
        return "Puck"

    def _get_api_keys(self):
        """Récupère et déduplique toutes les clés Gemini configurées dans l'environnement."""
        keys = []
        try:
            from soni_translate.translate_segments import get_gemini_api_keys
            keys = get_gemini_api_keys()
        except Exception:
            pass
        for var in ["GOOGLE_API_KEY", "google_api_key", "GEMINI_API_KEY", "gemini_api_key"]:
            raw = os.getenv(var) or ""
            for part in re.split(r'[,;\n]+', raw):
                k = part.strip()
                if k and k not in keys:
                    keys.append(k)
        return keys

    def generate(self, text, voice, target_lang=None, output_file=None, **kwargs):
        """
        Génère la parole via l'API Gemini et écrit un fichier WAV valide et normalisé.
        """
        if not text or not str(text).strip():
            logger.warning("Gemini TTS : Texte vide fourni, saut de la génération.")
            return False

        if not output_file:
            logger.error("Gemini TTS : Aucun fichier de sortie 'output_file' spécifié.")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        keys = self._get_api_keys()
        if not keys:
            logger.error("Gemini TTS : Aucune clé GOOGLE_API_KEY ou GEMINI_API_KEY trouvée dans l'environnement.")
            return False

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.error("Gemini TTS : La bibliothèque 'google-genai' est requise. Veuillez faire : pip install google-genai")
            return False

        voice_clean = self.normalize_voice(voice)

        # Modèles à tenter par ordre de priorité
        preferred_model = os.getenv("GEMINI_TTS_MODEL", "").strip()
        models_to_try = []
        if preferred_model:
            models_to_try.append(preferred_model)
        for m in [
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-flash-tts",
            "gemini-2.5-pro-preview-tts",
            "gemini-2.0-flash"
        ]:
            if m not in models_to_try:
                models_to_try.append(m)

        clean_text = str(text).strip()

        # Essais avec rotation de clés et fallback de modèles
        for key in keys:
            try:
                client = genai.Client(api_key=key)
            except Exception as e_client:
                logger.warning(f"Gemini TTS : Erreur d'initialisation client avec la clé ({e_client}).")
                continue

            for model_name in models_to_try:
                try:
                    # Construction robuste de la configuration audio
                    try:
                        speech_cfg = types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_clean
                                )
                            )
                        )
                    except Exception:
                        speech_cfg = {
                            "voice_config": {
                                "prebuilt_voice_config": {
                                    "voice_name": voice_clean
                                }
                            }
                        }

                    config = types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=speech_cfg
                    )

                    logger.info(f"Gemini TTS : Synthèse avec modèle '{model_name}', voix '{voice_clean}' ({len(clean_text)} caractères)...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=clean_text,
                        config=config
                    )

                    # Extraction des octets audio
                    audio_bytes = None
                    mime_type = "audio/pcm;rate=24000"

                    candidates = getattr(response, "candidates", None) or []
                    for cand in candidates:
                        content = getattr(cand, "content", None)
                        if not content:
                            continue
                        parts = getattr(content, "parts", None) or []
                        for part in parts:
                            inline_data = getattr(part, "inline_data", None)
                            if inline_data and getattr(inline_data, "data", None):
                                audio_bytes = inline_data.data
                                mime_type = getattr(inline_data, "mime_type", mime_type) or mime_type
                                break
                        if audio_bytes:
                            break

                    if not audio_bytes:
                        logger.warning(f"Gemini TTS : Pas de données audio dans la réponse du modèle '{model_name}'.")
                        continue

                    # Décodage et conversion en fichier WAV standard valide
                    saved = False

                    # 1. Vérification si c'est déjà un conteneur audio (WAV / MP3 / OGG)
                    if audio_bytes.startswith(b"RIFF") or audio_bytes.startswith(b"ID3") or audio_bytes[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]:
                        try:
                            audio_np, sr = sf.read(io.BytesIO(audio_bytes))
                            if audio_np.ndim > 1:
                                audio_np = audio_np.mean(axis=1)
                            max_val = np.max(np.abs(audio_np))
                            if max_val > 0.001:
                                audio_np = audio_np * (0.96 / max_val)
                            sf.write(output_file, audio_np.astype(np.float32), sr)
                            saved = True
                        except Exception as e_dec:
                            logger.debug(f"Gemini TTS : Échec décodage conteneur ({e_dec}), tentative PCM direct...")

                    # 2. Décodage PCM brut (16-bit little-endian)
                    if not saved:
                        sr = 24000
                        if "rate=" in str(mime_type).lower():
                            try:
                                sr_part = str(mime_type).lower().split("rate=")[-1].split(";")[0].strip()
                                sr = int(sr_part)
                            except Exception:
                                sr = 24000

                        # S'assurer que la longueur est un multiple de 2
                        if len(audio_bytes) % 2 != 0:
                            audio_bytes = audio_bytes[:len(audio_bytes) - 1]

                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                        # Normalisation crête optimale (-0.35 dB)
                        max_val = np.max(np.abs(audio_np))
                        if max_val > 0.001:
                            audio_np = audio_np * (0.96 / max_val)

                        sf.write(output_file, audio_np, sr)
                        saved = True

                    if saved and os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                        logger.info(f"Gemini TTS : Audio généré avec succès -> {output_file} ({os.path.getsize(output_file)} octets)")
                        return True

                except Exception as e_gen:
                    err_str = str(e_gen)
                    logger.warning(f"Gemini TTS : Erreur avec modèle '{model_name}': {err_str}")
                    if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                        # Pause légère et rotation vers la clé suivante
                        time.sleep(1.0)
                        break
                    # Si erreur 404 / ModelNotFound, continuer vers le modèle suivant
                    continue

        logger.error(f"Gemini TTS : Échec de la génération pour '{clean_text[:40]}...' après épuisement des modèles et clés.")
        return False
