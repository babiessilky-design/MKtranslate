import os
import sys
import torch
import numpy as np
import soundfile as sf
from .base import BaseTTSProvider

# Assurer l'accès au package omnivoice local si présent
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_omnivoice_src = os.path.join(_base_dir, "omnivoice")
if os.path.exists(_omnivoice_src) and _omnivoice_src not in sys.path:
    sys.path.insert(0, _omnivoice_src)

# Cache global de session pour garder le modèle et les profils de voix en mémoire vive
OMNIVOICE_MODEL_CACHE = None
OMNIVOICE_PROMPT_CACHE = {}


class OmniVoiceProvider(BaseTTSProvider):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Dossier de stockage des poids du modèle dans SoniTranslate
        self.local_dir = "weights/OmniVoice"

    def get_language_id(self, target_lang):
        """Mappe la langue de SoniTranslate vers le code ISO à 2 lettres reconnu par OmniVoice"""
        mapping = {
            "fr": "fr", "fra": "fr", "french": "fr", "français": "fr",
            "en": "en", "eng": "en", "english": "en", "anglais": "en",
            "es": "es", "spa": "es", "spanish": "es", "espagnol": "es",
            "de": "de", "deu": "de", "german": "de", "allemand": "de",
            "it": "it", "ita": "it", "italian": "it", "italien": "it",
            "pt": "pt", "por": "pt", "portuguese": "pt", "portugais": "pt",
            "ru": "ru", "rus": "ru", "russian": "ru", "russe": "ru",
            "zh": "zh", "zho": "zh", "chinese": "zh", "chinois": "zh",
            "ja": "ja", "jpn": "ja", "japanese": "ja", "japonais": "ja",
            "ko": "ko", "kor": "ko", "korean": "ko", "coréen": "ko",
            "ar": "ar", "ara": "ar", "arabic": "ar", "arabe": "ar",
            "hi": "hi", "hin": "hi", "hindi": "hi"
        }
        clean_lang = str(target_lang).lower().strip()
        return mapping.get(clean_lang, clean_lang[:2] if len(clean_lang) >= 2 else "en")

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
            self.logger.info("OmniVoice model loaded successfully.")
        return OMNIVOICE_MODEL_CACHE

    def find_reference_materials(self, voice_name):
        """Localise intelligemment le fichier audio de référence et son texte associé."""
        voice_lib_dirs = [
            os.path.join(_base_dir, "voice_library"),
            os.path.join(_base_dir, "soni_translate", "voice_library"),
            "voice_library",
            os.path.join("soni_translate", "voice_library"),
            "/content/SoniTranslate/voice_library",
        ]
        
        ref_audio = None
        ref_text = None
        ref_txt_path = None

        names_to_try = [voice_name]
        v_low = voice_name.lower()
        if v_low in ["cosy", "f5-tts_cosy"]:
            names_to_try = ["cosy", "f5-tts_cosy"]
        elif v_low in ["ref_homme_fr", "ref_homme", "homme"]:
            names_to_try = ["ref_homme_fr", "ref_homme", "homme"]
        elif v_low in ["ref_femme_fr", "ref_femme", "femme"]:
            names_to_try = ["ref_femme_fr", "ref_femme", "femme"]

        for vname in names_to_try:
            for vdir in voice_lib_dirs:
                if not os.path.exists(vdir):
                    continue
                for ext in [".wav", ".mp3", ".flac"]:
                    cand = os.path.join(vdir, f"{vname}{ext}")
                    if os.path.exists(cand):
                        ref_audio = os.path.abspath(cand)
                        cand_txt = os.path.join(vdir, f"{vname}.txt")
                        if os.path.exists(cand_txt):
                            ref_txt_path = os.path.abspath(cand_txt)
                        break
                if ref_audio:
                    break
            if ref_audio:
                break

        if not ref_audio:
            for fallback_name in ["ref_homme_fr", "ref_femme_fr", "cosy", "f5-tts_cosy"]:
                for vdir in voice_lib_dirs:
                    cand = os.path.join(vdir, f"{fallback_name}.wav")
                    if os.path.exists(cand):
                        ref_audio = os.path.abspath(cand)
                        cand_txt = os.path.join(vdir, f"{fallback_name}.txt")
                        if os.path.exists(cand_txt):
                            ref_txt_path = os.path.abspath(cand_txt)
                        break
                if ref_audio:
                    break

        if not ref_audio:
            raise FileNotFoundError(f"Reference audio not found for voice: {voice_name}")

        if ref_txt_path and os.path.exists(ref_txt_path):
            with open(ref_txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    ref_text = content

        return ref_audio, ref_text, ref_txt_path

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        model = self.load_model()
        from omnivoice import OmniVoiceGenerationConfig
        
        # Le nom de la voix passée est sous la forme : "Custom/ref_homme_fr" ou "ref_homme_fr"
        voice_name = voice.split("/")[-1] if "/" in voice else voice
        
        # Récupération de l'audio de référence et du texte associé
        ref_audio, ref_text, ref_txt_path = self.find_reference_materials(voice_name)
        lang_id = self.get_language_id(target_lang)

        # Gestion et mise en cache du VoiceClonePrompt pour garantir une voix 100% cohérente et fidèle
        global OMNIVOICE_PROMPT_CACHE
        if OMNIVOICE_PROMPT_CACHE is None:
            OMNIVOICE_PROMPT_CACHE = {}

        prompt = OMNIVOICE_PROMPT_CACHE.get(ref_audio)
        if prompt is None:
            self.logger.info(f"OmniVoice : Création et encodage du profil vocal pour '{voice_name}'...")
            prompt = model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,  # Si None, OmniVoice transcrit automatiquement l'audio fidèlement
                preprocess_prompt=True
            )
            OMNIVOICE_PROMPT_CACHE[ref_audio] = prompt

            # Sauvegarde de la transcription automatique si le fichier .txt n'existait pas
            if not ref_text and hasattr(prompt, "ref_text") and prompt.ref_text:
                try:
                    auto_txt = os.path.splitext(ref_audio)[0] + ".txt"
                    with open(auto_txt, "w", encoding="utf-8") as f:
                        f.write(prompt.ref_text.strip())
                    self.logger.info(f"OmniVoice : Transcription de référence auto-générée dans '{auto_txt}'")
                except Exception as e_write:
                    self.logger.debug(f"Impossible de sauvegarder le fichier texte ({e_write})")

        # Fixation du Seed pour reproductibilité absolue et stabilité parfaite du timbre (anti-dérive)
        seed = int(kwargs.get("seed", os.getenv("OMNIVOICE_SEED", 42)))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # Configuration optimisée pour fidélité maximale de clonage et élimination des hallucinations
        # guidance_scale: 1.9 est la valeur idéale recommandée par k2-fsa (évite la sur-saturation métallique)
        guidance_scale = float(kwargs.get("guidance_scale", os.getenv("OMNIVOICE_GUIDANCE_SCALE", 1.9)))
        
        # position_temperature: 0.0 pour un échantillonnage déterministe et stable
        position_temperature = float(kwargs.get("position_temperature", os.getenv("OMNIVOICE_POS_TEMP", 0.0)))
        
        # num_step: 48 étapes pour une fidélité acoustique supérieure, grain soyeux et élimination des artefacts de diffusion
        num_step = int(kwargs.get("num_step", os.getenv("OMNIVOICE_NUM_STEP", 48)))

        clean_text = text.strip()
        sample_rate = getattr(model, "sampling_rate", 24000) or 24000

        # Si le texte ne contient aucun caractère alphanumérique (ex: ".", "...", "-"), on crée un micro-silence
        if not any(c.isalnum() for c in clean_text):
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            silence_arr = np.zeros(int(sample_rate * 0.4), dtype=np.float32)
            sf.write(output_file, silence_arr, sample_rate)
            return

        # Ponctuation terminale forcée pour stabiliser le masque de fin
        if clean_text[-1] not in ('.', '!', '?', ':', ';', '…'):
            clean_text += '.'

        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            position_temperature=position_temperature,
            class_temperature=0.0,  # Échantillonnage déterministe
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=False,  # Désactivé : évite que remove_silence ne vide les segments courts !
        )

        speed = float(kwargs.get("speed", 1.0))

        # Plafond de durée anti-débordement basé sur le segment vidéo original
        target_duration = None
        seg_duration = kwargs.get("seg_duration")
        if seg_duration and float(seg_duration) > 0:
            target_duration = min(float(seg_duration) * 1.25 + 0.5, 30.0)

        # Synthèse vocale avec conditionnement exact du profil vocal, durée bornée et normalisation
        wav = model.generate(
            text=clean_text,
            language=lang_id,
            voice_clone_prompt=prompt,
            generation_config=gen_config,
            speed=speed,
            duration=target_duration,
            normalize_text=True,
        )
        
        wav_out = wav[0] if (wav is not None and len(wav) > 0) else np.array([], dtype=np.float32)

        # Protection absolue contre les tableaux audio vides
        if wav_out is None or getattr(wav_out, "size", len(wav_out)) == 0:
            self.logger.warning(f"OmniVoice a produit un segment vide pour '{clean_text[:30]}'. Utilisation d'un silence de secours.")
            wav_out = np.zeros(int(sample_rate * 0.4), dtype=np.float32)
        else:
            # Normalisation studio broadcast (-1.0 dBFS, amplitude crête 0.89)
            max_val = float(np.abs(wav_out).max())
            if max_val > 0.01:
                wav_out = (wav_out / max_val) * 0.89

            # Fondu sortant doux de 50ms pour éliminer tout souffle résiduel ou clic
            fade_len = int(sample_rate * 0.05)
            if fade_len > 0 and len(wav_out) >= fade_len:
                fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                wav_out[-fade_len:] *= fade_curve

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        try:
            sf.write(output_file, wav_out, sample_rate)
        except Exception:
            wav_fallback = output_file.rsplit('.', 1)[0] + ".wav"
            sf.write(wav_fallback, wav_out, sample_rate)
            if output_file != wav_fallback:
                import subprocess
                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_fallback, output_file],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
