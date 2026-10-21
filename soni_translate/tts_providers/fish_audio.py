import os
import sys
import torch
import numpy as np
import soundfile as sf
from .base import BaseTTSProvider

# Assurer l'accès au package fish_speech local
_base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for candidate_dir in [
    os.path.join(_base_dir, "fish_speech"),
    os.path.join(_base_dir, "fish_speech", "fish_speech"),
    "/content/SoniTranslate/fish_speech",
]:
    if os.path.exists(candidate_dir) and candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

# Cache global de session pour garder le modèle chargé en VRAM (FP16)
FISH_AUDIO_MODEL_CACHE = None
FISH_AUDIO_DECODE_FN = None
FISH_AUDIO_DECODER_CACHE = None
FISH_AUDIO_PROMPT_CACHE = {}


def load_direct_codec(codec_checkpoint_path, device, precision=torch.float16):
    """Instanciation directe du codec DAC Fish Audio en pur Python sans dépendre d'Hydra."""
    try:
        from fish_speech.models.dac.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
        from fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize
    except ImportError:
        from fish_speech.fish_speech.models.dac.modded_dac import DAC, ModelArgs, WindowLimitedTransformer
        from fish_speech.fish_speech.models.dac.rvq import DownsampleResidualVectorQuantize

    from functools import partial
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    transformer_general_config = partial(
        ModelArgs,
        block_size=8192,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )

    post_cfg = ModelArgs(
        block_size=2048,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    post_module = WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=post_cfg,
    )
    pre_cfg = ModelArgs(
        block_size=2048,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    pre_module = WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=pre_cfg,
    )
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=(2, 2),
        post_module=post_module,
        pre_module=pre_module,
        semantic_codebook_size=4096,
    )
    codec = DAC(
        sample_rate=44100,
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=transformer_general_config,
        quantizer=quantizer,
    )

    state_dict = torch.load(
        codec_checkpoint_path, map_location="cpu", mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }
    rebuilt_buffers = {
        f"{name}.{buffer}" if name else buffer
        for name, module in codec.named_modules()
        for buffer in module._non_persistent_buffers_set
    }
    state_dict = {k: v for k, v in state_dict.items() if k not in rebuilt_buffers}
    codec.load_state_dict(state_dict, strict=True)
    codec.eval()
    for module in list(codec.modules()):
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
        elif hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    codec.to(device=device, dtype=precision)
    return codec


class FishAudioProvider(BaseTTSProvider):

    def __init__(self):
        super().__init__()
        # Détection du GPU et forçage de la précision FP16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Dossier de stockage local des poids du modèle s2-pro dans MKtranslate
        self.local_dir = os.path.join(_base_dir, "weights", "FishAudio")
        self.api_url = os.getenv("FISH_API_URL", "http://127.0.0.1:8080/v1/tts")
        self.api_key = os.getenv("FISH_API_KEY", "").strip()

    def get_language_id(self, target_lang):
        """Mappe la langue de SoniTranslate vers le code ISO reconnu par Fish Audio S2-Pro."""
        clean_lang = str(target_lang).lower().strip()
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
        return mapping.get(clean_lang, clean_lang[:2] if len(clean_lang) >= 2 else "en")

    def load_local_model(self):
        """Charge le modèle Dual-AR Fish Audio S2-Pro et le codec DAC en mémoire vive en FP16."""
        global FISH_AUDIO_MODEL_CACHE, FISH_AUDIO_DECODE_FN, FISH_AUDIO_DECODER_CACHE
        if FISH_AUDIO_MODEL_CACHE is not None and FISH_AUDIO_DECODER_CACHE is not None:
            return FISH_AUDIO_MODEL_CACHE, FISH_AUDIO_DECODER_CACHE, FISH_AUDIO_DECODE_FN

        self.logger.info(f"Fish Audio S2-Pro : Initialisation locale en FP16 sur '{self.device}'...")
        os.makedirs(self.local_dir, exist_ok=True)

        codec_file = os.path.join(self.local_dir, "codec.pth")
        if not os.path.exists(codec_file) or not os.listdir(self.local_dir):
            self.logger.info("Téléchargement des poids officiels Fish Audio S2-Pro depuis Hugging Face...")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="fishaudio/s2-pro",
                local_dir=self.local_dir,
                local_dir_use_symlinks=False
            )

        # 1. Chargement du modèle DualAR / Transformer en FP16 via init_model officiel
        if FISH_AUDIO_MODEL_CACHE is None or FISH_AUDIO_DECODE_FN is None:
            try:
                from fish_speech.models.text2semantic.inference import init_model
                self.logger.info(f"Chargement du transformer Dual-AR S2-Pro ({self.dtype})...")
                model, decode_one_token = init_model(
                    checkpoint_path=self.local_dir,
                    device=self.device,
                    precision=self.dtype,
                    compile=False
                )
                FISH_AUDIO_MODEL_CACHE = model
                FISH_AUDIO_DECODE_FN = decode_one_token
                self.logger.info("Modèle Dual-AR S2-Pro chargé avec succès.")
            except Exception as e_model:
                self.logger.warning(f"Impossible de charger le modèle via fish_speech ({e_model}).")
                FISH_AUDIO_MODEL_CACHE = None
                FISH_AUDIO_DECODE_FN = None

        # 2. Chargement du décodeur Codec DAC en FP16

        if FISH_AUDIO_DECODER_CACHE is None:
            if os.path.exists(codec_file):
                self.logger.info(f"Chargement du codec DAC S2-Pro ({self.dtype})...")
                codec = None
                try:
                    from fish_speech.models.text2semantic.inference import load_codec_model
                    codec = load_codec_model(
                        codec_checkpoint_path=codec_file,
                        device=self.device,
                        precision=self.dtype
                    )
                except Exception as e_hydra:
                    self.logger.info(f"Basculement sur l'instanciation directe du codec DAC ({e_hydra})...")
                    try:
                        codec = load_direct_codec(
                            codec_checkpoint_path=codec_file,
                            device=self.device,
                            precision=self.dtype
                        )
                    except Exception as e_direct:
                        import traceback
                        self.logger.warning(f"Impossible de charger le décodeur codec ({e_direct}): {traceback.format_exc()}")
                        codec = None

                if codec is not None:
                    FISH_AUDIO_DECODER_CACHE = codec
                    self.logger.info("Codec DAC chargé avec succès.")
                else:
                    FISH_AUDIO_DECODER_CACHE = None

        return FISH_AUDIO_MODEL_CACHE, FISH_AUDIO_DECODER_CACHE, FISH_AUDIO_DECODE_FN


    def find_reference_materials(self, voice_name):
        """Localise le fichier audio de référence et son texte associé dans la voice library."""
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
            raise FileNotFoundError(f"Reference audio not found for Fish Audio voice: {voice_name}")

        if ref_txt_path and os.path.exists(ref_txt_path):
            with open(ref_txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    ref_text = content

        return ref_audio, ref_text, ref_txt_path

    def _generate_via_sdk(self, text, ref_audio, output_file):
        """Génération ultra-rapide via le SDK officiel Fish Audio (si FISH_API_KEY renseignée)."""
        from fishaudio import FishAudio
        from fishaudio.utils import save
        
        client = FishAudio(api_key=self.api_key)
        self.logger.info("Fish Audio : Synthèse via Cloud API (s2-pro)...")
        audio_stream = client.tts.convert(
            text=text,
            model="s2-pro",
            format="wav"
        )
        save(audio_stream, output_file)
        return True

    def _generate_via_http_server(self, text, ref_audio, ref_text, output_file):
        """Génération via le serveur API local Fish Audio (tools/api_server.py)."""
        import requests
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "text": text,
            "format": "wav",
            "streaming": False,
        }
        if ref_audio and os.path.exists(ref_audio):
            try:
                import base64
                with open(ref_audio, "rb") as fa:
                    audio_b64 = base64.b64encode(fa.read()).decode("utf-8")
                payload["references"] = [{"audio": audio_b64, "text": ref_text or ""}]
            except Exception as e_ref:
                self.logger.debug(f"Impossible d'encoder le fichier de référence: {e_ref}")

        response = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            return True
        return False

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        """Synthèse vocale Fish Audio S2-Pro en précision FP16."""
        # Résolution du nom de voix
        voice_name = voice.split("/")[-1] if "/" in voice else voice
        ref_audio, ref_text, ref_txt_path = self.find_reference_materials(voice_name)
        lang_id = self.get_language_id(target_lang)

        # Seed déterministe pour stabilité absolue du timbre vocal
        seed = int(kwargs.get("seed", os.getenv("FISH_SEED", 42)))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        clean_text = text.strip()
        sample_rate = 44100  # Fréquence haute-fidélité native Fish Audio S2-Pro

        # Micro-silence si texte vide ou composé uniquement de ponctuations
        if not any(c.isalnum() for c in clean_text):
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            silence_arr = np.zeros(int(sample_rate * 0.4), dtype=np.float32)
            sf.write(output_file, silence_arr, sample_rate)
            return

        # Ponctuation terminale protectrice pour clore l'auto-régression
        if clean_text[-1] not in ('.', '!', '?', ':', ';', '…'):
            clean_text += '.'

        # 1. Priorité A : Clé API Fish Audio Cloud (si configurée)
        if self.api_key:
            try:
                if self._generate_via_sdk(clean_text, ref_audio, output_file):
                    return
            except Exception as e_sdk:
                self.logger.warning(f"Fish Audio SDK API error ({e_sdk}), basculement sur le moteur local...")

        # 2. Priorité B : Serveur API local si actif (ex: sur port 8080)
        try:
            if self._generate_via_http_server(clean_text, ref_audio, ref_text, output_file):
                return
        except Exception:
            pass

        # 3. Priorité C : Pipeline local direct en FP16
        try:
            model, decoder, decode_one_token = self.load_local_model()
            if model is not None and decoder is not None:
                from fish_speech.models.text2semantic.inference import (
                    encode_audio,
                    decode_to_audio,
                    generate_long,
                )
                
                self.logger.info(f"Fish Audio S2-Pro : Synthèse FP16 de '{clean_text[:40]}...'")
                
                # Encodage ou récupération en cache des tokens acoustiques de référence
                global FISH_AUDIO_PROMPT_CACHE
                prompt_tokens = FISH_AUDIO_PROMPT_CACHE.get(ref_audio)
                if prompt_tokens is None:
                    prompt_tokens = encode_audio(ref_audio, decoder, self.device)
                    FISH_AUDIO_PROMPT_CACHE[ref_audio] = prompt_tokens

                # Génération des codes sémantiques Dual-AR en FP16
                with torch.inference_mode():
                    generator = generate_long(
                        model=model,
                        device=self.device,
                        decode_one_token=decode_one_token,
                        text=clean_text,
                        prompt_text=ref_text or "Voix de reference.",
                        prompt_tokens=[prompt_tokens] if isinstance(prompt_tokens, torch.Tensor) else prompt_tokens,
                        num_samples=1,
                        max_new_tokens=int(kwargs.get("max_new_tokens", 2048)),
                        top_p=float(kwargs.get("top_p", 0.7)),
                        top_k=int(kwargs.get("top_k", 30)),
                        temperature=float(kwargs.get("temperature", 0.7)),
                        compile=False,
                    )
                    
                    audio_chunks = []
                    for resp in generator:
                        if resp.action == "sample" and resp.codes is not None and resp.codes.size(1) > 0:
                            chunk = decode_to_audio(resp.codes.to(self.device), decoder)
                            audio_chunks.append(chunk.detach().cpu().float().numpy())
                    
                    if audio_chunks:
                        wav_out = np.concatenate(audio_chunks, axis=-1)
                    else:
                        wav_out = np.array([], dtype=np.float32)

                # Normalisation anti-clipping
                if wav_out is not None and getattr(wav_out, "size", len(wav_out)) > 0:
                    max_val = float(np.abs(wav_out).max())
                    if max_val > 1.0:
                        wav_out = wav_out / max_val * 0.98
                    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                    sf.write(output_file, wav_out, sample_rate)
                    return True
        except Exception as e_local:
            import traceback
            self.logger.error(f"Fish Audio Local FP16 Generation Error: {str(e_local)}\n{traceback.format_exc()}")


        # 4. Filet de sécurité anti-crash (silence propre)
        self.logger.warning(f"Fish Audio : Création d'un segment de sécurité pour '{clean_text[:30]}'")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        seg_duration = kwargs.get("seg_duration", 1.0) or 1.0
        silence_arr = np.zeros(int(sample_rate * float(seg_duration)), dtype=np.float32)
        sf.write(output_file, silence_arr, sample_rate)
        return False
