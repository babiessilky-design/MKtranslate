from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch
import gc
import os
import json
import re
import time
import soundfile as sf
from IPython.utils import capture # noqa
from .language_configuration import EXTRA_ALIGN, INVERTED_LANGUAGES
from .logging_setup import logger
from .postprocessor import sanitize_file_name
from .utils import remove_directory_contents, run_command

ASR_MODEL_OPTIONS = [
    "Gemini 3.5 Transcribe",
    "large-v3",
    "large-v2",
    "medium",
    "small",
    "base",
    "tiny",
    "distil-large-v2",
    "Systran/faster-distil-whisper-large-v3",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "distil-small.en",
    "distil-medium.en",
    "OpenAI_API_Whisper",
]

COMPUTE_TYPE_GPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int8_float16",
    "int8_bfloat16",
    "float16",
    "bfloat16",
    "float32"
]

COMPUTE_TYPE_CPU = [
    "default",
    "auto",
    "int8",
    "int8_float32",
    "int16",
    "float32",
]

WHISPER_MODELS_PATH = './WHISPER_MODELS'


def openai_api_whisper(
    input_audio_file,
    source_lang=None,
    chunk_duration=1800
):

    info = sf.info(input_audio_file)
    duration = info.duration

    output_directory = "./whisper_api_audio_parts"
    os.makedirs(output_directory, exist_ok=True)
    remove_directory_contents(output_directory)

    if duration > chunk_duration:
        # Split the audio file into smaller chunks with 30-minute duration
        cm = f'ffmpeg -i "{input_audio_file}" -f segment -segment_time {chunk_duration} -c:a libvorbis "{output_directory}/output%03d.ogg"'
        run_command(cm)
        # Get list of generated chunk files
        chunk_files = sorted(
            [f"{output_directory}/{f}" for f in os.listdir(output_directory) if f.endswith('.ogg')]
        )
    else:
        one_file = f"{output_directory}/output000.ogg"
        cm = f'ffmpeg -i "{input_audio_file}" -c:a libvorbis {one_file}'
        run_command(cm)
        chunk_files = [one_file]

    # Transcript
    segments = []
    language = source_lang if source_lang else None
    for i, chunk in enumerate(chunk_files):
        from openai import OpenAI
        client = OpenAI()

        audio_file = open(chunk, "rb")
        transcription = client.audio.transcriptions.create(
          model="whisper-1",
          file=audio_file,
          language=language,
          response_format="verbose_json",
          timestamp_granularities=["segment"],
        )

        try:
            transcript_dict = transcription.model_dump()
        except: # noqa
            transcript_dict = transcription.to_dict()

        if language is None:
            logger.info(f'Language detected: {transcript_dict["language"]}')
            language = TO_LANGUAGE_CODE[transcript_dict["language"]]

        chunk_time = chunk_duration * (i)

        for seg in transcript_dict["segments"]:

            if "start" in seg.keys():
                segments.append(
                    {
                        "text": seg["text"],
                        "start": seg["start"] + chunk_time,
                        "end": seg["end"] + chunk_time,
                    }
                )

    audio = whisperx.load_audio(input_audio_file)
    result = {"segments": segments, "language": language}

    return audio, result


def parse_gemini_transcription(text_resp, default_lang="en", audio_duration=None, max_speakers=None):
    """
    Parse la sortie textuelle de Gemini (JSON, markdown, ou format horodaté)
    et extrait les segments avec 'start', 'end', 'text', 'speaker' (SPEAKER_00, SPEAKER_01...).
    """
    text_resp = (text_resp or "").strip()
    if not text_resp:
        return [], default_lang

    if max_speakers is None:
        try:
            max_spk_val = int(os.environ.get("MAX_SPEAKERS", 0))
        except (ValueError, TypeError):
            max_spk_val = 0
    else:
        try:
            max_spk_val = int(max_speakers)
        except (ValueError, TypeError):
            max_spk_val = 0

    # 1. Tentatives d'extraction et de parsing JSON
    json_candidates = []

    # Nettoyage direct des balises markdown ```json ... ```
    clean_md = text_resp
    if "```" in clean_md:
        clean_md = re.sub(r"^```(?:json)?\s*", "", clean_md, flags=re.MULTILINE)
        clean_md = re.sub(r"\s*```$", "", clean_md, flags=re.MULTILINE)
        clean_md = clean_md.strip()
        json_candidates.append(clean_md)

    json_candidates.append(text_resp)

    # Recherche d'un objet JSON contenant "segments"
    m_obj = re.search(r'\{[\s\S]*"segments"[\s\S]*\}', text_resp)
    if m_obj:
        json_candidates.append(m_obj.group(0))

    # Recherche d'une liste JSON [...]
    m_list = re.search(r'\[[\s\S]*\]', text_resp)
    if m_list:
        json_candidates.append(m_list.group(0))

    parsed_data = None
    for cand in json_candidates:
        try:
            parsed_data = json.loads(cand)
            if isinstance(parsed_data, (dict, list)):
                break
        except Exception:
            continue

    raw_segments = []
    detected_lang = default_lang

    if isinstance(parsed_data, dict):
        raw_segments = parsed_data.get("segments", [])
        detected_lang = parsed_data.get("language", default_lang)
    elif isinstance(parsed_data, list):
        raw_segments = parsed_data

    formatted_segments = []
    if raw_segments:
        for seg in raw_segments:
            if not isinstance(seg, dict):
                continue
            if "start" in seg and "end" in seg and "text" in seg:
                try:
                    start_val = float(seg["start"])
                    end_val = float(seg["end"])
                except (ValueError, TypeError):
                    continue
                if end_val <= start_val:
                    end_val = start_val + 0.5
                spk_raw = seg.get("speaker", "SPEAKER_00")
                spk_nums = re.findall(r'\d+', str(spk_raw))
                spk_id = int(spk_nums[0]) if spk_nums else 0
                if max_spk_val > 0 and spk_id >= max_spk_val:
                    spk_id = max_spk_val - 1
                spk = f"SPEAKER_{spk_id:02d}"

                text_val = str(seg["text"]).strip()
                if text_val:
                    formatted_segments.append({
                        "start": round(start_val, 2),
                        "end": round(end_val, 2),
                        "text": text_val,
                        "speaker": spk
                    })

    # 2. Si aucun segment JSON valide, parser les lignes horodatées (ex: [00:01.25 -> 00:04.50] SPEAKER_00: Texte)
    if not formatted_segments:
        logger.info("Parsing ligne par ligne de la transcription Gemini...")
        time_pattern = re.compile(
            r'(?:\[|\()?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(?:-->|->|-)\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(?:\]|\))?'
            r'(?:\s*\[?(?:SPEAKER|Speaker|locuteur|Intervenant)?\s*[_#:\-]?\s*(\d+)\]?)?'
            r'[:\s]+(.*)',
            re.IGNORECASE
        )

        def parse_time(s):
            s = s.strip().replace(",", ".")
            parts = s.split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(s)

        for line in text_resp.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = time_pattern.match(line)
            if m:
                try:
                    t_start = parse_time(m.group(1))
                    t_end = parse_time(m.group(2))
                    spk_idx = m.group(3)
                    spk_id = int(spk_idx) if spk_idx else 0
                    if max_spk_val > 0 and spk_id >= max_spk_val:
                        spk_id = max_spk_val - 1
                    spk = f"SPEAKER_{spk_id:02d}"
                    txt = m.group(4).strip()
                    if t_end <= t_start:
                        t_end = t_start + 0.5
                    if txt:
                        formatted_segments.append({
                            "start": round(t_start, 2),
                            "end": round(t_end, 2),
                            "text": txt,
                            "speaker": spk
                        })
                except Exception:
                    continue

    # 3. Fallback si texte brut sans horodatage
    if not formatted_segments and text_resp:
        logger.warning("Gemini a renvoyé du texte brut. Synthèse des segments...")
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text_resp) if s.strip()]
        total_dur = audio_duration if audio_duration and audio_duration > 0 else (len(sentences) * 3.0)
        dur_per_s = total_dur / max(len(sentences), 1)
        cur_time = 0.0
        for s in sentences:
            end_time = min(cur_time + dur_per_s, total_dur)
            formatted_segments.append({
                "start": round(cur_time, 2),
                "end": round(end_time, 2),
                "text": s,
                "speaker": "SPEAKER_00"
            })
            cur_time = end_time

    return formatted_segments, detected_lang


def gemini_api_transcribe(
    input_audio_file,
    source_lang=None,
    model_name="gemini-3.8-flash",
    max_speakers=None,
):
    """
    Transcrit l'audio et effectue la diarisation nativement avec Google Gemini.
    Renvoie audio, result {"segments": [...], "language": ...}
    Chaque segment contient: start, end, text, speaker
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("google_api_key")
    if not api_key:
        raise ValueError("Erreur : La variable d'environnement GOOGLE_API_KEY est manquante pour utiliser Gemini Transcribe.")

    logger.info(f"Gemini Transcribe : Traitement du fichier '{input_audio_file}'...")

    audio = None
    audio_duration = None
    try:
        audio = whisperx.load_audio(input_audio_file)
        if audio is not None and len(audio) > 0:
            audio_duration = len(audio) / 16000.0
    except Exception:
        audio = None
        try:
            import soundfile as sf
            info = sf.info(input_audio_file)
            audio_duration = info.duration
        except Exception:
            pass

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Upload du fichier audio via Files API de Gemini
    uploaded_file = client.files.upload(file=input_audio_file)

    # Attente si le fichier est en cours de traitement
    try:
        if hasattr(uploaded_file, "state"):
            while getattr(uploaded_file.state, "name", str(uploaded_file.state)) == "PROCESSING":
                logger.info("En attente du traitement du fichier audio par Gemini...")
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
            if getattr(uploaded_file.state, "name", str(uploaded_file.state)) == "FAILED":
                raise ValueError("Le traitement du fichier audio par Gemini a échoué.")
    except Exception as e_poll:
        logger.debug(f"Vérification de l'état du fichier ignorée ({e_poll})")

    try:
        # Détection dynamique des modèles actifs sur la clé de l'utilisateur
        available_api_models = []
        try:
            if hasattr(client, "models") and hasattr(client.models, "list"):
                for m in client.models.list():
                    m_name = getattr(m, "name", "") or ""
                    clean_m = m_name.replace("models/", "")
                    if "2.0" not in clean_m:
                        available_api_models.append(clean_m)
        except Exception as e_list:
            logger.debug(f"Impossible de lister les modèles API ({e_list})")

        candidate_models = [
            os.environ.get("GEMINI_TRANSCRIBE_MODEL", model_name),
            "gemini-3.8-flash",
            "gemini-3.5-transcribe",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
            "gemini-3.8-pro",
            "gemini-2.5-pro",
            "gemini-1.5-pro",
        ]
        for m in available_api_models:
            if ("flash" in m.lower() or "transcribe" in m.lower()) and "2.0" not in m:
                if m not in candidate_models:
                    candidate_models.append(m)

        unique_models = []
        for m in candidate_models:
            if m and m not in unique_models and "2.0" not in m:
                unique_models.append(m)

        lang_instruction = (
            f"The source language is {source_lang}."
            if source_lang and source_lang != "Automatic detection"
            else "Detect the primary spoken language and set 'language' to its ISO-639-1 code."
        )

        max_spk = None
        if max_speakers is not None:
            try:
                max_spk = int(max_speakers)
            except (ValueError, TypeError):
                max_spk = None
        if max_spk is None:
            try:
                env_spk = os.environ.get("MAX_SPEAKERS")
                if env_spk:
                    max_spk = int(env_spk)
            except (ValueError, TypeError):
                max_spk = None

        if max_spk and max_spk > 0:
            if max_spk == 1:
                rule_speakers = "1. There is at most 1 speaker in this audio. Label all speech strictly as SPEAKER_00."
            else:
                last_spk = f"SPEAKER_{max_spk - 1:02d}"
                rule_speakers = (
                    f"1. There are at most {max_spk} speakers in this audio. "
                    f"Label them strictly among SPEAKER_00 to {last_spk}."
                )
        else:
            rule_speakers = "1. Identify all distinct speakers and label them strictly as SPEAKER_00, SPEAKER_01, SPEAKER_02, etc. (0-indexed)."

        prompt = f"""You are an expert speech recognition and speaker diarization engine.
Transcribe the provided audio accurately.
{lang_instruction}

CRITICAL RULES:
{rule_speakers}
2. Segment speech into natural, coherent sentences or dialogue phrases (ideal for subtitles and video dubbing).
3. For each segment, output:
   - "start": start time in seconds (float, e.g. 1.25)
   - "end": end time in seconds (float, e.g. 3.40)
   - "text": the transcribed text in the original spoken language.
   - "speaker": speaker ID (e.g. "SPEAKER_00", "SPEAKER_01").
4. Do NOT translate. Keep the original spoken words.
5. Filter out filler sounds (like "um", "uh", "euh", stuttering).
6. Return ONLY valid JSON in this exact structure:
{{
  "language": "en",
  "segments": [
    {{
      "start": 0.0,
      "end": 2.5,
      "text": "First sentence here.",
      "speaker": "SPEAKER_00"
    }}
  ]
}}"""

        response = None
        last_error = None
        response_text = ""

        def extract_text(resp):
            if not resp:
                return ""
            if isinstance(resp, str):
                return resp
            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text
            if hasattr(resp, "text") and resp.text:
                return resp.text
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    parts = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
                    if parts:
                        return "\n".join(parts)
            return ""

        for mod in unique_models:
            try:
                logger.info(f"Gemini Transcribe : Tentative avec le modèle '{mod}'...")
                response_text = ""

                # 1. Priorité absolue : Interactions API (Recommandé officiellement par Google pour gemini-3.8-flash)
                if hasattr(client, "interactions") and hasattr(client.interactions, "create"):
                    try:
                        inter_input = [
                            {"type": "text", "text": prompt},
                            {
                                "type": "audio",
                                "uri": getattr(uploaded_file, "uri", None) or getattr(uploaded_file, "name", None),
                                "mime_type": getattr(uploaded_file, "mime_type", "audio/wav") or "audio/wav",
                            }
                        ]
                        interaction = client.interactions.create(
                            model=mod,
                            input=inter_input
                        )
                        response_text = extract_text(interaction)
                    except Exception as err_inter:
                        logger.debug(f"Interactions API non disponible pour '{mod}' ({err_inter}), essai generate_content...")

                # 2. Repli : client.models.generate_content
                if not response_text.strip():
                    if "transcribe" in mod.lower():
                        # Modèle dédié de transcription (mode audio direct)
                        logger.info(f"Mode audio direct pour '{mod}'...")
                        try:
                            response = client.models.generate_content(
                                model=mod,
                                contents=[uploaded_file, prompt]
                            )
                        except Exception as err_prompt:
                            logger.warning(f"Appel avec prompt sur '{mod}' a échoué ({err_prompt}), tentative audio seul...")
                            response = client.models.generate_content(
                                model=mod,
                                contents=[uploaded_file]
                            )
                    else:
                        # Modèles multimodaux généraux (gemini-3.8-flash, etc.)
                        try:
                            config = types.GenerateContentConfig(
                                response_mime_type="application/json",
                                temperature=0.1,
                            )
                            response = client.models.generate_content(
                                model=mod,
                                contents=[uploaded_file, prompt],
                                config=config
                            )
                        except Exception as err_json:
                            logger.warning(f"Mode JSON refusé pour '{mod}' ({err_json}), tentative sans config JSON...")
                            response = client.models.generate_content(
                                model=mod,
                                contents=[uploaded_file, prompt]
                            )
                    response_text = extract_text(response)

                if response_text.strip():
                    logger.info(f"Gemini Transcribe : Succès avec le modèle '{mod}' !")
                    break
                else:
                    logger.warning(f"Modèle '{mod}' a renvoyé une réponse vide.")

            except Exception as e:
                logger.warning(f"Modèle '{mod}' indisponible ({e}). Bascule sur le modèle suivant...")
                last_error = e

        if not response_text.strip():
            raise RuntimeError(f"Échec de la transcription Gemini sur tous les modèles: {last_error}")

        formatted_segments, detected_lang = parse_gemini_transcription(
            response_text,
            default_lang=source_lang or "en",
            audio_duration=audio_duration,
            max_speakers=max_spk,
        )

        if not formatted_segments:
            raise ValueError(f"Impossible d'extraire des segments de la réponse Gemini. Réponse brute: {response_text[:300]}")

        logger.info(f"Gemini Transcribe : {len(formatted_segments)} segments transcrits et diarizés avec succès.")
        result = {
            "segments": formatted_segments,
            "language": detected_lang
        }
        return audio, result

    finally:
        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass


def find_whisper_models():
    path = WHISPER_MODELS_PATH
    folders = []

    if os.path.exists(path):
        for folder in os.listdir(path):
            folder_path = os.path.join(path, folder)
            if (
                os.path.isdir(folder_path)
                and 'model.bin' in os.listdir(folder_path)
            ):
                folders.append(folder)
    return folders


def transcribe_speech(
    audio_wav,
    asr_model,
    compute_type,
    batch_size,
    SOURCE_LANGUAGE,
    literalize_numbers=True,
    segment_duration_limit=15,
    max_speakers=None,
):
    """
    Transcribe speech using a whisper model.

    Parameters:
    - audio_wav (str): Path to the audio file in WAV format.
    - asr_model (str): The whisper model to be loaded.
    - compute_type (str): Type of compute to be used (e.g., 'int8', 'float16').
    - batch_size (int): Batch size for transcription.
    - SOURCE_LANGUAGE (str): Source language for transcription.

    Returns:
    - Tuple containing:
        - audio: Loaded audio file.
        - result: Transcription result as a dictionary.
    """

    if asr_model == "Gemini 3.5 Transcribe":
        return gemini_api_transcribe(audio_wav, SOURCE_LANGUAGE, max_speakers=max_speakers)

    if asr_model == "OpenAI_API_Whisper":
        if literalize_numbers:
            logger.info(
                "OpenAI's API Whisper does not support "
                "the literalization of numbers."
            )
        return openai_api_whisper(audio_wav, SOURCE_LANGUAGE)

    # https://github.com/openai/whisper/discussions/277
    prompt = "以下是普通话的句子。" if SOURCE_LANGUAGE == "zh" else None
    SOURCE_LANGUAGE = (
        SOURCE_LANGUAGE if SOURCE_LANGUAGE != "zh-TW" else "zh"
    )
    asr_options = {
        "initial_prompt": prompt,
        "suppress_numerals": literalize_numbers
    }

    if asr_model not in ASR_MODEL_OPTIONS:

        base_dir = WHISPER_MODELS_PATH
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        model_dir = os.path.join(base_dir, sanitize_file_name(asr_model))

        if not os.path.exists(model_dir):
            from ctranslate2.converters import TransformersConverter

            quantization = "float32"
            # Download new model
            try:
                converter = TransformersConverter(
                    asr_model,
                    low_cpu_mem_usage=True,
                    copy_files=[
                        "tokenizer_config.json", "preprocessor_config.json"
                    ]
                )
                converter.convert(
                    model_dir,
                    quantization=quantization,
                    force=False
                )
            except Exception as error:
                if "File tokenizer_config.json does not exist" in str(error):
                    converter._copy_files = [
                        "tokenizer.json", "preprocessor_config.json"
                    ]
                    converter.convert(
                        model_dir,
                        quantization=quantization,
                        force=True
                    )
                else:
                    raise error

        asr_model = model_dir
        logger.info(f"ASR Model: {str(model_dir)}")

    model = whisperx.load_model(
        asr_model,
        os.environ.get("SONITR_DEVICE"),
        compute_type=compute_type,
        language=SOURCE_LANGUAGE,
        asr_options=asr_options,
    )

    audio = whisperx.load_audio(audio_wav)
    result = model.transcribe(
        audio,
        batch_size=batch_size,
        chunk_size=segment_duration_limit,
        print_progress=True,
    )

    if result["language"] == "zh" and not prompt:
        result["language"] = "zh-TW"
        logger.info("Chinese - Traditional (zh-TW)")

    del model
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return audio, result


def align_speech(audio, result):
    """
    Aligns speech segments based on the provided audio and result metadata.

    Parameters:
    - audio (array): The audio data in a suitable format for alignment.
    - result (dict): Metadata containing information about the segments
         and language.

    Returns:
    - result (dict): Updated metadata after aligning the segments with
        the audio. This includes character-level alignments if
        'return_char_alignments' is set to True.

    Notes:
    - This function uses language-specific models to align speech segments.
    - It performs language compatibility checks and selects the
        appropriate alignment model.
    - Cleans up memory by releasing resources after alignment.
    """
    DAMHF.update(DAMT)  # lang align
    if (
        not result["language"] in DAMHF.keys()
        and not result["language"] in EXTRA_ALIGN.keys()
    ):
        logger.warning(
            "Automatic detection: Source language not compatible with align"
        )
        raise ValueError(
            f"Detected language {result['language']}  incompatible, "
            "you can select the source language to avoid this error."
        )
    if (
        result["language"] in EXTRA_ALIGN.keys()
        and EXTRA_ALIGN[result["language"]] == ""
    ):
        lang_name = (
            INVERTED_LANGUAGES[result["language"]]
            if result["language"] in INVERTED_LANGUAGES.keys()
            else result["language"]
        )
        logger.warning(
            "No compatible wav2vec2 model found "
            f"for the language '{lang_name}', skipping alignment."
        )
        return result

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=os.environ.get("SONITR_DEVICE"),
        model_name=None
        if result["language"] in DAMHF.keys()
        else EXTRA_ALIGN[result["language"]],
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        os.environ.get("SONITR_DEVICE"),
        return_char_alignments=True,
        print_progress=False,
    )
    del model_a
    gc.collect()
    torch.cuda.empty_cache()  # noqa
    return result


diarization_models = {
    "pyannote_3.1": "pyannote/speaker-diarization-3.1",
    "pyannote_2.1": "pyannote/speaker-diarization@2.1",
    "disable": "",
}


def reencode_speakers(result, max_speakers=None):
    if max_speakers is None:
        try:
            max_spk_val = int(os.environ.get("MAX_SPEAKERS", 0))
        except (ValueError, TypeError):
            max_spk_val = 0
    else:
        try:
            max_spk_val = int(max_speakers)
        except (ValueError, TypeError):
            max_spk_val = 0

    speaker_mapping = {}
    counter = 0
    
    # On enlève la vérification du début pour forcer le nettoyage
    # logger.debug("Reencode speakers") (Optionnel)

    for segment in result["segments"]:
        # On utilise .get() pour éviter le crash si la clé manque
        old_speaker = segment.get("speaker", "SPEAKER_00")
        
        if old_speaker not in speaker_mapping:
            if max_spk_val > 0 and counter >= max_spk_val:
                speaker_mapping[old_speaker] = f"SPEAKER_{max_spk_val - 1:02d}"
            else:
                speaker_mapping[old_speaker] = f"SPEAKER_{counter:02d}"
                counter += 1
            
        segment["speaker"] = speaker_mapping[old_speaker]

    return result

def diarize_speech(
    audio_wav,
    result,
    min_speakers,
    max_speakers,
    YOUR_HF_TOKEN,
    model_name="pyannote/speaker-diarization@2.1",
):
    import os
    import gc
    import torch
    import whisperx
    from soni_translate.speech_segmentation import diarization_models

    # On nettoie le token au cas où
    if YOUR_HF_TOKEN:
        YOUR_HF_TOKEN = str(YOUR_HF_TOKEN).strip()

    if max(min_speakers, max_speakers) > 1:
        diarize_model = None
        
        # --- TENTATIVE 1 : OFFICIELLE (Avec Token) ---
        try:
            print(">> Tentative 1 : Connexion officielle Pyannote...")
            diarize_model = whisperx.DiarizationPipeline(
                model_name=model_name,
                use_auth_token=YOUR_HF_TOKEN,
                device=os.environ.get("SONITR_DEVICE"),
            )
        except Exception as e:
            print(f">> Echec Tentative 1 ({e}).")
            
            # --- TENTATIVE 2 : LA SOLUTION MAGIQUE (Sans Token) ---
            print(">> Tentative 2 : Passage au modèle PUBLIC (fatymatariq)... 🏴‍☠️")
            try:
                # On force le modèle public qui ne demande pas de token
                # Note: On utilise le modèle 3.1 public car il est meilleur
                public_model = "fatymatariq/speaker-diarization-3.1"
                
                diarize_model = whisperx.DiarizationPipeline(
                    model_name=public_model,
                    use_auth_token=None, # Pas besoin de token !
                    device=os.environ.get("SONITR_DEVICE"),
                )
                print(">> SUCCÈS ! Modèle public chargé.")
            except Exception as e2:
                # Si même ça échoue, on lève la vraie erreur
                raise ValueError(f"Tout a échoué. Erreur finale: {e2}")

        # La suite est identique au code original...
        diarize_segments = diarize_model(
            audio_wav, min_speakers=min_speakers, max_speakers=max_speakers
        )

        result_diarize = whisperx.assign_word_speakers(
            diarize_segments, result
        )

        for segment in result_diarize["segments"]:
            if "speaker" not in segment:
                segment["speaker"] = "SPEAKER_00"
        
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        result_diarize = result
        result_diarize["segments"] = [
            {**item, "speaker": "SPEAKER_00"}
            for item in result_diarize["segments"]
        ]
        # ... fin du code de diarisation ...

        # --- DEBUG A AJOUTER ICI ---
        liste_speakers = set()
        for seg in result_diarize["segments"]:
            if "speaker" in seg:
                liste_speakers.add(seg["speaker"])
        
        print(f"\n📢 BILAN IA : {len(liste_speakers)} voix détectées.")
        print(f"📢 QUI SONT-ILS ? : {liste_speakers}")
        # ---------------------------
    
    return reencode_speakers(result_diarize)
