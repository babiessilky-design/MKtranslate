from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch
import numpy as np
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

    speaker_map = {}
    speaker_counter = 0

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

                spk_raw = str(seg.get("speaker", "")).strip()
                if not spk_raw:
                    spk_raw = "SPEAKER_00"

                if spk_raw not in speaker_map:
                    if max_spk_val > 0 and speaker_counter >= max_spk_val:
                        speaker_map[spk_raw] = f"SPEAKER_{max_spk_val - 1:02d}"
                    else:
                        speaker_map[spk_raw] = f"SPEAKER_{speaker_counter:02d}"
                        speaker_counter += 1
                spk = speaker_map[spk_raw]

                text_val = str(seg["text"]).strip()
                if text_val:
                    formatted_segments.append({
                        "start": round(start_val, 2),
                        "end": round(end_val, 2),
                        "text": text_val,
                        "speaker": spk
                    })

    # 2. Si aucun segment JSON valide, parser les lignes horodatees (ex: [00:01.25 -> 00:04.50] SPEAKER_00: Texte)
    if not formatted_segments:
        logger.info("Parsing ligne par ligne de la transcription Gemini...")
        time_pattern = re.compile(
            r'(?:\[|\()?\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(?:-->|->|-)\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(?:\]|\))?'
            r'(?:\s*\[?(?:SPEAKER|Speaker|locuteur|Intervenant)?\s*[_#:\-]?\s*([a-zA-Z0-9_\-]+)\]?)?'
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
                    spk_raw = m.group(3)
                    spk_raw = str(spk_raw).strip() if spk_raw else "SPEAKER_00"
                    if spk_raw not in speaker_map:
                        if max_spk_val > 0 and speaker_counter >= max_spk_val:
                            speaker_map[spk_raw] = f"SPEAKER_{max_spk_val - 1:02d}"
                        else:
                            speaker_map[spk_raw] = f"SPEAKER_{speaker_counter:02d}"
                            speaker_counter += 1
                    spk = speaker_map[spk_raw]
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
        cur_spk_idx = 0
        for i, s in enumerate(sentences):
            end_time = min(cur_time + dur_per_s, total_dur)
            if max_spk_val > 1:
                spk = f"SPEAKER_{cur_spk_idx:02d}"
                if s.endswith("?") or s.endswith("!") or (i % 2 == 1):
                    cur_spk_idx = (cur_spk_idx + 1) % max_spk_val
            else:
                spk = "SPEAKER_00"
            formatted_segments.append({
                "start": round(cur_time, 2),
                "end": round(end_time, 2),
                "text": s,
                "speaker": spk
            })
            cur_time = end_time

    return formatted_segments, detected_lang


def force_speakers_on_segments(segments, max_speakers=2, client=None, uploaded_file=None, audio_np=None):
    """
    Force le nombre exact de locuteurs (max_speakers) sur les segments transcrits.
    1. Essaie d'abord une diarisation contextuelle LLM Gemini (ultra-rapide et sémantique).
    2. Si échec, utilise le clustering acoustique F0 (pitch Praat/PyWorld/NumPy).
    3. Si F0 plat, alterne intelligemment sur les répliques de dialogue.
    """
    if not segments or not max_speakers or max_speakers <= 1:
        return segments

    try:
        max_spk = int(max_speakers)
    except (ValueError, TypeError):
        max_spk = 2

    existing_spks = set(s.get("speaker") for s in segments if s.get("speaker"))
    if len(existing_spks) >= max_spk:
        return segments

    logger.info(f"Forçage des locuteurs : {len(segments)} segments à distribuer sur {max_spk} locuteurs (actuellement {len(existing_spks)})...")

    # --- MÉTHODE 1 : Re-diarisation contextuelle Gemini ---
    if client is not None:
        try:
            from google.genai import types
            sentences_payload = [
                {"id": i, "text": seg["text"], "time": f"{seg['start']}s-{seg['end']}s"}
                for i, seg in enumerate(segments)
            ]
            diarize_prompt = f"""You are an expert dialogue speaker diarization analyst.
The following {len(segments)} transcribed sentences are from a spoken dialogue.
There are EXACTLY {max_spk} distinct speakers in this audio: 'SPEAKER_00' and 'SPEAKER_01' (up to 'SPEAKER_{max_spk - 1:02d}').

You MUST assign each sentence ID to its speaker based on dialogue context, conversation turns, questions and answers, and dialogue flow.
MANDATORY RULES:
- You MUST use ALL {max_spk} speakers across the dialogue!
- Alternate naturally between the speakers when conversational turns change.
- Return ONLY valid JSON in this exact structure:
{{
  "speakers": ["SPEAKER_00", "SPEAKER_01", ...]
}}
The "speakers" array MUST have exactly {len(segments)} elements corresponding to the sentence IDs in order."""

            for mod_eval in ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-transcribe"]:
                try:
                    cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
                    contents = [uploaded_file, diarize_prompt] if uploaded_file else [diarize_prompt]
                    resp = client.models.generate_content(
                        model=mod_eval,
                        contents=contents,
                        config=cfg
                    )
                    txt = getattr(resp, "text", "") or ""
                    if txt:
                        data = json.loads(txt)
                        spk_list = data.get("speakers", [])
                        if isinstance(spk_list, list) and len(spk_list) == len(segments):
                            assigned_spks = set(spk_list)
                            if len(assigned_spks) >= max_spk or len(assigned_spks) > 1:
                                for idx, seg in enumerate(segments):
                                    seg["speaker"] = spk_list[idx]
                                logger.info(f"Diarisation contextuelle Gemini réussie : {len(segments)} segments répartis sur {assigned_spks} !")
                                return segments
                except Exception as e_mod:
                    logger.debug(f"Essai re-diarisation avec '{mod_eval}' ({e_mod})")
                    continue
        except Exception as e_llm:
            logger.debug(f"Re-diarisation LLM non disponible ({e_llm})")

    # --- MÉTHODE 2 : Clustering Acoustique par Pitch F0 ---
    if audio_np is not None and len(audio_np) > 0:
        try:
            from soni_translate.gender_detection import extract_f0_pitch
            import numpy as np
            segment_pitches = []
            for seg in segments:
                t0 = int(seg["start"] * 16000)
                t1 = int(seg["end"] * 16000)
                chunk = audio_np[t0:t1]
                if len(chunk) > 160:
                    f0, _ = extract_f0_pitch(chunk)
                    segment_pitches.append(f0 if f0 and f0 > 0 else None)
                else:
                    segment_pitches.append(None)

            valid_pitches = [p for p in segment_pitches if p is not None]
            if len(valid_pitches) >= max(len(segments) // 4, 2):
                median_pitch = float(np.median(valid_pitches))
                for i, seg in enumerate(segments):
                    p = segment_pitches[i]
                    if p is not None:
                        spk_idx = 1 if p > median_pitch else 0
                    else:
                        spk_idx = i % max_spk
                    seg["speaker"] = f"SPEAKER_{spk_idx:02d}"
                assigned_spks = set(s["speaker"] for s in segments)
                logger.info(f"Diarisation acoustique F0 réussie : {len(segments)} segments répartis sur {assigned_spks} !")
                return segments
        except Exception as e_acoust:
            logger.debug(f"Clustering F0 ignoré ({e_acoust})")

    # --- MÉTHODE 3 : Alternance Intelligente des Tours de Parole (Zero-Failure) ---
    logger.info("Application de l'alternance séquentielle des répliques pour garantir les locuteurs demandés...")
    current_spk = 0
    for i, seg in enumerate(segments):
        seg["speaker"] = f"SPEAKER_{current_spk:02d}"
        txt = seg.get("text", "")
        if txt.endswith("?") or txt.endswith("!") or (i % 2 == 1):
            current_spk = (current_spk + 1) % max_spk

    assigned_spks = set(s["speaker"] for s in segments)
    logger.info(f"Alternance des tours de parole appliquée : locuteurs {assigned_spks}")
    return segments


def gemini_api_transcribe(
    input_audio_file,
    source_lang=None,
    model_name="gemini-3.5-transcribe",
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
                    if not any(bad in clean_m for bad in ["1.0", "1.5", "2.0", "2.5"]):
                        available_api_models.append(clean_m)
        except Exception as e_list:
            logger.debug(f"Impossible de lister les modèles API ({e_list})")

        flash_api_models = [m for m in available_api_models if "flash" in m.lower() and "transcribe" not in m.lower()]
        pro_api_models = [m for m in available_api_models if "pro" in m.lower() and "transcribe" not in m.lower()]
        transcribe_api_models = [m for m in available_api_models if "transcribe" in m.lower()]

        candidate_models = []
        if os.environ.get("GEMINI_TRANSCRIBE_MODEL"):
            candidate_models.append(os.environ["GEMINI_TRANSCRIBE_MODEL"])

        # Priorité absolue au modèle sélectionné dans l'interface (Gemini 3.5 Transcribe)
        candidate_models.append("gemini-3.5-transcribe")

        # Modèles de secours multimodaux (Gemini 3.x) en cas d'indisponibilité
        candidate_models.extend([
            "gemini-3.8-flash",
            "gemini-3.6-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.5-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-pro",
        ])
        candidate_models.extend(transcribe_api_models)
        candidate_models.extend(flash_api_models)
        candidate_models.extend(pro_api_models)

        unique_models = []
        for m in candidate_models:
            if m and m not in unique_models and not any(bad in m for bad in ["1.0", "1.5", "2.0", "2.5"]):
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
                rule_speakers = "1. There is only 1 speaker in this audio. Label all speech strictly as 'SPEAKER_00'."
            else:
                last_spk = f"SPEAKER_{max_spk - 1:02d}"
                rule_speakers = (
                    f"1. MANDATORY SPEAKER DIARIZATION ({max_spk} SPEAKERS): There are {max_spk} distinct speakers in this audio. "
                    f"You MUST identify who is speaking each sentence and assign each segment strictly to its speaker tag from 'SPEAKER_00' to '{last_spk}'. "
                    f"Carefully distinguish speakers by their vocal pitch, gender, tone, and dialogue turns. "
                    f"CRITICAL: Do NOT put all segments on SPEAKER_00! You must alternate between speakers when different people talk."
                )
        else:
            rule_speakers = (
                "1. MANDATORY SPEAKER DIARIZATION: Identify all distinct speakers and label them strictly as SPEAKER_00, SPEAKER_01, SPEAKER_02, etc. (0-indexed). "
                "Carefully distinguish speakers by vocal pitch, gender, and conversational turns."
            )

        prompt = f"""You are an expert speech recognition and speaker diarization engine.
Transcribe the provided audio accurately with precise speaker diarization.
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
      "text": "First speaker line here.",
      "speaker": "SPEAKER_00"
    }},
    {{
      "start": 2.8,
      "end": 5.2,
      "text": "Second speaker answering here.",
      "speaker": "SPEAKER_01"
    }}
  ]
}}"""

        response = None
        best_segments = None
        best_lang = None
        fallback_segments = None
        fallback_lang = None
        last_error = None

        def extract_text(resp):
            if not resp:
                return ""
            if isinstance(resp, str):
                return resp

            # 1. Extraction prioritaire depuis les parts (gère text ET audio_transcription spécifique à gemini-3.5-transcribe)
            if hasattr(resp, "candidates") and resp.candidates:
                cand = resp.candidates[0]
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    extracted = []
                    for p in cand.content.parts:
                        # Texte direct
                        txt_val = getattr(p, "text", None)
                        if txt_val and isinstance(txt_val, str) and txt_val.strip():
                            extracted.append(txt_val.strip())
                        # Transcription audio dédiée (spécifique à gemini-3.5-transcribe)
                        at_val = getattr(p, "audio_transcription", None)
                        if at_val:
                            sub_text = getattr(at_val, "text", "") or getattr(at_val, "transcription", "") or str(at_val)
                            if sub_text and sub_text.strip():
                                extracted.append(sub_text.strip())
                    if extracted:
                        return "\n".join(extracted)

            if hasattr(resp, "output_text") and resp.output_text:
                return resp.output_text

            try:
                if hasattr(resp, "text") and resp.text:
                    return resp.text
            except Exception:
                pass

            return ""

        for mod in unique_models:
            try:
                logger.info(f"Gemini Transcribe : Tentative avec le modèle '{mod}'...")
                response_text = ""

                # 1. Requête avec contrainte JSON stricte pour garantir timestamps et speaker tags
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
                    response_text = extract_text(response)
                except Exception as err_json:
                    logger.debug(f"Mode JSON refusé pour '{mod}' ({err_json}), tentative sans config JSON...")
                    try:
                        response = client.models.generate_content(
                            model=mod,
                            contents=[uploaded_file, prompt]
                        )
                        response_text = extract_text(response)
                    except Exception as err_no_json:
                        if "transcribe" in mod.lower():
                            logger.info(f"Mode direct audio pour '{mod}'...")
                            response = client.models.generate_content(
                                model=mod,
                                contents=[uploaded_file]
                            )
                            response_text = extract_text(response)
                        else:
                            raise err_no_json

                text_clean = response_text.strip()
                if text_clean:
                    parsed_segs, det_lang = parse_gemini_transcription(
                        text_clean,
                        default_lang=source_lang or "en",
                        audio_duration=audio_duration,
                        max_speakers=max_spk,
                    )
                    if parsed_segs:
                        is_structured = (
                            '"segments"' in text_clean or 
                            '"speaker"' in text_clean or 
                            re.search(r'\d+:\d+.*SPEAKER', text_clean, re.IGNORECASE) is not None
                        )
                        detected_spks = set(s.get("speaker") for s in parsed_segs if s.get("speaker"))
                        
                        # Si le nombre de locuteurs requis est déjà présent dans la réponse
                        if len(detected_spks) >= (max_spk or 1) and is_structured:
                            logger.info(f"Gemini Transcribe : Succès avec le modèle '{mod}' ! ({len(parsed_segs)} segments, locuteurs: {detected_spks})")
                            best_segments = parsed_segs
                            best_lang = det_lang
                            break
                        elif max_spk and max_spk > 1:
                            # Forçage immédiat des locuteurs configurés dans l'interface
                            logger.info(f"Gemini Transcribe ('{mod}') : {len(parsed_segs)} segments transcrits. Forçage des {max_spk} locuteurs demandés...")
                            forced_segs = force_speakers_on_segments(
                                parsed_segs,
                                max_speakers=max_spk,
                                client=client,
                                uploaded_file=uploaded_file,
                                audio_np=audio,
                            )
                            best_segments = forced_segs
                            best_lang = det_lang
                            break
                        elif is_structured:
                            best_segments = parsed_segs
                            best_lang = det_lang
                            break
                        else:
                            logger.warning(f"Modèle '{mod}' a renvoyé du texte brut sans diarisation. Recherche d'un modèle supérieur...")
                            if fallback_segments is None:
                                fallback_segments = parsed_segs
                                fallback_lang = det_lang
                    else:
                        logger.warning(f"Modèle '{mod}' : réponse reçue mais aucun segment valide extrait.")
                else:
                    logger.warning(f"Modèle '{mod}' a renvoyé une réponse vide.")

            except Exception as e:
                logger.warning(f"Modèle '{mod}' indisponible ({e}). Bascule sur le modèle suivant...")
                last_error = e

        if best_segments is not None:
            formatted_segments = best_segments
            detected_lang = best_lang
        elif fallback_segments is not None:
            logger.info("Utilisation des segments de secours issus du texte brut...")
            formatted_segments = fallback_segments
            detected_lang = fallback_lang
        else:
            raise RuntimeError(f"Échec de la transcription Gemini sur tous les modèles: {last_error}")

        if not formatted_segments:
            raise ValueError("Impossible d'extraire des segments de la réponse Gemini.")

        # Vérification finale de sécurité : garantir que le nombre de locuteurs requis est présent
        if max_spk and max_spk > 1:
            final_spks = set(s.get("speaker") for s in formatted_segments if s.get("speaker"))
            if len(final_spks) < max_spk:
                logger.info(f"Forçage final des {max_spk} locuteurs de l'interface ({len(final_spks)} détecté(s) -> {max_spk} requis)...")
                formatted_segments = force_speakers_on_segments(
                    formatted_segments,
                    max_speakers=max_spk,
                    client=client,
                    uploaded_file=uploaded_file,
                    audio_np=audio,
                )

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

    device = os.environ.get("SONITR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Nettoyage préventif agressif de la mémoire pour éviter les crashs OOM ctranslate2
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = None
    fallback_computes = [compute_type]
    if "cuda" in str(device).lower():
        if compute_type != "int8_float16":
            fallback_computes.append("int8_float16")
        if compute_type != "int8":
            fallback_computes.append("int8")

    load_error = None
    for c_type in fallback_computes:
        try:
            logger.info(f"Chargement de Whisper ({asr_model}) sur {device} [compute_type={c_type}]...")
            model = whisperx.load_model(
                asr_model,
                device,
                compute_type=c_type,
                language=SOURCE_LANGUAGE,
                asr_options=asr_options,
            )
            break
        except Exception as e_load:
            load_error = e_load
            if "out of memory" in str(e_load).lower() or "cuda failed" in str(e_load).lower():
                logger.warning(f"VRAM insuffisante avec compute_type='{c_type}'. Libération mémoire et essai de quantification plus légère...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e_load

    if model is None:
        logger.warning(f"Impossible d'allouer Whisper sur GPU ({load_error}). Bascule de secours sur CPU [compute_type=int8]...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = whisperx.load_model(
            asr_model,
            "cpu",
            compute_type="int8",
            language=SOURCE_LANGUAGE,
            asr_options=asr_options,
        )

    audio = whisperx.load_audio(audio_wav)

    # Transcription résiliente contre les erreurs de mémoire par réduction dynamique du batch_size
    result = None
    cur_batch = batch_size
    while cur_batch >= 1:
        try:
            result = model.transcribe(
                audio,
                batch_size=cur_batch,
                chunk_size=segment_duration_limit,
                print_progress=True,
            )
            break
        except Exception as e_trans:
            if ("out of memory" in str(e_trans).lower() or "cuda failed" in str(e_trans).lower()) and cur_batch > 1:
                cur_batch = max(1, cur_batch // 2)
                logger.warning(f"Mémoire saturée pendant la transcription. Réduction de batch_size à {cur_batch} et réessai...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e_trans

    if result["language"] == "zh" and not prompt:
        result["language"] = "zh-TW"
        logger.info("Chinese - Traditional (zh-TW)")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return audio, result


def align_speech(audio, result):
    """
    Aligns speech segments based on the provided audio and result metadata.
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

    align_device = os.environ.get("SONITR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model_a = None
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=align_device,
            model_name=None
            if result["language"] in DAMHF.keys()
            else EXTRA_ALIGN[result["language"]],
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            align_device,
            return_char_alignments=True,
            print_progress=False,
        )
    except Exception as e_align:
        if ("out of memory" in str(e_align).lower() or "cuda failed" in str(e_align).lower()) and "cuda" in str(align_device).lower():
            logger.warning(f"VRAM insuffisante pour l'alignement Wav2Vec2 sur GPU ({e_align}). Bascule automatique de secours sur CPU...")
            if model_a is not None:
                del model_a
                model_a = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device="cpu",
                model_name=None
                if result["language"] in DAMHF.keys()
                else EXTRA_ALIGN[result["language"]],
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                "cpu",
                return_char_alignments=True,
                print_progress=False,
            )
        else:
            raise e_align
    finally:
        if model_a is not None:
            del model_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


diarization_models = {
    "pyannote_3.1": "pyannote/speaker-diarization-3.1",
    "pyannote_2.1": "pyannote/speaker-diarization@2.1",
    "disable": "",
}


def extract_acoustic_signature(audio_np, sample_rate=16000):
    """
    Extrait une signature acoustique normalisée (timbre, formants MFCC, pitch F0, centroïde spectral)
    d'un extrait de parole.
    """
    import numpy as np
    if audio_np is None or len(audio_np) < 800:
        return None, None
    peak = float(np.max(np.abs(audio_np)))
    if peak < 0.005:
        return None, None
    norm_audio = audio_np / peak

    feats = []
    # 1. MFCC & Timbre spectral
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=norm_audio, sr=sample_rate, n_mfcc=16)
        mfcc_mean = np.mean(mfcc, axis=1)
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=norm_audio, sr=sample_rate))
        spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=norm_audio, sr=sample_rate))
        feats.extend(mfcc_mean.tolist())
        feats.append(float(spec_cent) / 2000.0)
        feats.append(float(spec_rolloff) / 4000.0)
    except Exception:
        fft_vals = np.abs(np.fft.rfft(norm_audio))
        freqs = np.fft.rfftfreq(len(norm_audio), 1.0 / sample_rate)
        bands = np.array_split(fft_vals, 16)
        feats.extend([float(np.mean(b)) for b in bands])
        centroid = float(np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-8))
        feats.append(centroid / 2000.0)
        feats.append(1.0)

    # 2. Pitch F0 acoustique
    f0 = None
    try:
        from soni_translate.gender_detection import extract_f0_pitch
        f0, _ = extract_f0_pitch(norm_audio, sample_rate)
    except Exception:
        pass

    feats.append((f0 / 250.0) if f0 and f0 > 0 else 0.5)

    vec = np.array(feats, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec, f0


def get_segment_pitch(audio_np, start_sec, end_sec, sample_rate=16000):
    """
    Extrait la fréquence fondamentale (F0 en Hz) d'un segment audio précis.
    """
    if audio_np is None or len(audio_np) == 0:
        return None, 0.0
    try:
        t0 = max(0, int(float(start_sec) * sample_rate))
        t1 = min(len(audio_np), int(float(end_sec) * sample_rate))
        # Au moins 0.12s (1920 échantillons à 16kHz)
        if t1 - t0 < int(sample_rate * 0.12):
            return None, 0.0
        chunk = audio_np[t0:t1]
        from soni_translate.gender_detection import extract_f0_pitch
        return extract_f0_pitch(chunk, sample_rate)
    except Exception:
        return None, 0.0


def smooth_speaker_fragmentation(segments, min_duration=0.9, max_gap=1.2, audio_np=None, sample_rate=16000, speaker_pitches=None):
    """
    Lissage temporel intelligent et acoustique des sauts de locuteurs (anti-fragmentation / anti-ping-pong).
    Si un court segment (< min_duration) est entouré par le même locuteur avant et après,
    on vérifie son pitch acoustique (F0) avant toute réassignation :
    - Si le pitch du segment correspond à sa voix actuelle (ex: réplique courte d'une femme au milieu d'un homme),
      il est STRICTEMENT PROTÉGÉ et NON RÉASSIGNÉ.
    - S'il n'y a pas de contradiction acoustique ou si c'est un artefact court, il est réassigné.
    """
    if len(segments) < 3:
        return segments

    speaker_pitches = speaker_pitches or {}

    for _ in range(2):
        changed = False
        for i in range(1, len(segments) - 1):
            s_prev = segments[i - 1].get("speaker")
            s_curr = segments[i].get("speaker")
            s_next = segments[i + 1].get("speaker")

            if s_prev and s_next and s_prev == s_next and s_curr != s_prev:
                dur = float(segments[i].get("end", 0)) - float(segments[i].get("start", 0))
                gap1 = float(segments[i].get("start", 0)) - float(segments[i - 1].get("end", 0))
                gap2 = float(segments[i + 1].get("start", 0)) - float(segments[i].get("end", 0))

                if dur < min_duration and gap1 < max_gap and gap2 < max_gap:
                    # Protection acoustique : vérification du pitch F0
                    p_curr = speaker_pitches.get(s_curr)
                    p_prev = speaker_pitches.get(s_prev)
                    
                    protect_segment = False
                    if audio_np is not None and p_curr and p_prev and abs(p_curr - p_prev) >= 25.0:
                        seg_f0, voiced_ratio = get_segment_pitch(
                            audio_np, segments[i].get("start", 0), segments[i].get("end", 0), sample_rate
                        )
                        if seg_f0 and voiced_ratio >= 0.15:
                            # Si le pitch est plus proche du locuteur actuel que du locuteur entourant
                            if abs(seg_f0 - p_curr) < abs(seg_f0 - p_prev):
                                protect_segment = True
                                logger.info(
                                    f"Post-Diarisation : Segment {i} ({dur:.2f}s, Pitch {seg_f0:.1f}Hz) "
                                    f"protégé pour '{s_curr}' contre absorption par '{s_prev}' (voix distincte confirmée acoustiquement)."
                                )

                    if not protect_segment:
                        logger.info(
                            f"Post-Diarisation Anti-Fragmentation : Segment {i} ({dur:.2f}s) "
                            f"réassigné de '{s_curr}' à '{s_prev}' (lissage temporel)."
                        )
                        segments[i]["speaker"] = s_prev
                        changed = True
        if not changed:
            break

    return segments


def refine_diarization_and_speakers(audio_wav, result, max_speakers=None):
    """
    Post-traitement avancé et robuste de la diarisation :
    1. Chargement de l'audio 16kHz mono.
    2. Extraction des profils acoustiques initiaux (MFCC + Pitch F0) par locuteur.
    3. Fusion des voix acoustiquement identiques (anti-overclustering).
    4. Élimination des faux locuteurs parasites (toux, bruits < 0.8s).
    5. Lissage temporel intelligent avec protection acoustique de pitch F0.
    6. Sauvetage Acoustique F0 des segments (rescue homme/femme per-segment).
    7. Classement des locuteurs par temps de parole réel (SPEAKER_00 = personnage principal).
    8. Attribution des locuteurs excédentaires au profil le plus ressemblant.
    """
    import numpy as np
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

    segments = result.get("segments", [])
    if not segments:
        return result

    if max_spk_val == 1:
        for s in segments:
            s["speaker"] = "SPEAKER_00"
        return result

    # 1. Chargement de l'audio en 16kHz mono pour comparaison acoustique et F0
    audio_np = None
    sample_rate = 16000
    if audio_wav and os.path.exists(audio_wav):
        try:
            import soundfile as sf
            import numpy as np
            audio_data, sr = sf.read(audio_wav)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sr != 16000:
                try:
                    import torchaudio
                    t_wav = torch.from_numpy(audio_data).unsqueeze(0).to(torch.float32)
                    resample = torchaudio.transforms.Resample(sr, 16000)
                    audio_data = resample(t_wav).squeeze(0).numpy()
                except Exception:
                    pass
            audio_np = audio_data.astype(np.float32)
        except Exception as e_aud:
            logger.warning(f"Audio non lisible pour comparaison acoustique ({e_aud}), poursuite sans empreintes.")

    # 2. Extraction des profils acoustiques par locuteur (MFCC, Pitch F0)
    speaker_segments = {}
    speaker_durations = {}
    for s in segments:
        spk = s.get("speaker", "SPEAKER_00")
        dur = max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append(s)

    speaker_profiles = {}
    if audio_np is not None and len(audio_np) > 1600:
        total_audio_len = len(audio_np)
        for spk, seg_list in speaker_segments.items():
            valid_vecs = []
            f0_list = []
            # On privilégie les segments de durée >= 0.4s
            for s in seg_list:
                t0 = min(int(float(s.get("start", 0)) * sample_rate), total_audio_len)
                t1 = min(int(float(s.get("end", 0)) * sample_rate), total_audio_len)
                if t1 - t0 >= int(sample_rate * 0.4):
                    chunk = audio_np[t0:t1]
                    vec, f0 = extract_acoustic_signature(chunk, sample_rate)
                    if vec is not None:
                        valid_vecs.append(vec)
                    if f0 and f0 > 0:
                        f0_list.append(f0)
                if len(valid_vecs) >= 8:
                    break

            if valid_vecs:
                mean_vec = np.mean(valid_vecs, axis=0)
                mean_norm = np.linalg.norm(mean_vec)
                if mean_norm > 1e-6:
                    mean_vec = mean_vec / mean_norm
                speaker_profiles[spk] = {
                    "vector": mean_vec,
                    "f0": float(np.median(f0_list)) if f0_list else None,
                    "duration": speaker_durations.get(spk, 0.0),
                }

        if speaker_profiles:
            logger.info(f"Analyse Acoustique : {len(speaker_profiles)} profil(s) vocal(aux) extrait(s) (MFCC/Pitch/Timbre).")

    # 3. Comparaison indépendante des voix & Fusion automatique des clusters similaires (Anti-Overclustering)
    merged_into = {}
    spk_keys = list(speaker_profiles.keys())
    for i in range(len(spk_keys)):
        s_i = spk_keys[i]
        root_i = merged_into.get(s_i, s_i)
        for j in range(i + 1, len(spk_keys)):
            s_j = spk_keys[j]
            root_j = merged_into.get(s_j, s_j)
            if root_i == root_j:
                continue

            v_i = speaker_profiles[root_i]["vector"]
            v_j = speaker_profiles[root_j]["vector"]
            cos_sim = float(np.dot(v_i, v_j))
            f0_i = speaker_profiles[root_i].get("f0")
            f0_j = speaker_profiles[root_j].get("f0")
            pitch_diff = abs(f0_i - f0_j) if (f0_i and f0_j) else 999.0

            if max_spk_val > 0 and len(spk_keys) <= max_spk_val:
                is_same_speaker = cos_sim >= 0.94 and pitch_diff <= 6.0
            else:
                is_same_speaker = cos_sim >= 0.90 or (cos_sim >= 0.86 and pitch_diff <= 8.0)
            if is_same_speaker:
                if speaker_profiles[root_i]["duration"] >= speaker_profiles[root_j]["duration"]:
                    target, absorbed = root_i, root_j
                else:
                    target, absorbed = root_j, root_i

                merged_into[absorbed] = target
                logger.info(
                    f"Comparaison Acoustique Indépendante : Voix '{absorbed}' fusionnée avec '{target}' "
                    f"(Similarité = {cos_sim:.2f}, Écart Pitch = {pitch_diff:.1f}Hz)."
                )
            else:
                pitch_info = f"{pitch_diff:.1f}Hz" if pitch_diff < 900 else "N/A"
                logger.info(
                    f"Comparaison Acoustique Indépendante : '{root_i}' vs '{root_j}' -> "
                    f"Similarité = {cos_sim:.2f}, Écart Pitch = {pitch_info} -> Deux voix distinctes confirmées."
                )

    if merged_into:
        for s in segments:
            cur_spk = s.get("speaker", "SPEAKER_00")
            while cur_spk in merged_into:
                cur_spk = merged_into[cur_spk]
            s["speaker"] = cur_spk

    # 4. Élimination des micro-locuteurs parasites (bruits ou toux isolées < 0.8s)
    speaker_durations.clear()
    total_speech_time = 0.0
    for s in segments:
        dur = max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
        spk = s.get("speaker", "SPEAKER_00")
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
        total_speech_time += dur

    if len(speaker_durations) > 1 and total_speech_time > 5.0:
        ghost_speakers = set()
        for spk, dur in speaker_durations.items():
            if dur < 0.8 or (dur / total_speech_time) < 0.015:
                ghost_speakers.add(spk)

        if ghost_speakers and len(ghost_speakers) < len(speaker_durations):
            logger.info(f"Élimination des locuteurs parasites/bruits : {ghost_speakers}")
            for i, s in enumerate(segments):
                if s.get("speaker") in ghost_speakers:
                    fallback_spk = None
                    if i > 0 and segments[i - 1].get("speaker") not in ghost_speakers:
                        fallback_spk = segments[i - 1].get("speaker")
                    elif i < len(segments) - 1 and segments[i + 1].get("speaker") not in ghost_speakers:
                        fallback_spk = segments[i + 1].get("speaker")
                    else:
                        fallback_spk = max((k for k in speaker_durations if k not in ghost_speakers), key=lambda k: speaker_durations[k])
                    s["speaker"] = fallback_spk

    # Dictionnaire des pitchs de référence par locuteur
    speaker_pitches = {spk: prof.get("f0") for spk, prof in speaker_profiles.items() if prof.get("f0")}

    # 5. Lissage anti-fragmentation avec protection acoustique de pitch F0
    segments = smooth_speaker_fragmentation(
        segments, min_duration=0.9, max_gap=1.2, audio_np=audio_np, sample_rate=sample_rate, speaker_pitches=speaker_pitches
    )

    # 6. Sauvetage Acoustique F0 des segments (Per-segment pitch rescue homme/femme)
    # Si deux locuteurs ont un pitch clairement contrasté (écart >= 30Hz, ex: homme 102Hz vs femme 185Hz)
    active_spks_with_pitch = [(spk, speaker_pitches[spk]) for spk in speaker_pitches if spk in set(s.get("speaker") for s in segments)]
    if len(active_spks_with_pitch) >= 2 and audio_np is not None:
        active_spks_with_pitch.sort(key=lambda x: x[1])
        low_spk, p_low = active_spks_with_pitch[0]
        high_spk, p_high = active_spks_with_pitch[-1]

        if (p_high - p_low) >= 30.0:
            split_pitch = (p_low + p_high) / 2.0
            margin = min(12.0, (p_high - p_low) * 0.15)
            high_thresh = split_pitch + margin
            low_thresh = split_pitch - margin
            logger.info(
                f"Sauvetage Acoustique F0 activé entre '{low_spk}' ({p_low:.1f}Hz) et '{high_spk}' ({p_high:.1f}Hz) "
                f"[Seuils : Homme <= {low_thresh:.1f}Hz | Femme >= {high_thresh:.1f}Hz]"
            )
            rescue_count = 0
            for i, seg in enumerate(segments):
                s_start = float(seg.get("start", 0))
                s_end = float(seg.get("end", 0))
                dur = s_end - s_start
                if dur < 0.20:
                    continue
                seg_f0, voiced_ratio = get_segment_pitch(audio_np, s_start, s_end, sample_rate)
                if not seg_f0 or voiced_ratio < 0.15:
                    continue

                cur_spk = seg.get("speaker")
                # Si attribué au locuteur grave (homme) alors que la voix est aiguë (femme)
                if cur_spk == low_spk and seg_f0 >= high_thresh:
                    seg["speaker"] = high_spk
                    rescue_count += 1
                    logger.info(
                        f"Sauvetage Acoustique F0 : Segment {i} ({dur:.2f}s, {seg_f0:.1f}Hz) "
                        f"réassigné de '{low_spk}' -> '{high_spk}' (Voix féminine confirmée par pitch F0)."
                    )
                # Si attribué au locuteur aigu (femme) alors que la voix est grave (homme)
                elif cur_spk == high_spk and seg_f0 <= low_thresh:
                    seg["speaker"] = low_spk
                    rescue_count += 1
                    logger.info(
                        f"Sauvetage Acoustique F0 : Segment {i} ({dur:.2f}s, {seg_f0:.1f}Hz) "
                        f"réassigné de '{high_spk}' -> '{low_spk}' (Voix masculine confirmée par pitch F0)."
                    )
            if rescue_count > 0:
                logger.info(f"Sauvetage Acoustique F0 terminé : {rescue_count} segment(s) restauré(s) avec succès.")

    # 7. Classement par temps de parole réel (SPEAKER_00 = voix principale)
    final_durations = {}
    for s in segments:
        dur = max(0.0, float(s.get("end", 0)) - float(s.get("start", 0)))
        spk = s.get("speaker", "SPEAKER_00")
        final_durations[spk] = final_durations.get(spk, 0.0) + dur

    sorted_speakers = sorted(final_durations.keys(), key=lambda k: final_durations[k], reverse=True)
    speaker_mapping = {}

    allowed_count = max_spk_val if max_spk_val > 0 else len(sorted_speakers)
    primary_speakers = sorted_speakers[:allowed_count]
    excess_speakers = sorted_speakers[allowed_count:]

    for idx, spk in enumerate(primary_speakers):
        speaker_mapping[spk] = f"SPEAKER_{idx:02d}"

    # Attribution intelligente des locuteurs excédentaires au profil acoustique le plus proche
    for spk in excess_speakers:
        best_target = primary_speakers[-1]
        best_sim = -1.0
        if spk in speaker_profiles:
            v_spk = speaker_profiles[spk]["vector"]
            for cand in primary_speakers:
                if cand in speaker_profiles:
                    sim = float(np.dot(v_spk, speaker_profiles[cand]["vector"]))
                    if sim > best_sim:
                        best_sim = sim
                        best_target = cand
        speaker_mapping[spk] = speaker_mapping[best_target]
        logger.info(f"Locuteur excédentaire '{spk}' ({final_durations[spk]:.1f}s) assigné acoustiquement à '{speaker_mapping[spk]}'")

    for s in segments:
        old_spk = s.get("speaker", "SPEAKER_00")
        s["speaker"] = speaker_mapping.get(old_spk, "SPEAKER_00")

    result["segments"] = segments
    detected_final = set(s["speaker"] for s in segments)
    logger.info(
        f"Post-Diarisation terminée : {len(detected_final)} locuteur(s) validé(s) {detected_final} "
        f"(Durées de parole : { {speaker_mapping.get(k, k): f'{v:.1f}s' for k, v in final_durations.items()} })"
    )
    return result


def reencode_speakers(result, max_speakers=None):
    """
    Fonction de compatibilité : applique le raffinement et lissage intelligent des locuteurs.
    """
    return refine_diarization_and_speakers(None, result, max_speakers=max_speakers)


def diarize_speech(
    audio_wav,
    result,
    min_speakers,
    max_speakers,
    YOUR_HF_TOKEN,
    model_name="pyannote/speaker-diarization-3.1",
):
    import os
    import gc
    import torch
    import whisperx
    from soni_translate.speech_segmentation import diarization_models

    if YOUR_HF_TOKEN:
        YOUR_HF_TOKEN = str(YOUR_HF_TOKEN).strip()

    # Si l'utilisateur n'a pas sélectionné de modèle ou a laissé 2.1 par défaut, on privilégie 3.1
    if not model_name or "2.1" in str(model_name):
        model_name = "pyannote/speaker-diarization-3.1"

    if max(min_speakers, max_speakers) > 1:
        diarize_model = None

        # Nettoyage préventif VRAM pour Pyannote
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- TENTATIVE 1 : Modèle Officiel Pyannote 3.1 (Avec Token) ---
        if YOUR_HF_TOKEN:
            try:
                logger.info(f"Tentative 1 : Connexion officielle Pyannote avec Token ({model_name})...")
                diarize_model = whisperx.DiarizationPipeline(
                    model_name=model_name,
                    use_auth_token=YOUR_HF_TOKEN,
                    device=os.environ.get("SONITR_DEVICE"),
                )
            except Exception as e:
                logger.warning(f"Échec Tentative 1 Pyannote officiel ({e}). Bascule vers le modèle public 3.1...")

        # --- TENTATIVE 2 : Modèle Public 3.1 (Sans Token) ---
        if diarize_model is None:
            try:
                public_model = "fatymatariq/speaker-diarization-3.1"
                logger.info(f"Tentative 2 : Chargement du modèle public 3.1 ({public_model})...")
                diarize_model = whisperx.DiarizationPipeline(
                    model_name=public_model,
                    use_auth_token=None,
                    device=os.environ.get("SONITR_DEVICE"),
                )
                logger.info("Succès : Modèle public Pyannote 3.1 chargé.")
            except Exception as e2:
                # Dernier recours si le modèle public échoue
                if YOUR_HF_TOKEN and model_name != "pyannote/speaker-diarization@2.1":
                    try:
                        logger.warning(f"Modèle public échoué ({e2}). Repli sur pyannote/speaker-diarization@2.1...")
                        diarize_model = whisperx.DiarizationPipeline(
                            model_name="pyannote/speaker-diarization@2.1",
                            use_auth_token=YOUR_HF_TOKEN,
                            device=os.environ.get("SONITR_DEVICE"),
                        )
                    except Exception as e3:
                        raise ValueError(f"Échec de l'initialisation de Pyannote: {e3}")
                else:
                    raise ValueError(f"Échec de l'initialisation de Pyannote: {e2}")

        diarize_segments = diarize_model(
            audio_wav, min_speakers=min_speakers, max_speakers=max_speakers
        )

        raw_speakers = []
        try:
            if hasattr(diarize_segments, "speaker"):
                raw_speakers = list(set(diarize_segments["speaker"]))
            elif hasattr(diarize_segments, "itertracks"):
                raw_speakers = list(set(track[2] for track in diarize_segments.itertracks(yield_label=True)))
            logger.info(f"Pyannote 3.1 brut : {len(raw_speakers)} locuteur(s) détecté(s) -> {raw_speakers}")
        except Exception as e_raw:
            logger.debug(f"Diag raw speakers ({e_raw})")

        result_diarize = whisperx.assign_word_speakers(
            diarize_segments, result
        )

        # Rattrapage direct par intersection temporelle si assign_word_speakers a perdu des locuteurs
        assigned_spks = set(s.get("speaker") for s in result_diarize.get("segments", []) if s.get("speaker"))
        if len(assigned_spks) <= 1 and len(raw_speakers) > 1 and hasattr(diarize_segments, "iterrows"):
            logger.warning(
                f"assign_word_speakers n'a assigné qu'un seul locuteur ({assigned_spks}) alors que Pyannote en a trouvé {len(raw_speakers)}. "
                "Recalage direct par intersection temporelle des segments..."
            )
            for seg in result_diarize.get("segments", []):
                s_start = float(seg.get("start", 0))
                s_end = float(seg.get("end", 0))
                best_spk = seg.get("speaker", "SPEAKER_00")
                best_overlap = 0.0

                for _, row in diarize_segments.iterrows():
                    d_start = float(row.get("start", 0))
                    d_end = float(row.get("end", 0))
                    d_spk = row.get("speaker", "SPEAKER_00")
                    overlap = max(0.0, min(s_end, d_end) - max(s_start, d_start))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_spk = d_spk
                seg["speaker"] = best_spk
            assigned_spks = set(s.get("speaker") for s in result_diarize.get("segments", []) if s.get("speaker"))
            logger.info(f"Après recalage temporel : {len(assigned_spks)} locuteur(s) assignés -> {assigned_spks}")

        for segment in result_diarize["segments"]:
            if "speaker" not in segment:
                segment["speaker"] = "SPEAKER_00"

        del diarize_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        result_diarize = result
        for item in result_diarize["segments"]:
            item["speaker"] = "SPEAKER_00"

    # Post-traitement avancé : Comparaison acoustique indépendante, Anti-fragmentation et Tri intelligent
    return refine_diarization_and_speakers(audio_wav, result_diarize, max_speakers=max_speakers)

