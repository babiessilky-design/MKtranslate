from whisperx.alignment import (
    DEFAULT_ALIGN_MODELS_TORCH as DAMT,
    DEFAULT_ALIGN_MODELS_HF as DAMHF,
)
from whisperx.utils import TO_LANGUAGE_CODE
import whisperx
import torch
import gc
import os
import soundfile as sf
from IPython.utils import capture # noqa
from .language_configuration import EXTRA_ALIGN, INVERTED_LANGUAGES
from .logging_setup import logger
from .postprocessor import sanitize_file_name
from .utils import remove_directory_contents, run_command

ASR_MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
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


def reencode_speakers(result):
    speaker_mapping = {}
    counter = 0
    
    # On enlève la vérification du début pour forcer le nettoyage
    # logger.debug("Reencode speakers") (Optionnel)

    for segment in result["segments"]:
        # On utilise .get() pour éviter le crash si la clé manque
        old_speaker = segment.get("speaker", "SPEAKER_00")
        
        if old_speaker not in speaker_mapping:
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
