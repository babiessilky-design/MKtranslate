from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger
import numpy as np
import os


def create_translated_audio(
    result_diarize, 
    audio_files, 
    final_file, 
    concat=False, 
    avoid_overlap=False,
    reference_voices=None,
    xtts_device="cuda",
    xtts_model=None
):
    """
    Assemble tous les segments audio traduits avec synchronisation milliseconde sur la vidéo.
    """
    if not result_diarize or not result_diarize.get("segments"):
        logger.warning("Aucun segment de diarisation à assembler.")
        return

    total_duration = float(result_diarize["segments"][-1]["end"])

    if concat:
        with open("list.txt", "w", encoding="utf-8") as file:
            for audio_file in audio_files:
                file.write(f"file '{os.path.abspath(audio_file)}'\n")
        command = f'ffmpeg -y -f concat -safe 0 -i list.txt -c:a libvorbis "{final_file}"'
        run_command(command)
    else:
        sample_rate = 44100
        total_ms = int((total_duration + 2.0) * 1000)
        base_audio = AudioSegment.silent(duration=total_ms, frame_rate=sample_rate)
        base_audio = base_audio.set_channels(1)

        last_end_time = 0.0
        previous_speaker = ""

        for line, audio_file in tqdm(zip(result_diarize["segments"], audio_files), total=len(audio_files), desc="Mixing Audio"):
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                logger.warning(f"Segment introuvable ou vide ignoré : {audio_file}")
                continue

            start = float(line["start"])

            try:
                audio = AudioSegment.from_file(audio_file)
                if audio.frame_rate != sample_rate:
                    audio = audio.set_frame_rate(sample_rate)
                if audio.channels != 1:
                    audio = audio.set_channels(1)

                if avoid_overlap:
                    speaker = line.get("speaker", "")
                    if (last_end_time - 0.500) > start:
                        if previous_speaker and previous_speaker != speaker:
                            start = last_end_time - 0.500
                        else:
                            start = last_end_time - 0.200
                    previous_speaker = speaker
                    duration_tts_seconds = len(audio) / 1000.0
                    last_end_time = start + duration_tts_seconds

                start_ms = max(0, int(start * 1000))
                base_audio = base_audio.overlay(audio, position=start_ms)
            except Exception as e:
                logger.error(f"Erreur lors du mixage de {audio_file}: {e}")

        # Tronquer à la durée de fin exacte de la vidéo
        final_duration_ms = int(total_duration * 1000)
        if len(base_audio) > final_duration_ms:
            base_audio = base_audio[:final_duration_ms]

        # Export au format attendu selon l'extension
        ext = os.path.splitext(final_file)[1].replace(".", "").lower()
        if ext == "ogg":
            base_audio.export(final_file, format="ogg")
        elif ext == "mp3":
            base_audio.export(final_file, format="mp3")
        else:
            base_audio.export(final_file, format="wav")

    logger.info(f"✅ Audio final créé : {final_file}")
