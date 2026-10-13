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
    Garantit l'intégrité intégrale de l'audio (zéro coupure) et la cohérence temporelle.
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
        total_ms = int((total_duration + 5.0) * 1000)
        base_audio = AudioSegment.silent(duration=total_ms, frame_rate=sample_rate).set_channels(1)

        last_end_time = 0.0

        for line, audio_file in tqdm(zip(result_diarize["segments"], audio_files), total=len(audio_files), desc="Mixing Audio"):
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                logger.warning(f"Segment introuvable ou vide ignoré : {audio_file}")
                continue

            orig_start = float(line["start"])
            start = orig_start

            try:
                audio = AudioSegment.from_file(audio_file)
                if audio.frame_rate != sample_rate:
                    audio = audio.set_frame_rate(sample_rate)
                if audio.channels != 1:
                    audio = audio.set_channels(1)

                duration_sec = len(audio) / 1000.0

                if avoid_overlap:
                    # Si le début théorique chevauche la parole précédente
                    if start < last_end_time:
                        # On décale pour que la parole commence dès la fin de la précédente
                        # mais on plafonne le décalage à 1.2s max par rapport au timing visuel d'origine
                        shifted_start = last_end_time + 0.05
                        max_allowed_start = orig_start + 1.2
                        start = min(shifted_start, max_allowed_start)
                    else:
                        # Si aucun chevauchement, resynchronisation directe sur le timing original (0 dérive)
                        start = orig_start

                    last_end_time = start + duration_sec
                else:
                    last_end_time = start + duration_sec

                start_ms = max(0, int(start * 1000))
                needed_ms = start_ms + len(audio) + 1000

                # Agrandir dynamiquement base_audio si le segment dépasse pour éviter TOUT tronquage Pydub
                if len(base_audio) < needed_ms:
                    extra_ms = needed_ms - len(base_audio)
                    base_audio += AudioSegment.silent(duration=extra_ms, frame_rate=sample_rate).set_channels(1)

                base_audio = base_audio.overlay(audio, position=start_ms)
            except Exception as e:
                logger.error(f"Erreur lors du mixage de {audio_file}: {e}")

        # Export au format attendu selon l'extension sans troncature prématurée
        ext = os.path.splitext(final_file)[1].replace(".", "").lower()
        if ext == "ogg":
            try:
                base_audio.export(final_file, format="ogg", codec="libvorbis")
            except Exception:
                base_audio.export(final_file, format="ogg")
        elif ext == "mp3":
            base_audio.export(final_file, format="mp3")
        else:
            base_audio.export(final_file, format="wav")

    logger.info(f"✅ Audio final créé : {final_file} (durée totale : {len(base_audio)/1000.0:.2f}s)")
