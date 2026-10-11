from pydub import AudioSegment
from tqdm import tqdm
from .utils import run_command
from .logging_setup import logger
import numpy as np
import os
from TTS.api import TTS  # Pour Coqui XTTS v2 multi-voix

class Mixer:
    def __init__(self):
        self.parts = []

    def __len__(self):
        parts = self._sync()
        seg = parts[0][1]
        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        return int(1000.0 * frame_count / seg.frame_rate)

    def overlay(self, sound, position=0):
        self.parts.append((position, sound))
        return self

    def _sync(self):
        positions, segs = zip(*self.parts)
        frame_rate = segs[0].frame_rate
        offsets = [int(frame_rate * pos / 1000.0) for pos in positions]
        segs = AudioSegment.empty()._sync(*segs)
        return list(zip(offsets, segs))

    def append(self, sound):
        self.overlay(sound, position=len(self))

    def to_audio_segment(self):
        parts = self._sync()
        seg = parts[0][1]
        channels = seg.channels
        frame_count = max(offset + seg.frame_count() for offset, seg in parts)
        sample_count = int(frame_count * seg.channels)
        output = np.zeros(sample_count, dtype="int32")
        for offset, seg in parts:
            sample_offset = offset * channels
            samples = np.frombuffer(seg.get_array_of_samples(), dtype="int32")
            samples = np.int16(samples / np.max(np.abs(samples)) * 32767)
            start = sample_offset
            end = start + len(samples)
            output[start:end] += samples
        return seg._spawn(output, overrides={"sample_width": 4}).normalize(headroom=0.0)


def create_translated_audio(
    result_diarize, 
    audio_files, 
    final_file, 
    concat=False, 
    avoid_overlap=False,
    # === NOUVEAUX PARAMÈTRES POUR MULTI-VOIX COQUI XTTS ===
    reference_voices=None,      # Liste des chemins des voix de référence (Voice 1, Voice 2...)
    xtts_device="cuda",         # "cuda" ou "cpu"
    xtts_model=None             # Instance TTS déjà chargée (pour éviter de recharger à chaque fois)
):
    """
    result_diarize : segments avec speaker_id
    audio_files    : liste des fichiers audio traduits (déjà générés par TTS)
    reference_voices : liste des chemins audio de référence (ex: ["voice_mec.wav", "voice_fille.wav"])
    """

    total_duration = result_diarize["segments"][-1]["end"]

    # ====================== NOUVEAU : MULTI-VOIX CLONING ======================
    if reference_voices and len(reference_voices) > 0:
        logger.info(f"🔄 Activation Coqui XTTS v2 Multi-Voice Cloning ({len(reference_voices)} voix)")

        if xtts_model is None:
            xtts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
                           progress_bar=False, 
                           gpu=(xtts_device == "cuda"))

        # Mapping speaker_id → voice_reference
        speaker_to_voice = {}
        for i, ref_path in enumerate(reference_voices):
            speaker_id = f"SPEAKER_{i:02d}"
            speaker_to_voice[speaker_id] = ref_path

        # On régénère les audio_files avec cloning si nécessaire
        new_audio_files = []
        for segment in result_diarize["segments"]:
            speaker = segment.get("speaker", "SPEAKER_00")
            ref_voice = speaker_to_voice.get(speaker, reference_voices[0])  # fallback sur la première voix

            # Génère avec cloning
            cloned_audio = xtts_model.tts(
                text=segment["text"],
                speaker_wav=ref_voice,
                language="fr",
                split_sentences=True
            )
            temp_file = f"temp_cloned_{len(new_audio_files)}.wav"
            AudioSegment.from_numpy(cloned_audio).export(temp_file, format="wav")
            new_audio_files.append(temp_file)

        audio_files = new_audio_files  # on remplace par les versions clonées

    # ====================== FIN MULTI-VOIX ======================

    if concat:
        with open("list.txt", "w") as file:
            for i, audio_file in enumerate(audio_files):
                file.write(f"file {audio_file}\n")
        command = f"ffmpeg -f concat -safe 0 -i list.txt -c:a pcm_s16le {final_file}"
        run_command(command)
    else:
        base_audio = AudioSegment.silent(duration=int(total_duration * 1000), frame_rate=41000)
        combined_audio = Mixer()
        combined_audio.overlay(base_audio)

        last_end_time = 0
        previous_speaker = ""

        for line, audio_file in tqdm(zip(result_diarize["segments"], audio_files)):
            start = float(line["start"])

            try:
                audio = AudioSegment.from_file(audio_file)

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

                start_time = start * 1000
                combined_audio = combined_audio.overlay(audio, position=start_time)
            except Exception as e:
                logger.error(f"Error audio file {audio_file}: {e}")

        combined_audio_data = combined_audio.to_audio_segment()
        combined_audio_data.export(final_file, format="wav")

    logger.info(f"✅ Audio final créé : {final_file}")
