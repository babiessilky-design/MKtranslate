import os
import torch
import soundfile as sf
from transformers import pipeline
from soni_translate.logging_setup import logger

class VoiceGenderDetector:
    def __init__(self, model_id="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"):
        self.model_id = model_id
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.classifier is None:
            logger.info(f"Loading multilingual voice gender classification model: {self.model_id}")
            self.classifier = pipeline(
                "audio-classification",
                model=self.model_id,
                device=0 if self.device == "cuda" else -1
            )

    def detect_speaker_genders(self, audio_path, diarization_result):
        self.load_model()
        
        speaker_segments = {}
        for segment in diarization_result.get("segments", []):
            speaker = segment.get("speaker")
            if speaker:
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((segment["start"], segment["end"]))
        
        try:
            # SÉCURITÉ ABSOLUE : On utilise soundfile à la place de torchaudio pour éviter l'incompatibilité CUDA 13 / torchcodec
            audio_data, sample_rate = sf.read(audio_path)
            
            # Conversion propre en tenseur PyTorch (channels, frames)
            if audio_data.ndim == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_data).transpose(0, 1)
                
            waveform = waveform.to(torch.float32)
        except Exception as e:
            logger.error(f"Error loading audio for gender detection: {e}")
            return {}

        speaker_genders = {}
        total_frames = waveform.shape[1]

        for speaker, segments in speaker_segments.items():
            speaker_chunks = []
            accumulated_duration = 0.0

            for start, end in segments:
                duration = end - start
                if duration < 0.5:
                    continue
                
                start_frame = min(int(start * sample_rate), total_frames)
                end_frame = min(int(end * sample_rate), total_frames)
                
                if start_frame >= end_frame:
                    continue
                
                chunk = waveform[:, start_frame:end_frame]
                speaker_chunks.append(chunk)
                accumulated_duration += duration
                
                if accumulated_duration >= 8.0:
                    break

            if not speaker_chunks:
                logger.warning(f"No valid audio segments found for speaker {speaker}")
                speaker_genders[speaker] = "unknown"
                continue

            combined_waveform = torch.cat(speaker_chunks, dim=1)
            
            # Ré-échantillonnage à 16000 Hz pour la compatibilité avec Wav2Vec2
            if sample_rate != 16000:
                import torchaudio
                transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                combined_waveform = transform(combined_waveform)

            # Conversion en Mono
            mono_waveform = combined_waveform.mean(dim=0, keepdim=True)
            
            # Normalisation du volume
            peak = torch.max(torch.abs(mono_waveform))
            if peak > 0.0:
                mono_waveform = (mono_waveform / peak) * 0.9

            temp_wav_path = f"temp_gender_{speaker}.wav"
            
            # Écriture propre avec soundfile pour éviter torchaudio.save (qui charge aussi torchcodec)
            audio_np = mono_waveform.squeeze(0).cpu().numpy()
            sf.write(temp_wav_path, audio_np, 16000)

            try:
                predictions = self.classifier(temp_wav_path)
                gender_label = predictions[0]["label"].lower()
                speaker_genders[speaker] = gender_label
                logger.info(f"Gender detected for {speaker}: {gender_label} (Confidence: {predictions[0]['score']:.2f})")
            except Exception as e:
                logger.error(f"Failed to detect gender for {speaker}: {e}")
                speaker_genders[speaker] = "unknown"
            finally:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)

        return speaker_genders


def auto_assign_voices(speaker_genders, target_language="french", default_male="fr-FR-HenriNeural-Male", default_female="fr-FR-DeniseNeural-Female"):
    assigned_voices = {}
    for speaker, gender in speaker_genders.items():
        if gender == "female":
            assigned_voices[speaker] = default_female
        else:
            assigned_voices[speaker] = default_male
            
    return assigned_voices
