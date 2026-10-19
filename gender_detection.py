import os
import torch
import numpy as np
import soundfile as sf
from transformers import pipeline
from soni_translate.logging_setup import logger

def extract_f0_pitch(audio_np, sample_rate=16000):
    """
    Extrait la fréquence fondamentale (F0 en Hz) du signal audio.
    Utilise Parselmouth (Praat) en priorité, puis PyWorld, et enfin un autocorrélateur NumPy ultra-robuste.
    """
    if len(audio_np) == 0:
        return None, 0.0

    # 1. Tentative avec Praat / Parselmouth (La référence acoustique mondiale)
    try:
        import parselmouth
        sound = parselmouth.Sound(audio_np.astype(np.float32), sampling_frequency=sample_rate)
        pitch = sound.to_pitch(time_step=0.01, pitch_floor=70.0, pitch_ceiling=350.0)
        pitch_values = pitch.selected_array['frequency']
        voiced = pitch_values[pitch_values > 0]
        if len(voiced) > 10:
            median_f0 = float(np.median(voiced))
            voiced_ratio = len(voiced) / max(len(pitch_values), 1)
            return median_f0, voiced_ratio
    except Exception:
        pass

    # 2. Tentative avec PyWorld (Vocodeur de haute précision)
    try:
        import pyworld as pw
        audio_f64 = audio_np.astype(np.float64)
        _f0, _t = pw.dio(audio_f64, sample_rate, frame_period=10.0)
        f0 = pw.stonemask(audio_f64, _f0, _t, sample_rate)
        voiced = f0[(f0 >= 70.0) & (f0 <= 350.0)]
        if len(voiced) > 10:
            median_f0 = float(np.median(voiced))
            voiced_ratio = len(voiced) / max(len(f0), 1)
            return median_f0, voiced_ratio
    except Exception:
        pass

    # 3. Fallback mathématique NumPy pur : Autocorrélation normalisée par fenêtre (Zero-dependency)
    try:
        frame_len = int(sample_rate * 0.035)  # fenêtre de 35ms
        hop_len = int(sample_rate * 0.010)    # saut de 10ms
        min_lag = int(sample_rate / 350.0)    # 350 Hz max (aigu)
        max_lag = int(sample_rate / 70.0)     # 70 Hz min (grave)

        f0_candidates = []
        total_frames = 0

        for i in range(0, len(audio_np) - frame_len, hop_len):
            total_frames += 1
            frame = audio_np[i:i + frame_len]
            # Fenêtrage Hanning
            frame = frame * np.hanning(len(frame))
            energy = np.sum(frame ** 2)
            if energy < 1e-4:
                continue

            # Corrélation croisée
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(frame) - 1:]

            if max_lag < len(corr):
                search_region = corr[min_lag:max_lag]
                peak_idx = np.argmax(search_region) + min_lag
                peak_val = corr[peak_idx]

                # Seuil de voisement
                if corr[0] > 0 and (peak_val / corr[0]) > 0.35:
                    freq = sample_rate / peak_idx
                    if 70.0 <= freq <= 350.0:
                        f0_candidates.append(freq)

        if len(f0_candidates) > 8:
            median_f0 = float(np.median(f0_candidates))
            voiced_ratio = len(f0_candidates) / max(total_frames, 1)
            return median_f0, voiced_ratio
    except Exception as e:
        logger.warning(f"Fallback Pitch Extraction error: {e}")

    return None, 0.0


class VoiceGenderDetector:
    def __init__(self, model_id="alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"):
        self.model_id = model_id
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speaker_details = {}

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
        self.speaker_details = {}
        
        speaker_segments = {}
        for segment in diarization_result.get("segments", []):
            speaker = segment.get("speaker")
            if speaker:
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((segment["start"], segment["end"]))
        
        try:
            # Lecture sécurisée avec soundfile
            audio_data, sample_rate = sf.read(audio_path)
            if audio_data.ndim == 1:
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(audio_data).transpose(0, 1)
            waveform = waveform.to(torch.float32)
        except Exception as e:
            logger.error(f"Error loading audio for gender detection: {e}")
            return {}

        total_frames = waveform.shape[1]
        speaker_genders = {}

        # =========================================================================
        # 1. SMART SAMPLER : SÉLECTION GLOBALE DES MEILLEURS SEGMENTS
        # On ne se limite plus aux premières secondes chronologiques de la vidéo !
        # =========================================================================
        for speaker, segments in speaker_segments.items():
            valid_candidates = []

            for start, end in segments:
                duration = end - start
                # On ignore les micro-bruits de moins de 0.8s
                if duration < 0.8:
                    continue

                start_frame = min(int(start * sample_rate), total_frames)
                end_frame = min(int(end * sample_rate), total_frames)
                if start_frame >= end_frame:
                    continue

                chunk = waveform[:, start_frame:end_frame]
                # Calcul de l'énergie RMS pour éviter les silences ou bruits de fond
                rms_energy = float(torch.sqrt(torch.mean(chunk ** 2)).item())
                if rms_energy < 0.005:
                    continue

                # Score de qualité : favorise les segments longs et clairs (jusqu'à 4s max par phrase)
                quality_score = min(duration, 4.0) * (rms_energy ** 0.5)
                valid_candidates.append({
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "chunk": chunk,
                    "quality": quality_score
                })

            # Si aucun segment >= 0.8s n'est trouvé, fallback sur les segments >= 0.4s
            if not valid_candidates:
                for start, end in segments:
                    duration = end - start
                    if duration < 0.4:
                        continue
                    start_frame = min(int(start * sample_rate), total_frames)
                    end_frame = min(int(end * sample_rate), total_frames)
                    if start_frame >= end_frame:
                        continue
                    chunk = waveform[:, start_frame:end_frame]
                    valid_candidates.append({
                        "start": start,
                        "end": end,
                        "duration": duration,
                        "chunk": chunk,
                        "quality": duration
                    })

            if not valid_candidates:
                logger.warning(f"No valid audio segments found for speaker {speaker}")
                speaker_genders[speaker] = "unknown"
                self.speaker_details[speaker] = {"gender": "unknown", "pitch_f0": None, "ai_score": 0.0}
                continue

            # Tri par qualité décroissante pour garder les MEILLEURS segments de toute la vidéo
            valid_candidates.sort(key=lambda x: x["quality"], reverse=True)

            # Assemblage de 6 à 10 secondes optimales
            selected_chunks = []
            accumulated_sec = 0.0
            for cand in valid_candidates:
                selected_chunks.append(cand["chunk"])
                accumulated_sec += cand["duration"]
                if accumulated_sec >= 8.0:
                    break

            combined_waveform = torch.cat(selected_chunks, dim=1)

            # Ré-échantillonnage à 16000 Hz si nécessaire
            if sample_rate != 16000:
                import torchaudio
                transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                combined_waveform = transform(combined_waveform)

            # Conversion en Mono et normalisation crête
            mono_waveform = combined_waveform.mean(dim=0, keepdim=True)
            peak = torch.max(torch.abs(mono_waveform))
            if peak > 0.0:
                mono_waveform = (mono_waveform / peak) * 0.9

            audio_np = mono_waveform.squeeze(0).cpu().numpy()

            # =====================================================================
            # 2. ANALYSE PHYSIQUE : MESURE DU PITCH ACOUSTIQUE F0 (HERTZ)
            # =====================================================================
            f0_pitch, voiced_ratio = extract_f0_pitch(audio_np, 16000)

            # =====================================================================
            # 3. ANALYSE IA : CLASSIFICATEUR DEEP LEARNING (WAV2VEC2)
            # =====================================================================
            temp_wav_path = f"temp_gender_{speaker}.wav"
            sf.write(temp_wav_path, audio_np, 16000)

            ai_label = "unknown"
            ai_score = 0.0
            try:
                predictions = self.classifier(temp_wav_path)
                ai_label = predictions[0]["label"].lower()
                ai_score = float(predictions[0]["score"])
            except Exception as e:
                logger.error(f"Classifier inference failed for {speaker}: {e}")
            finally:
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)

            # =====================================================================
            # 4. ARBITRAGE HYBRIDE (PHYSIQUE + IA)
            # =====================================================================
            final_gender = ai_label

            if f0_pitch is not None and f0_pitch > 0:
                # Zones physiologiques réelles des cordes vocales adultes :
                # Homme : ~85-155 Hz | Femme : ~165-265 Hz | Zone d'ambiguïté : 155-165 Hz
                if f0_pitch < 150.0:
                    pitch_verdict = "male"
                elif f0_pitch > 168.0:
                    pitch_verdict = "female"
                else:
                    pitch_verdict = "ambiguous"

                if pitch_verdict == "female" and ai_label == "male" and f0_pitch >= 175.0:
                    # Correction physique : l'IA s'est trompée, mais la voix est clairement aiguë (femme)
                    final_gender = "female"
                    logger.info(f"Physiological Pitch Override for {speaker}: AI said '{ai_label}', but Pitch is {f0_pitch:.1f}Hz -> Corrected to FEMALE")
                elif pitch_verdict == "male" and ai_label == "female" and f0_pitch <= 142.0:
                    # Correction physique : l'IA s'est trompée, mais la voix est clairement grave (homme)
                    final_gender = "male"
                    logger.info(f"Physiological Pitch Override for {speaker}: AI said '{ai_label}', but Pitch is {f0_pitch:.1f}Hz -> Corrected to MALE")
                elif pitch_verdict in ["male", "female"]:
                    # Si l'IA et le Pitch sont d'accord
                    final_gender = pitch_verdict
                else:
                    # Zone ambiguë (150-168 Hz), on se fie à l'IA
                    final_gender = ai_label if ai_label in ["male", "female"] else ("female" if f0_pitch > 160 else "male")
            else:
                final_gender = ai_label if ai_label in ["male", "female"] else "unknown"

            speaker_genders[speaker] = final_gender
            self.speaker_details[speaker] = {
                "gender": final_gender,
                "pitch_f0": round(f0_pitch, 1) if f0_pitch else None,
                "ai_label": ai_label,
                "ai_score": round(ai_score, 2),
                "voiced_ratio": round(voiced_ratio, 2)
            }
            logger.info(
                f"Speaker {speaker} detected as: {final_gender.upper()} "
                f"(Pitch: {f'{f0_pitch:.1f}Hz' if f0_pitch else 'N/A'}, AI: {ai_label} @ {ai_score:.2f})"
            )

        # =========================================================================
        # 5. COMPARATEUR RELATIF MULTI-LOCUTEURS (Duo Homme / Femme)
        # Si deux locuteurs ont été classés dans le même genre mais qu'un écart
        # de pitch flagrant existe entre eux, on corrige automatiquement !
        # =========================================================================
        speakers_list = list(speaker_genders.keys())
        if len(speakers_list) == 2:
            spk0, spk1 = speakers_list[0], speakers_list[1]
            p0 = self.speaker_details[spk0].get("pitch_f0")
            p1 = self.speaker_details[spk1].get("pitch_f0")

            if p0 and p1:
                pitch_diff = abs(p0 - p1)
                # Un écart de plus de 35 Hz indique une différence physiologique nette
                if pitch_diff >= 35.0:
                    higher_spk = spk0 if p0 > p1 else spk1
                    lower_spk = spk1 if p0 > p1 else spk0

                    if speaker_genders[higher_spk] == speaker_genders[lower_spk]:
                        logger.info(
                            f"Relative Contrast Correction between {spk0} ({p0:.1f}Hz) and {spk1} ({p1:.1f}Hz): "
                            f"{higher_spk} is corrected to FEMALE, {lower_spk} is confirmed as MALE."
                        )
                        speaker_genders[higher_spk] = "female"
                        speaker_genders[lower_spk] = "male"
                        self.speaker_details[higher_spk]["gender"] = "female"
                        self.speaker_details[lower_spk]["gender"] = "male"

        return speaker_genders


def auto_assign_voices(speaker_genders, target_language="french", default_male="fr-FR-HenriNeural-Male", default_female="fr-FR-DeniseNeural-Female"):
    """
    Assigne intelligemment les voix de doublage.
    Évite d'attribuer deux fois la même voix d'homme par défaut si un contraste est possible.
    """
    assigned_voices = {}
    speakers = list(speaker_genders.keys())

    for idx, (speaker, gender) in enumerate(speaker_genders.items()):
        if gender == "female":
            assigned_voices[speaker] = default_female
        elif gender == "male":
            assigned_voices[speaker] = default_male
        else:
            # Fallback contrasté si le genre est inconnu :
            # Si le premier locuteur est un homme, on propose la femme au second pour diversifier
            if idx > 0 and assigned_voices.get(speakers[0]) == default_male:
                assigned_voices[speaker] = default_female
            else:
                assigned_voices[speaker] = default_male
            
    return assigned_voices
