import os
import re
import time
import shutil
import logging
import tempfile
import numpy as np
import soundfile as sf
import subprocess

logger = logging.getLogger("soni_translate.fastdub")

# ==============================================================================
# MAPPING INTELLIGENT DES ÉMOTIONS (FRANÇAIS -> FISH AUDIO TAGS)
# ==============================================================================
FISH_EMOTIONS_MAP = {
    "admiratif": "excited",
    "enthousiaste": "excited",
    "joyeux": "excited",
    "heureux": "excited",
    "charmeur": "calm",
    "séducteur": "calm",
    "dragueur": "calm",
    "amoureux": "calm",
    "curieux": "calm",
    "calme": "calm",
    "posé": "calm",
    "neutre": "calm",
    "confiant": "calm",
    "froide": "calm",
    "distante": "calm",
    "méprisante": "calm",
    "hautaine": "calm",
    "hésitante": "calm",
    "gênée": "calm",
    "intéressée": "excited",
    "mielleuse": "excited",
    "surpris": "surprised",
    "étonné": "surprised",
    "choqué": "surprised",
    "abasourdi": "surprised",
    "amusé": "laughing",
    "riant": "laughing",
    "moqueur": "laughing",
    "rigole": "laughing",
    "taquin": "laughing",
    "ironique": "laughing",
    "sarcastique": "laughing",
    "agacé": "angry",
    "énervé": "angry",
    "colère": "angry",
    "furieux": "angry",
    "irrité": "angry",
    "triste": "sad",
    "déçu": "sad",
    "désespéré": "sad",
    "chuchote": "whisper",
    "discret": "whisper",
    "secret": "whisper",
    "soupir": "sigh",
    "soulagé": "sigh",
    "fatigué": "sigh",
}

VALID_FISH_TAGS = {"excited", "laughing", "whisper", "angry", "surprised", "sad", "sigh", "calm"}

def normalize_emotion_for_fish(raw_tag):
    """Convertit une étiquette d'émotion en balise native Fish Audio."""
    if not raw_tag:
        return "[calm]"
    tag_clean = raw_tag.lower().strip("[] \t\r\n")
    if tag_clean in VALID_FISH_TAGS:
        return f"[{tag_clean}]"
    mapped = FISH_EMOTIONS_MAP.get(tag_clean, "calm")
    return f"[{mapped}]"


def get_fastdub_voices():
    """Récupère la liste des voix disponibles dans voice_library."""
    voices = []
    candidate_dirs = [
        "voice_library",
        os.path.join("soni_translate", "voice_library"),
        "/content/SoniTranslate/voice_library"
    ]
    for d in candidate_dirs:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.endswith((".wav", ".mp3", ".flac")):
                    name = os.path.splitext(f)[0]
                    if name not in voices:
                        voices.append(name)
    if not voices:
        voices = ["ref_homme_fr", "ref_femme_fr"]
    ordered = []
    for pref in ["ref_homme_fr", "ref_femme_fr"]:
        if pref in voices:
            ordered.append(pref)
    for v in voices:
        if v not in ordered:
            ordered.append(v)
    return ordered


# ==============================================================================
# UTILITAIRES HORODATAGE SRT
# ==============================================================================
def seconds_to_srt_time(seconds):
    """Convertit des secondes flottantes en chaîne d'horodatage SRT 'HH:MM:SS,mmm'."""
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def srt_time_to_seconds(time_str):
    """Convertit 'HH:MM:SS,mmm' ou 'HH:MM:SS.mmm' en secondes flottantes."""
    time_str = time_str.strip().replace(',', '.')
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return float(time_str)


# ==============================================================================
# ==============================================================================
# DÉCOUPAGE INTELLIGENT DES SEGMENTS EN PHRASES & TOURS DE PAROLE (1-3s)
# ==============================================================================
def split_segment_into_dialogue_sentences(seg):
    """
    Découpe un gros segment de transcription (ex: 14s) en répliques courtes et rythmées (1 à 3s) :
    1. Si des mots horodatés ('words') sont disponibles via Wav2Vec2 ou Gemini :
       Regroupe les mots en phrases en coupant sur la ponctuation ([.?!…]) et sur les silences inter-mots (>0.35s).
    2. Si aucun mot horodaté n'est disponible (Whisper brut) :
       Découpe le texte sur les frontières de phrases ([.?!…]) et distribue le temps proportionnellement
       à la longueur des phrases.
    """
    raw_text = str(seg.get("text", "")).strip()
    if not raw_text:
        return []

    words = seg.get("words", [])
    speaker = seg.get("speaker", "SPEAKER_00")
    seg_start = float(seg.get("start", 0.0))
    seg_end = float(seg.get("end", 0.0))
    seg_dur = max(0.1, seg_end - seg_start)

    # 1. CAS AVEC MOTS HORODATÉS (Wav2Vec2 ou Gemini ASR)
    valid_words = [
        w for w in words 
        if isinstance(w, dict) and (w.get("word") or w.get("text")) and "start" in w and "end" in w
    ]
    if valid_words and len(valid_words) >= 2:
        sub_segs = []
        cur_words = []
        for i, w in enumerate(valid_words):
            w_text = str(w.get("word") or w.get("text") or "").strip()
            if not w_text:
                continue
            cur_words.append(w)
            
            ends_sentence = w_text[-1:] in ['.', '!', '?', '…']
            pause_after = False
            if i + 1 < len(valid_words):
                next_w = valid_words[i + 1]
                try:
                    if float(next_w.get("start", 0)) - float(w.get("end", 0)) >= 0.35:
                        pause_after = True
                except (ValueError, TypeError):
                    pass

            if (ends_sentence or pause_after) and len(cur_words) >= 1:
                w_str = " ".join(str(cw.get("word") or cw.get("text") or "").strip() for cw in cur_words)
                s_start = float(cur_words[0]["start"])
                s_end = float(cur_words[-1]["end"])
                if s_end <= s_start:
                    s_end = s_start + 0.4
                sub_segs.append({
                    "start": round(s_start, 3),
                    "end": round(s_end, 3),
                    "text": w_str,
                    "speaker": speaker,
                    "words": cur_words
                })
                cur_words = []

        if cur_words:
            w_str = " ".join(str(cw.get("word") or cw.get("text") or "").strip() for cw in cur_words)
            s_start = float(cur_words[0]["start"])
            s_end = float(cur_words[-1]["end"])
            if s_end <= s_start:
                s_end = s_start + 0.4
            sub_segs.append({
                "start": round(s_start, 3),
                "end": round(s_end, 3),
                "text": w_str,
                "speaker": speaker,
                "words": cur_words
            })

        if sub_segs:
            return sub_segs

    # 2. CAS TEXTE BRUT : DÉCOUPAGE RÉGEX ET INTERPOLATION TEMPORELLE
    sentences = [s.strip() for s in re.split(r'(?<=[.!?…])\s+', raw_text) if s.strip()]
    if len(sentences) <= 1:
        return [seg]

    # Regrouper les micro-fragments accidentels (< 2 caractères) avec la phrase précédente
    merged_sentences = []
    for s in sentences:
        if merged_sentences and len(s) <= 2 and not s[-1:] in ['.', '!', '?']:
            merged_sentences[-1] += " " + s
        else:
            merged_sentences.append(s)

    if len(merged_sentences) <= 1:
        return [seg]

    total_weight = sum(max(len(s.split()), 1) for s in merged_sentences)
    sub_segs = []
    cur_start = seg_start

    for s in merged_sentences:
        weight = max(len(s.split()), 1)
        dur = (weight / total_weight) * seg_dur
        s_end = min(seg_end, cur_start + dur)
        if s_end <= cur_start:
            s_end = cur_start + 0.4
        sub_segs.append({
            "start": round(cur_start, 3),
            "end": round(s_end, 3),
            "text": s,
            "speaker": speaker
        })
        cur_start = s_end

    return sub_segs


# ==============================================================================
# 1. TRANSCRIPTION BRUTE VERS FORMAT SRT (WHISPER / GEMINI)
# ==============================================================================
def transcribe_media_to_srt(media_path, asr_model="large-v3", source_lang="en", compute_type="float16"):
    """
    Extrait l'audio d'une vidéo/audio et génère la transcription brute au format SRT,
    scindée intelligemment phrase par phrase (1 à 3 secondes).
    """
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError(f"Fichier média introuvable : {media_path}")

    temp_dir = tempfile.mkdtemp(prefix="fastdub_asr_")
    temp_wav = os.path.join(temp_dir, "extracted_audio.wav")

    logger.info(f"FastDub ASR : Extraction audio depuis {media_path}...")
    cmd_extract = [
        "ffmpeg", "-y", "-i", media_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        temp_wav
    ]
    subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Normalisation du modèle ASR pour correspondre exactement à ASR_MODEL_OPTIONS
    raw_asr = str(asr_model).lower().strip()
    if "gemini" in raw_asr:
        asr_selected = "Gemini 3.5 Transcribe"
    elif "large-v3" in raw_asr or "large_v3" in raw_asr:
        asr_selected = "large-v3"
    elif "large-v2" in raw_asr or "large_v2" in raw_asr:
        asr_selected = "large-v2"
    elif "medium" in raw_asr:
        asr_selected = "medium"
    elif "small" in raw_asr:
        asr_selected = "small"
    elif "base" in raw_asr:
        asr_selected = "base"
    elif "tiny" in raw_asr:
        asr_selected = "tiny"
    else:
        asr_selected = "large-v3"

    import torch
    if compute_type == "float16" and not torch.cuda.is_available():
        compute_type = "int8"

    logger.info(f"FastDub ASR : Lancement de la transcription ({asr_selected}, {compute_type})...")
    from soni_translate.speech_segmentation import transcribe_speech
    
    audio, result = transcribe_speech(
        audio_wav=temp_wav,
        asr_model=asr_selected,
        compute_type=compute_type,
        batch_size=8,
        SOURCE_LANGUAGE=source_lang or "en",
        literalize_numbers=True,
        segment_duration_limit=15
    )

    # Tentative d'alignement phonétique Wav2Vec2 pour obtenir les mots horodatés
    if asr_selected != "Gemini 3.5 Transcribe":
        try:
            from soni_translate.speech_segmentation import align_speech
            logger.info("FastDub ASR : Lancement de l'alignement phonétique Wav2Vec2...")
            aligned_result = align_speech(audio, result)
            if aligned_result and aligned_result.get("segments"):
                result = aligned_result
                logger.info(f"FastDub ASR : Alignement Wav2Vec2 réussi ({len(result.get('segments', []))} segments).")
        except Exception as e_align:
            logger.warning(f"FastDub ASR : Alignement Wav2Vec2 non disponible ({e_align}), utilisation du découpage intelligent de phrases.")

    raw_segments = result.get("segments", [])
    if not raw_segments:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return ""

    # Découpage intelligent de tous les segments géants en répliques naturelles courtes
    fine_segments = []
    for seg in raw_segments:
        sub_segs = split_segment_into_dialogue_sentences(seg)
        fine_segments.extend(sub_segs)

    # Détection si plusieurs locuteurs sont étiquetés (ex: Gemini ou Diarization)
    speakers_in_result = set(s.get("speaker") for s in fine_segments if s.get("speaker"))
    has_multiple_speakers = len(speakers_in_result) > 1

    # Construction du texte SRT
    srt_lines = []
    for idx, seg in enumerate(fine_segments, 1):
        start_str = seconds_to_srt_time(float(seg["start"]))
        end_str = seconds_to_srt_time(float(seg["end"]))
        text = str(seg.get("text", "")).strip()
        spk = seg.get("speaker")
        if has_multiple_speakers and spk:
            text = f"{spk}: {text}"
        srt_lines.append(f"{idx}\n{start_str} --> {end_str}\n{text}\n")

    shutil.rmtree(temp_dir, ignore_errors=True)
    return "\n".join(srt_lines).strip()


# ==============================================================================
# 2. ADAPTATION VIA MASTER PROMPT GEMINI (1 SEUL APPEL API)
# ==============================================================================
FAST_DUB_MASTER_PROMPT = """**RÔLE ET MISSION :**
Tu es l'Adaptateur en Chef et Directeur Artistique de doublage français pour des vidéos virales de type **"Gold Digger Prank"**, caméras cachées et micro-trottoirs de drague (USA).
Ta mission : transformer la transcription anglaise horodatée en un script de doublage français oral, ultra-naturel, percutant et parfaitement synchronisé avec les durées d'origine pour la synthèse vocale TTS.

**ANALYSE ET COMPRÉHENSION NARRATIVE (GOLD DIGGER PRANK) :**
Dans ces vidéos, l'histoire suit un schéma psychologique précis :
1. **Speaker 1 (L'Homme / Le Piégeur / Le Créateur)** :
   - C'est lui qui aborde la fille dans la rue (démarche décontractée, charmeur ou taquin).
   - C'est lui qui possède la voiture de luxe / supercar (Lamborghini, Ferrari, BMW...).
   - **RÈGLE CRUCIALE DES APARTÉS :** Quand quelqu'un parle seul en voix basse face caméra ("Another failed attempt", "She's walking back", "Whole energy change", etc.), c'est OBLIGATOIREMENT Speaker 1 qui commente la scène ! Balise [whisper] ou [amuse].
2. **Speaker 2 (La Fille abordée)** :
   - *Cas fréquent (Gold Digger)* : Au début, elle est froide, réticente, distante ou expéditive ("I'm busy", "I have a boyfriend", "My name's not here"). Dès qu'elle voit la voiture ou le signe de richesse, son énergie bascule radicalement : elle devient curieuse, mielleuse, charmeuse, flatteuse ("Is this your car?", "You must got money", "Can I get a ride?").
   - *Cas des filles sympas* : Certaines filles restent naturelles, polies ou souriantes dès le début. Ne force pas l'agressivité si elle est douce.
3. **VIDÉOS MULTIPLES / COMPILATIONS & PHASES DE TRANSITION (CTA / CONSEILS) :**
   - La vidéo peut compiler 2 ou plusieurs pranks à la suite.
   - Entre deux pranks ou au milieu/fin de la vidéo, il y a souvent une **phase transitoire ou d'interaction avec le public** :
     * Appels à l'action : *"Drop a like, subscribe guys"*, *"Lâchez un pouce bleu, abonnez-vous les gars !"*, *"Let's go to the next one!"*, *"C'est parti pour la suivante !"*.
     * Conseils aux viewers ou débriefing : *"Ne vous faites jamais avoir par les michtos"*, *"Dites-moi ce que vous en pensez en commentaire"*.
   - **RÈGLE FORMELLE :** Tout appel à s'abonner/liker, conseil face caméra ou phrase de transition vers la suite est **OBLIGATOIREMENT attribué à Speaker 1 (Le Créateur)**, généralement avec la balise `[excited]`, `[amuse]` ou `[calme]`. Ce n'est JAMAIS la fille !
   - Après cette transition, un nouveau prank démarre : Speaker 1 reste le créateur, mais Speaker 2 devient une **nouvelle fille** (la dynamique d'approche recommence donc à zéro).
4. **Cohérence absolue des rôles :**
   Ne confonds JAMAIS Speaker 1 (le créateur) et Speaker 2 (la fille) au cours de la vidéo.

**RÈGLES D'OR DU DOUBLAGE RYTHMIQUE (LABIALE & TEMPO) :**
1. **CONTRAINTE DE DURÉE (VÉLOCITÉ MAXIMALE) :**
   Le français est naturellement 20% plus long que l'anglais. Tu dois CONDENSER et PUNCHER les répliques pour qu'elles rentrent pile dans le chronométrage sans que le TTS ait besoin d'accélérer :
   - Réplique courte (< 1.5 seconde) : Maximum 3 à 5 mots percutants.
   - Réplique moyenne (2.0 à 3.5 secondes) : Maximum 6 à 9 mots.
   - Réplique longue (> 4.0 secondes) : Maximum 12 à 15 mots bien rythmés.
2. **STYLE STREET / PARLÉ VIVANT (EXIT LE FRANÇAIS SCOLAIRE) :**
   - Tutoiement direct et immédiat.
   - Élimine les "ne" de négation : "J'sais pas", "T'as pas vu", "C'est mort", "Laisse tomber".
   - Vocabulaire naturel et percutant : "Caisse" / "Bagnole" au lieu de "Voiture", "Fric" / "Cash" / "Oseille" au lieu de "Argent", "Râteau" au lieu de "Rejet".
   - Ponctuation orale : utilise des virgules et des points de suspension pour donner du souffle et du rythme au TTS.

**BALISES D'ÉMOTIONS (NATIVES POUR LE TTS) :**
Place obligatoirement une balise entre crochets au début de chaque réplique :
- Pour le mec : [charmeur], [amuse], [calme], [whisper], [moqueur], [taquin], [ironique]
- Pour la fille : [froide], [distante], [surpris], [curieux], [charmeur], [mielleuse], [amuse], [agace], [choque]
- Balises audio directes supportées : [excited], [laughing], [whisper], [angry], [surprised], [sad], [calm]

**FORMAT STRICT DE SORTIE :**
Conserve strictement la numérotation et les horodatages originaux de chaque segment.
Numéro
HH:MM:SS,mmm --> HH:MM:SS,mmm
Speaker X: [émotion] Texte français adapté

(Ne mets JAMAIS de crochets autour de 'Speaker X'. Ne renvoie AUCUN commentaire, explication ou balise markdown ```. Commence directement par le numéro 1)."""

def adapt_script_gemini(srt_text, custom_prompt=None, model_name="gemini-3.5-flash-lite"):
    """
    Envoie le script SRT complet à Gemini en un seul appel API pour produire le script de doublage adapté.
    """
    if not srt_text or not srt_text.strip():
        return ""

    from soni_translate.translate_segments import get_gemini_api_keys
    keys = get_gemini_api_keys()
    if not keys:
        raise ValueError("Clé GOOGLE_API_KEY manquante. Renseigne ta clé Gemini pour utiliser l'adaptation automatique.")

    from google import genai
    from google.genai import types

    sys_instruction = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else FAST_DUB_MASTER_PROMPT

    candidates = [model_name, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
    last_err = None

    for key in keys:
        client = genai.Client(api_key=key)
        for m in candidates:
            try:
                logger.info(f"FastDub Gemini : Envoi du script complet ({len(srt_text)} car.) vers '{m}'...")
                response = client.models.generate_content(
                    model=m,
                    contents=srt_text.strip(),
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        temperature=0.3,
                        max_output_tokens=8192
                    )
                )
                raw_out = response.text.strip()
                # Nettoyage des balises markdown si le LLM en a ajouté
                clean_out = re.sub(r'^```(?:srt)?\s*', '', raw_out)
                clean_out = re.sub(r'\s*```$', '', clean_out).strip()
                return clean_out
            except Exception as e:
                last_err = e
                logger.warning(f"FastDub Gemini tentative '{m}' a échoué ({e}). Essai suivant...")
                continue

    # Fallback optionnel Groq si Gemini échoue
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")
    if groq_key:
        try:
            logger.info("FastDub : Bascule de secours sur Groq LLaMA-3.3 70B...")
            from openai import OpenAI
            import httpx
            g_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key, http_client=httpx.Client(timeout=60))
            chat = g_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_instruction},
                    {"role": "user", "content": srt_text.strip()}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=8000
            )
            raw_groq = chat.choices[0].message.content.strip()
            clean_groq = re.sub(r'^```(?:srt)?\s*', '', raw_groq)
            return re.sub(r'\s*```$', '', clean_groq).strip()
        except Exception as ge:
            logger.error(f"Secours Groq FastDub échoué : {ge}")

    if last_err:
        raise last_err
    return ""


# ==============================================================================
# 3. PARSEUR DE SCRIPT DE DOUBLAGE FASTDUB
# ==============================================================================
def parse_dubbing_script(script_text):
    """
    Parse le script de doublage formaté :
    1
    00:00:00,249 --> 00:00:01,171
    Speaker 1: [admiratif] Waouh, t'es magnifique.
    """
    blocks = re.split(r'\n\s*\n', script_text.strip())
    parsed_dialogues = []

    time_pattern = re.compile(r'(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})')
    speaker_pattern = re.compile(r'^(?:Speaker\s*(\d+)|(Homme|Femme)|SPEAKER_(\d+))\s*:\s*(.*)$', re.IGNORECASE)
    emotion_pattern = re.compile(r'^\[([^\]]+)\]\s*(.*)$')

    for blk in blocks:
        lines = [line.strip() for line in blk.strip().split('\n') if line.strip()]
        if len(lines) < 2:
            continue

        # Recherche de la ligne de temps
        time_match = None
        time_line_idx = -1
        for idx, l in enumerate(lines):
            m = time_pattern.search(l)
            if m:
                time_match = m
                time_line_idx = idx
                break

        if not time_match or time_line_idx == -1:
            continue

        start_sec = srt_time_to_seconds(time_match.group(1))
        end_sec = srt_time_to_seconds(time_match.group(2))
        if end_sec <= start_sec:
            end_sec = start_sec + 1.0

        # Le texte de la réplique se trouve après la ligne de temps
        text_lines = [l.strip() for l in lines[time_line_idx + 1:] if l.strip()]
        if not text_lines:
            continue

        cues = []
        for tl in text_lines:
            if speaker_pattern.match(tl):
                cues.append(tl)
            elif cues:
                cues[-1] += " " + tl
            else:
                cues.append(tl)

        total_cue_weight = sum(max(len(c.split()), 1) for c in cues)
        cue_start = start_sec
        block_dur = max(end_sec - start_sec, 0.4)

        for c_text in cues:
            weight = max(len(c_text.split()), 1)
            dur = (weight / max(total_cue_weight, 1)) * block_dur
            cue_end = min(end_sec, cue_start + dur)
            if cue_end <= cue_start:
                cue_end = cue_start + 0.4

            speaker_id = 1
            emotion_tag = "[calm]"
            speech_text = c_text

            spk_match = speaker_pattern.match(c_text)
            if spk_match:
                if spk_match.group(1):
                    speaker_id = int(spk_match.group(1))
                elif spk_match.group(2):
                    speaker_id = 2 if "femme" in spk_match.group(2).lower() else 1
                elif spk_match.group(3):
                    speaker_id = int(spk_match.group(3)) + 1
                speech_text = spk_match.group(4).strip()

            emo_match = emotion_pattern.match(speech_text)
            if emo_match:
                raw_emo = emo_match.group(1)
                emotion_tag = normalize_emotion_for_fish(raw_emo)
                speech_text = emo_match.group(2).strip()

            # Nettoyage ultime du texte pour le TTS
            speech_text = re.sub(r'^(?:Speaker\s*\d+|Homme|Femme)\s*:\s*', '', speech_text, flags=re.IGNORECASE).strip()

            parsed_dialogues.append({
                "start": round(cue_start, 3),
                "end": round(cue_end, 3),
                "duration": round(cue_end - cue_start, 3),
                "speaker": speaker_id,
                "emotion": emotion_tag,
                "text": speech_text,
                "full_tts_text": f"{emotion_tag} {speech_text}".strip()
            })
            cue_start = cue_end


    return parsed_dialogues


# ==============================================================================
# 4. SYNTHÈSE VOCALE ET MIXAGE VIDÉO FLASH
# ==============================================================================
def render_fastdub_video(
    media_path,
    script_text,
    speaker1_voice="ref_homme_fr",
    speaker2_voice="ref_femme_fr",
    tts_engine="fish_audio",
    background_volume=0.15,
    progress=None
):
    """
    Synthétise toutes les répliques avec Fish Audio (cache VRAM) et les monte sur la vidéo avec FFmpeg.
    """
    if not media_path or not os.path.exists(media_path):
        raise FileNotFoundError(f"Fichier vidéo/audio introuvable : {media_path}")

    dialogues = parse_dubbing_script(script_text)
    if not dialogues:
        raise ValueError("Aucune réplique valide trouvée dans le script de doublage.")

    logger.info(f"FastDub Render : {len(dialogues)} répliques à synthétiser et caler...")
    if progress:
        progress(0.05, desc="Préparation du mixage FastDub...")

    work_dir = tempfile.mkdtemp(prefix="fastdub_render_")
    chunks_dir = os.path.join(work_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    # Obtenir la durée totale du média source
    info_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", media_path
    ]
    try:
        dur_res = subprocess.run(info_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        total_media_duration = float(dur_res.stdout.strip())
    except Exception:
        total_media_duration = max([d["end"] for d in dialogues]) + 2.0

    target_sr = 44100
    total_samples = int(np.ceil(total_media_duration * target_sr))
    master_audio = np.zeros(total_samples, dtype=np.float32)

    # Initialisation du TTS Provider
    fish_provider = None
    if tts_engine == "fish_audio":
        try:
            from soni_translate.tts_providers.fish_audio import FishAudioProvider
            fish_provider = FishAudioProvider()
        except ImportError:
            from soni_translate.tts_providers.fish_audio import FishAudioTTSProvider
            fish_provider = FishAudioTTSProvider()

    total_diag = len(dialogues)
    for idx, diag in enumerate(dialogues):
        if progress:
            progress(0.1 + (0.75 * (idx / max(total_diag, 1))), desc=f"Synthèse réplique {idx+1}/{total_diag} ({diag['emotion']})...")

        voice_chosen = speaker2_voice if diag["speaker"] == 2 else speaker1_voice
        tts_text = diag["full_tts_text"]
        out_chunk_path = os.path.join(chunks_dir, f"chunk_{idx:04d}.wav")

        try:
            if tts_engine == "fish_audio" and fish_provider:
                fish_provider.generate(
                    text=tts_text,
                    voice=voice_chosen,
                    target_lang="fr",
                    output_file=out_chunk_path
                )
            else:
                # Secours Edge-TTS si demandé
                import edge_tts
                import asyncio
                clean_raw_text = diag["text"]
                edge_v = "fr-FR-DeniseNeural" if diag["speaker"] == 2 else "fr-FR-HenriNeural"
                asyncio.run(edge_tts.Communicate(clean_raw_text, edge_v).save(out_chunk_path))

            # Lecture de l'audio généré
            if os.path.exists(out_chunk_path) and os.path.getsize(out_chunk_path) > 1000:
                audio_data, sr = sf.read(out_chunk_path)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)

                # Rééchantillonnage vers 44.1kHz si nécessaire
                if sr != target_sr:
                    import librosa
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)

                # Ajustement de tempo si la réplique dépasse trop la durée allouée (+20%)
                target_chunk_len = int(diag["duration"] * target_sr)
                if len(audio_data) > target_chunk_len * 1.25 and target_chunk_len > 0:
                    speed_factor = min(len(audio_data) / target_chunk_len, 1.4)
                    speed_wav = os.path.join(chunks_dir, f"speed_{idx:04d}.wav")
                    cmd_speed = [
                        "ffmpeg", "-y", "-i", out_chunk_path,
                        "-filter:a", f"atempo={speed_factor:.3f}",
                        speed_wav
                    ]
                    subprocess.run(cmd_speed, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if os.path.exists(speed_wav):
                        audio_data, _ = sf.read(speed_wav)
                        if audio_data.ndim > 1:
                            audio_data = audio_data.mean(axis=1)

                # Placement précis sur la piste master aux millisecondes indiquées
                start_sample = int(diag["start"] * target_sr)
                end_sample = min(start_sample + len(audio_data), total_samples)
                length_to_paste = end_sample - start_sample

                if length_to_paste > 0:
                    master_audio[start_sample:end_sample] += audio_data[:length_to_paste]

        except Exception as e_tts:
            logger.error(f"Erreur synthèse FastDub réplique {idx+1}: {e_tts}")

    # Normalisation de la piste voix
    max_amp = float(np.abs(master_audio).max())
    if max_amp > 0.98:
        master_audio = master_audio / max_amp * 0.95

    dubbed_vocals_wav = os.path.join(work_dir, "master_dubbed_vocals.wav")
    sf.write(dubbed_vocals_wav, master_audio, target_sr)

    if progress:
        progress(0.90, desc="Mixage avec le fond sonore original...")

    # Mixage avec le fond sonore original (ducking de la piste d'origine à background_volume)
    mixed_audio_wav = os.path.join(work_dir, "final_mixed_audio.wav")
    vol_str = f"{background_volume:.2f}"
    cmd_mix = [
        "ffmpeg", "-y",
        "-i", media_path,
        "-i", dubbed_vocals_wav,
        "-filter_complex", f"[0:a]volume={vol_str}[bg];[bg][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "[aout]",
        "-ac", "2", "-ar", "44100",
        mixed_audio_wav
    ]
    subprocess.run(cmd_mix, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    def has_video_stream(fpath):
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            fpath
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return "video" in res.stdout.lower()
        except Exception:
            ext = os.path.splitext(fpath)[1].lower()
            return ext in [".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv"]

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(media_path))[0]
    is_video = has_video_stream(media_path)

    if is_video:
        if progress:
            progress(0.95, desc="Rendu vidéo final (Muxing instantané -c:v copy)...")
        final_output_path = os.path.join(output_dir, f"{base_name}_FastDub.mp4")
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", media_path,
            "-i", mixed_audio_wav,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            final_output_path
        ]
        subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    else:
        if progress:
            progress(0.95, desc="Rendu audio final (Encodage MP3)...")
        final_output_path = os.path.join(output_dir, f"{base_name}_FastDub.mp3")
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", mixed_audio_wav,
            "-c:a", "libmp3lame", "-b:a", "192k",
            final_output_path
        ]
        subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    logger.info(f"FastDub terminé avec succès ! Média généré : {final_output_path}")
    if progress:
        progress(1.0, desc="FastDub terminé avec succès !")

    return final_output_path
