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
        return ""
    tag_clean = raw_tag.lower().strip("[] \t\r\n")
    if not tag_clean:
        return ""
    if tag_clean in VALID_FISH_TAGS:
        return f"[{tag_clean}]"
    mapped = FISH_EMOTIONS_MAP.get(tag_clean, "")
    if mapped:
        return f"[{mapped}]"
    return ""


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
# DÉCOUPAGE INTELLIGENT DES SEGMENTS EN PHRASES & TOURS DE PAROLE (1-4s)
# ==============================================================================
DANGLING_WORDS = {
    'you', 'i', 'he', 'she', 'we', 'they', 'it', 'my', 'your', 'his', 'her', 'our', 'their',
    'a', 'an', 'the', 'and', 'or', 'but', 'so', 'if', 'because', 'why', 'what', 'who', 'how',
    'when', 'where', 'to', 'in', 'on', 'at', 'for', 'with', 'by', 'about', 'tu', 'je', 'il',
    'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'mon', 'ton', 'son', 'ma', 'ta', 'sa',
    'mes', 'tes', 'ses', 'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'donc',
    'car', 'que', 'qui', 'ce', 'cet', 'cette', 'ces'
}

ABBREVIATIONS = {
    'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'st.', 'vs.', 'u.s.', 'etc.', 'approx.', 'jr.', 'sr.'
}

STANDALONE_WORDS = {
    'yes', 'no', 'yeah', 'yep', 'nope', 'oui', 'non', 'ouais', 'cool', 'exactement',
    'merci', 'thanks', 'sure', 'ok', 'okay', 'bye', 'hello', 'salut', 'wow', 'yo', 'right'
}

def split_segment_into_dialogue_sentences(seg):
    """
    Découpe un gros segment de transcription (ex: 14s) en répliques naturelles et rythmées (1 à 4s) :
    1. Si des mots horodatés ('words') sont disponibles via Wav2Vec2 ou Gemini :
       Regroupe les mots en phrases grammaticales complètes sans jamais découper les sujets/pronoms orphelins.
    2. Si aucun mot horodaté n'est disponible (Whisper brut) :
       Découpe le texte sur les frontières de phrases ([.?!…]) et distribue le temps proportionnellement.
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
            
            clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', w_text).lower()
            ends_sentence = w_text[-1:] in ['.', '!', '?', '…'] and w_text.lower() not in ABBREVIATIONS
            
            pause_after = 0.0
            if i + 1 < len(valid_words):
                next_w = valid_words[i + 1]
                try:
                    pause_after = max(0.0, float(next_w.get("start", 0)) - float(w.get("end", 0)))
                except (ValueError, TypeError):
                    pass

            cur_start = float(cur_words[0]["start"])
            cur_end = float(cur_words[-1]["end"])
            cur_dur = max(0.0, cur_end - cur_start)
            cur_count = len(cur_words)

            should_split = False
            # 1. Ponctuation forte de fin de phrase
            if ends_sentence:
                if clean_word in STANDALONE_WORDS or cur_count >= 2 or cur_dur >= 0.8:
                    should_split = True
            # 2. Vrai silence conversationnel (>= 0.85s) avec phrase consistante
            elif pause_after >= 0.85:
                if clean_word not in DANGLING_WORDS and (cur_count >= 3 or cur_dur >= 1.0):
                    should_split = True
            # 3. Sécurité longueur excessive (>= 4.5s) pour éviter l'emballement TTS
            elif cur_dur >= 4.5:
                if (w_text[-1:] == ',' or pause_after >= 0.4) and cur_count >= 4 and clean_word not in DANGLING_WORDS:
                    should_split = True
                elif cur_dur >= 6.0 and clean_word not in DANGLING_WORDS:
                    should_split = True

            if should_split:
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
            
            # Si le reliquat est un mot isolé ou trop court, le rattacher au segment précédent
            if sub_segs and (len(cur_words) == 1 or (s_end - s_start) < 0.6):
                sub_segs[-1]["end"] = round(s_end, 3)
                sub_segs[-1]["text"] = sub_segs[-1]["text"] + " " + w_str
                if "words" in sub_segs[-1]:
                    sub_segs[-1]["words"].extend(cur_words)
            else:
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

    # Regrouper les micro-fragments accidentels avec la phrase voisine
    merged_sentences = []
    for s in sentences:
        if merged_sentences and (len(s.split()) <= 2 or len(s) <= 4) and s[-1:] not in ['.', '!', '?']:
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


def merge_only_broken_word_fragments(segments, max_gap=0.9):
    """
    Rattache UNIQUEMENT les fragments grammaticaux incomplets (mots orphelins sans ponctuation)
    à leur phrase suivante (ex: "You" + "look amazing." -> "You look amazing.").
    NE FUSIONNE JAMAIS deux répliques complètes ayant une ponctuation de fin ([.?!…])
    afin de préserver STRICTEMENT la séparation des locuteurs (Speaker 1 / Speaker 2).
    """
    if not segments or len(segments) <= 1:
        return segments

    merged = []
    i = 0
    while i < len(segments):
        cur = dict(segments[i])
        
        while i + 1 < len(segments):
            nxt = segments[i + 1]
            gap = float(nxt["start"]) - float(cur["end"])
            
            cur_text = cur.get("text", "").strip()
            nxt_text = nxt.get("text", "").strip()
            cur_words = cur_text.split()
            
            clean_last = re.sub(r'^[^\w]+|[^\w]+$', '', cur_words[-1] if cur_words else '').lower()
            ends_punct = cur_text[-1:] in ['.', '!', '?', '…']
            
            # Ne fusionne QUE si cur est manifestement un mot brisé/incomplet (sans ponctuation de fin)
            is_broken_fragment = False
            if not ends_punct:
                if len(cur_words) == 1 or clean_last in DANGLING_WORDS:
                    is_broken_fragment = True

            if is_broken_fragment and gap <= max_gap:
                cur["end"] = nxt["end"]
                cur["text"] = f"{cur_text} {nxt_text}".strip()
                if "words" in cur and "words" in nxt:
                    cur["words"].extend(nxt.get("words", []))
                i += 1
            else:
                break
        
        merged.append(cur)
        i += 1

    return merged


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

    # Rattachement exclusif des mots orphelins (ex: "You" + "look amazing.") sans jamais fusionner deux répliques distinctes
    fine_segments = merge_only_broken_word_fragments(fine_segments, max_gap=0.9)

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
Tu es l'Adaptateur en Chef et Directeur Artistique de doublage français pour des vidéos virales de type **"Gold Digger Prank"**, caméras cachées et micro-trottoirs de drague aux USA.
Ta mission : transformer la transcription anglaise horodatée en un script de doublage français oral, ultra-naturel, percutant et parfaitement synchronisé avec les durées d'origine pour la synthèse vocale TTS.

**1. STRUCTURE NARRATIVE & RÔLES :**
- **Speaker 1 (L'Homme / Le Piégeur / Le Créateur)** :
  * Il aborde les filles dans la rue d'un ton détendu, charmeur ou taquin.
  * C'est lui qui possède la supercar / voiture de luxe (Lamborghini, Ferrari, BMW...).
  * C'est lui qui parle à la caméra cachée en aparté et qui fait les transitions pour ses abonnés.
- **Speaker 2 (La Fille abordée)** :
  * *Cas typique (Gold Digger)* : Méfiante, froide, occupée ou distante au début ("I have a boyfriend", "I'm busy", "My name's not here"). Dès qu'elle voit la voiture ou le signe de richesse, son énergie bascule radicalement : elle devient curieuse, mielleuse, charmeuse, flatteuse ("Is this your car?", "You must got money", "Can I get a ride?").
  * *Cas des filles sympas* : Certaines filles restent naturelles, polies ou souriantes dès le début. Adapte leur ton sans forcer d'agressivité.

**2. PHARES D'ANCRAGE LEXICAUX (RÈGLES D'ATTRIBUTION ABSOLUES) :**
Utilise ces expressions anglaises clés comme repères 100% indiscutables dans le texte :
- **Phrases EXCLUSIVES à Speaker 1 (L'Homme)** :
  * Abordage & compliments : *"Excuse me"*, *"How you doing?"*, *"You look amazing"*, *"What's your name?"*, *"Can I get your number / Insta?"*, *"Have a nice day"*, *"Take care"*.
  * Apartés caméra cachée : *"Another failed attempt"*, *"She's walking back"*, *"Look at her"*, *"Whole energy change"*, *"I spoke too soon"*.
  * Réactions propriétaire voiture : *"Yeah, it's mine"*, *"It's a Lambo / Ferrari"*, *"Hop in"*, *"Just kidding, I don't like gold diggers"*, *"I gotta go alone"*.
  * Transitions & CTA viewers : *"Drop a like"*, *"Subscribe guys"*, *"Lâchez un pouce bleu"*, *"Stay safe out there"*.
- **Phrases EXCLUSIVES à Speaker 2 (La Fille)** :
  * Rejet initial : *"I have a boyfriend / a man"*, *"My name's not here"*, *"I'm in a rush / busy"*, *"I'm waiting for someone"*, *"No thank you"*.
  * Intérêt soudain pour la richesse : *"Wait, is this your car?"*, *"What car is that?"*, *"What do you do for a living?"*, *"You must got money"*, *"Can you take me with you?"*, *"You're actually cute"*.

**3. HEURISTIQUE DES ÉCARTS TEMPORELS (SILENCES DU SRT) :**
Analyse les timestamps pour comprendre les mouvements spatiaux des personnages :
- **Écart court (< 1.5s entre deux répliques)** : C'est du **ping-pong conversationnel direct** en face-à-face. Applique l'alternance stricte (Speaker 1 -> Speaker 2 -> Speaker 1 -> Speaker 2).
- **Écart long (> 3.0s de silence dans le SRT)** : C'est une **rupture de scène ou un déplacement physique** :
  * Soit la fille s'est éloignée et le mec marche vers sa voiture en chuchotant à la caméra -> **Speaker 1 [whisper]**.
  * Soit la fille a aperçu la supercar, a fait demi-tour et revient interpeller le mec -> **Speaker 2 [curieux] ou [charmeur]**.

**4. PRINCIPE DU PING-PONG CONVERSATIONNEL (TURN-TAKING) :**
En face-à-face, les répliques s'alternent presque systématiquement :
Speaker 1 -> Speaker 2 -> Speaker 1 -> Speaker 2.
Une répétition du même Speaker (ex: Speaker 1 deux fois de suite) ne doit arriver QUE si :
a) Le créateur commente la scène tout seul face caméra.
b) Le créateur fait un appel aux abonnés / conseil (CTA).
c) Un interlocuteur hésite et complète sa propre pensée.

**5. COMPILATIONS MULTI-PRANKS & PHASES DE TRANSITION :**
- Entre deux pranks ou en cours de vidéo, le créateur s'adresse souvent directement à son audience (*"Lâchez un pouce bleu, abonnez-vous les gars !"*, *"C'est parti pour la suivante !"*, conseils de drague/michtos...).
- **RÈGLE FORMELLE :** Tout appel à l'action (like/abonnement), conseil face caméra ou phrase de transition est **OBLIGATOIREMENT Speaker 1 (Le Créateur)** avec l'émotion `[excited]`, `[amuse]` ou `[calme]`.
- Après cette transition, un nouveau prank démarre : Speaker 1 reste le créateur, mais Speaker 2 devient une **nouvelle fille** (la dynamique repart de zéro).

**6. RÈGLES DE CALIBRAGE RYTHMIQUE (LABIALE & TEMPO) :**
- **VÉLOCITÉ MAXIMALE :** Le français est naturellement plus long que l'anglais. Tu dois CONDENSER et PUNCHER pour respecter strictement la durée originale sans accélérer la voix TTS :
  * Réplique < 1.5s : 3 à 5 mots percutants max.
  * Réplique 2.0s à 3.5s : 6 à 9 mots max.
  * Réplique > 4.0s : 12 à 15 mots max.
- **STYLE STREET / PARLÉ VIVANT :**
  * Tutoiement direct et immédiat.
  * Bannis le français formel : supprime les "ne" de négation ("J'sais pas", "T'as pas vu", "C'est mort", "Laisse tomber").
  * Vocabulaire percutant : "Caisse" / "Bagnole", "Fric" / "Cash" / "Oseille", "Râteau", "Michto".
  * Ponctuation orale : utilise des virgules pour les micro-respirations et des points d'interrogation pour l'intonation TTS.

**7. BALISES D'ÉMOTIONS (NATIVES POUR FISH AUDIO TTS) :**
Place obligatoirement une balise entre crochets au début de chaque réplique :
- Pour le mec : [charmeur], [amuse], [calme], [whisper], [moqueur], [taquin], [ironique], [excited]
- Pour la fille : [froide], [distante], [surpris], [curieux], [charmeur], [mielleuse], [amuse], [agace], [choque]
- Balises natives reconnues : [excited], [laughing], [whisper], [angry], [surprised], [sad], [calm]

**8. EXEMPLES TYPES (FEW-SHOT LEARNING) :**
Exemple 1 (Abordage & Râteau) :
1
00:00:01,186 --> 00:00:03,800
Speaker 1: [charmeur] Salut ! T'es magnifique, c'est quoi ton prénom ?
2
00:00:04,100 --> 00:00:06,200
Speaker 2: [froide] Je suis occupée, et j'ai un copain.
3
00:00:06,500 --> 00:00:08,900
Speaker 1: [calme] Ça marche, passe une bonne journée alors !

Exemple 2 (Aparté & Demi-tour) :
4
00:00:15,000 --> 00:00:20,000
Speaker 1: [whisper] Bon, un râteau de plus... Attends, elle a vu la caisse, elle fait demi-tour direct !

Exemple 3 (Le Pivot de la voiture) :
5
00:00:22,000 --> 00:00:24,500
Speaker 2: [mielleuse] Re-salut... Dis, c'est ta voiture la Lamborghini ?
6
00:00:24,800 --> 00:00:26,500
Speaker 1: [amuse] Ouais, c'est la mienne. Tu kiffes ?
7
00:00:26,800 --> 00:00:29,400
Speaker 2: [charmeur] Elle est folle... T'as l'air d'avoir du fric toi, avoue...

Exemple 4 (Transition / Compilations) :
8
00:01:10,000 --> 00:01:13,200
Speaker 1: [excited] Lâchez un gros pouce bleu et abonnez-vous les gars ! C'est parti pour la suivante !

**FORMAT STRICT DE SORTIE :**
Conserve strictement la numérotation et les horodatages originaux de chaque segment.
Numéro
HH:MM:SS,mmm --> HH:MM:SS,mmm
Speaker X: [émotion] Texte français adapté

**RÈGLE D'OR ABSOLUE : UN SEUL LOCUTEUR PAR BLOC SRT !**
Chaque numéro de bloc SRT correspond à UNE SEULE personne qui parle (Speaker 1 OU Speaker 2).
- INTERDICTION ABSOLUE de mettre deux répliques pour deux personnes dans le même bloc.
- INTERDICTION ABSOLUE d'utiliser des tirets '-' de dialogue pour faire parler deux personnes dans un même bloc (ex: "Attends, c'est ta caisse ? - Ouais, c'est ma caisse." est TOTALEMENT INTERDIT).
- Chaque bloc SRT en entrée est la réplique d'UNE SEULE personne. Traduis UNIQUEMENT ce que dit cette personne dans ce bloc sans inventer la suite ou la réponse !
- Chaque bloc SRT doit comporter EXACTEMENT 3 lignes :
  1. Le numéro
  2. L'horodatage original
  3. "Speaker X: [émotion] Texte français"

(Ne mets JAMAIS de crochets autour de 'Speaker X'. Ne renvoie AUCUN commentaire, explication ou balise markdown ```. Commence directement par le premier numéro de bloc)."""

def enforce_srt_timestamps(generated_srt, original_srt):
    """
    Assure une fidélité temporelle absolue à 100% :
    Pour chaque bloc numéroté k dans le texte généré par l'IA,
    remplace intégralement la ligne d'horodatage par l'horodatage EXACT d'origine
    du bloc k de original_srt, peu importe le format ou les hallucinations du LLM.
    """
    if not generated_srt or not original_srt:
        return generated_srt

    gen_norm = generated_srt.replace('\r\n', '\n').strip()
    orig_norm = original_srt.replace('\r\n', '\n').strip()

    # 1. Extraction de tous les blocs d'origine
    orig_time_map = {}
    raw_orig_blocks = re.split(r'\n\s*\n', orig_norm)
    for b in raw_orig_blocks:
        b_lines = [l.strip() for l in b.split('\n') if l.strip()]
        if len(b_lines) >= 2:
            idx_match = re.match(r'^(\d+)$', b_lines[0])
            time_match = re.search(r'(\d{1,2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{3})', b_lines[1])
            if idx_match and time_match:
                orig_time_map[int(idx_match.group(1))] = time_match.group(1).replace('.', ',')

    if not orig_time_map:
        return generated_srt

    # 2. Reconstruction des blocs générés
    raw_gen_blocks = re.split(r'\n\s*\n', gen_norm)
    reconstructed_blocks = []

    for b in raw_gen_blocks:
        lines = [l.strip() for l in b.split('\n') if l.strip()]
        if not lines:
            continue

        idx = None
        start_line_idx = 0
        for i, l in enumerate(lines):
            m = re.match(r'^(\d+)$', l)
            if m:
                idx = int(m.group(1))
                start_line_idx = i
                break

        if idx is None or idx not in orig_time_map:
            reconstructed_blocks.append("\n".join(lines))
            continue

        orig_time = orig_time_map[idx]
        
        content_lines = []
        found_arrow = False
        for l in lines[start_line_idx + 1:]:
            if '-->' in l and not found_arrow:
                found_arrow = True
                continue
            content_lines.append(l)

        if not found_arrow and len(lines) > start_line_idx + 1:
            content_lines = lines[start_line_idx + 2:]

        content_str = "\n".join(content_lines).strip()
        reconstructed_blocks.append(f"{idx}\n{orig_time}\n{content_str}")

    return "\n\n".join(reconstructed_blocks).strip()


def split_srt_into_cue_chunks(srt_text, chunk_size=70):
    """
    Découpe un texte SRT en sous-blocs de N répliques (cues) complètes
    pour éviter d'atteindre la limite max_output_tokens (8192) des LLM.
    """
    clean_text = srt_text.replace('\r\n', '\n').strip()
    raw_blocks = re.split(r'\n\s*\n', clean_text)
    valid_blocks = [b.strip() for b in raw_blocks if b.strip() and re.search(r'\d{1,2}:\d{2}:\d{2}', b)]
    if len(valid_blocks) <= chunk_size:
        return [clean_text]

    chunks = []
    for i in range(0, len(valid_blocks), chunk_size):
        chunk = "\n\n".join(valid_blocks[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def _call_gemini_or_groq_srt(input_text, sys_instruction, model_name="gemini-3.5-flash-lite"):
    """
    Exécute l'appel API LLM (Gemini avec fallback Groq) pour un segment de texte SRT.
    """
    if not input_text or not input_text.strip():
        return ""

    from soni_translate.translate_segments import get_gemini_api_keys
    keys = get_gemini_api_keys()
    if not keys:
        raise ValueError("Clé GOOGLE_API_KEY manquante. Renseigne ta clé Gemini pour utiliser l'adaptation automatique.")

    from google import genai
    from google.genai import types

    candidates = [model_name, "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
    last_err = None

    for key in keys:
        client = genai.Client(api_key=key)
        for m in candidates:
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=input_text.strip(),
                    config=types.GenerateContentConfig(
                        system_instruction=sys_instruction,
                        temperature=0.3,
                        max_output_tokens=8192
                    )
                )
                raw_out = response.text.strip()
                clean_out = re.sub(r'^```(?:srt)?\s*', '', raw_out)
                clean_out = re.sub(r'\s*```$', '', clean_out).strip()
                first_num_match = re.search(r'(?:\A|\n)(\d+\s*\n\s*\d{1,2}:\d{2}:\d{2})', clean_out)
                if first_num_match:
                    clean_out = clean_out[first_num_match.start(1):].strip()
                return clean_out
            except Exception as e:
                last_err = e
                logger.warning(f"FastDub LLM tentative '{m}' a échoué ({e}). Essai suivant...")
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
                    {"role": "user", "content": input_text.strip()}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=8000
            )
            raw_groq = chat.choices[0].message.content.strip()
            clean_groq = re.sub(r'^```(?:srt)?\s*', '', raw_groq)
            clean_groq = re.sub(r'\s*```$', '', clean_groq).strip()
            first_num_match = re.search(r'(?:\A|\n)(\d+\s*\n\s*\d{1,2}:\d{2}:\d{2})', clean_groq)
            if first_num_match:
                clean_groq = clean_groq[first_num_match.start(1):].strip()
            return clean_groq
        except Exception as ge:
            logger.error(f"Secours Groq FastDub échoué : {ge}")

    if last_err:
        raise last_err
    return ""


def adapt_script_gemini(srt_text, custom_prompt=None, model_name="gemini-3.5-flash-lite"):
    """
    Envoie le script SRT complet à Gemini (par lots si > 70 répliques) pour produire le script de doublage adapté.
    Garantit une fidélité temporelle absolue et aucune troncature de jetons.
    """
    if not srt_text or not srt_text.strip():
        return ""

    sys_instruction = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else FAST_DUB_MASTER_PROMPT

    chunks = split_srt_into_cue_chunks(srt_text, chunk_size=70)
    adapted_parts = []

    for idx, chk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(f"FastDub Gemini : Adaptation du lot {idx + 1}/{len(chunks)} ({len(chk)} car.)...")
        else:
            logger.info(f"FastDub Gemini : Adaptation du script ({len(chk)} car.)...")
        part = _call_gemini_or_groq_srt(chk, sys_instruction, model_name=model_name)
        adapted_parts.append(part)

    full_adapted = "\n\n".join(p for p in adapted_parts if p.strip()).strip()
    # Rétablissement strict des horodatages originaux pour éliminer toute hallucination
    full_adapted = enforce_srt_timestamps(full_adapted, srt_text)
    return full_adapted


def translate_dubbing_script(script_text, target_lang="fr", model_name="gemini-3.5-flash-lite"):
    """
    Traduit et adapte localement le script de doublage (par exemple du Français vers l'Espagnol, l'Anglais, etc.)
    en préservant strictement les horodatages, les locuteurs (Speaker 1/2) et les balises d'émotions.
    Si la langue cible est le français, renvoie directement le script sans aucun appel API.
    """
    if not script_text or not script_text.strip():
        return ""

    clean_code = "fr"
    clean_name = "Français"
    lang_mapping = {
        "fr": "Français",
        "es": "Espagnol",
        "en": "Anglais",
        "pt": "Portugais",
        "it": "Italien",
        "de": "Allemand",
        "ru": "Russe",
        "ar": "Arabe",
        "ja": "Japonais",
        "zh": "Chinois",
        "ko": "Coréen"
    }
    raw_lower = str(target_lang).lower()
    for code, name in lang_mapping.items():
        if code in raw_lower or name.lower() in raw_lower:
            clean_code = code
            clean_name = name
            break

    # Si c'est déjà du français, aucune traduction nécessaire : copie directe instantanée !
    if clean_code == "fr":
        logger.info("FastDub : Langue cible française demandée, réutilisation directe du script français édité.")
        return script_text.strip()

    logger.info(f"FastDub : Traduction et localisation du script vers '{clean_name}' ({clean_code})...")

    translate_instruction = f"""**RÔLE :**
Tu es un Expert Adaptateur et Directeur Artistique de Doublage Rythmique Multimodal.
Ta mission : traduire et adapter localement ce script de doublage français pour des vidéos de type "Gold Digger Prank" / caméras cachées vers la langue cible : **{clean_name} ({clean_code})**.

**RÈGLES D'ADAPTATION STRICTES :**
1. **CONSERVER EXACTEMENT :**
   - La numérotation de chaque bloc.
   - Les horodatages exacts (HH:MM:SS,mmm --> HH:MM:SS,mmm).
   - Les étiquettes des locuteurs ("Speaker 1:", "Speaker 2:").
   - Les balises d'émotions entre crochets au début de chaque réplique (ex: [charmeur], [froide], [amuse], [whisper], [curieux], [excited]...).
2. **STYLE ET LOCALISATION NATURELLE :**
   - Traduis dans un registre **oral, vivant, jeune et contemporain (street / slang / argot naturel)** propre à la culture de la langue cible ({clean_name}).
   - Reste percutant et condensé : le texte traduit doit respecter la durée exacte de la réplique (ne fais pas de phrases trop longues pour le TTS).
3. **FORMAT STRICT DE SORTIE :**
   Numéro
   HH:MM:SS,mmm --> HH:MM:SS,mmm
   Speaker X: [émotion] Texte traduit et adapté

(Commence directement par le premier numéro de bloc sans aucun commentaire, explication ou balises markdown ```)."""

    chunks = split_srt_into_cue_chunks(script_text, chunk_size=70)
    translated_parts = []

    for idx, chk in enumerate(chunks):
        if len(chunks) > 1:
            logger.info(f"FastDub : Traduction vers {clean_name} du lot {idx + 1}/{len(chunks)}...")
        part = _call_gemini_or_groq_srt(chk, translate_instruction, model_name=model_name)
        translated_parts.append(part)

    full_translated = "\n\n".join(p for p in translated_parts if p.strip()).strip()
    full_translated = enforce_srt_timestamps(full_translated, script_text)
    return full_translated



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
    clean_text = script_text.replace('\r\n', '\n').strip()
    # Normalisation : force un double saut de ligne avant chaque numéro suivi de timestamp
    normalized = re.sub(r'(\n|^)\s*(\d+)\s*\n\s*(\d{1,2}:\d{2}:\d{2})', r'\n\n\2\n\3', clean_text).strip()
    blocks = re.split(r'\n\s*\n', normalized)
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
        # Filtre absolu : élimine toute ligne de timestamp ou de numéro résiduel
        raw_text_lines = [l.strip() for l in lines[time_line_idx + 1:] if l.strip()]
        text_lines = [tl for tl in raw_text_lines if not (time_pattern.search(tl) or tl.isdigit())]
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
            emotion_tag = ""  # Pas d'émotion forcée par défaut
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
                raw_emo = emo_match.group(1).strip()
                emotion_tag = normalize_emotion_for_fish(raw_emo)
                speech_text = emo_match.group(2).strip()

            # Nettoyage ultime du texte pour le TTS
            speech_text = re.sub(r'^(?:Speaker\s*\d+|Homme|Femme)\s*:\s*', '', speech_text, flags=re.IGNORECASE).strip()
            # Supprime tout timestamp égaré ou numéro de cue terminal
            speech_text = time_pattern.sub('', speech_text).strip()
            speech_text = re.sub(r'\b\d+\s*$', '', speech_text).strip()

            full_tts = f"{emotion_tag} {speech_text}".strip() if emotion_tag else speech_text

            parsed_dialogues.append({
                "start": round(cue_start, 3),
                "end": round(cue_end, 3),
                "duration": round(cue_end - cue_start, 3),
                "speaker": speaker_id,
                "emotion": emotion_tag,
                "text": speech_text,
                "full_tts_text": full_tts
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
    target_lang="fr",
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

    clean_target_lang = "fr"
    if target_lang:
        raw_t = str(target_lang).lower()
        for c in ["fr", "es", "en", "pt", "it", "de", "ru", "ar", "ja", "zh", "ko"]:
            if c in raw_t:
                clean_target_lang = c
                break

    logger.info(f"FastDub Render : {len(dialogues)} répliques à synthétiser en '{clean_target_lang}' ({tts_engine})...")
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
            emo_info = f" ({diag['emotion']})" if diag.get("emotion") else ""
            progress(0.1 + (0.75 * (idx / max(total_diag, 1))), desc=f"Synthèse réplique {idx+1}/{total_diag}{emo_info}...")

        voice_chosen = speaker2_voice if diag["speaker"] == 2 else speaker1_voice
        tts_text = diag["full_tts_text"]
        out_chunk_path = os.path.join(chunks_dir, f"chunk_{idx:04d}.wav")

        try:
            if tts_engine == "fish_audio" and fish_provider:
                fish_provider.generate(
                    text=tts_text,
                    voice=voice_chosen,
                    target_lang=clean_target_lang,
                    output_file=out_chunk_path
                )
            else:
                # Secours Edge-TTS multilingue
                import edge_tts
                import asyncio
                clean_raw_text = diag["text"]
                edge_voices_by_lang = {
                    "fr": ("fr-FR-HenriNeural", "fr-FR-DeniseNeural"),
                    "es": ("es-ES-AlvaroNeural", "es-ES-ElviraNeural"),
                    "en": ("en-US-GuyNeural", "en-US-JennyNeural"),
                    "pt": ("pt-BR-AntonioNeural", "pt-BR-FranciscaNeural"),
                    "it": ("it-IT-DiegoNeural", "it-IT-ElsaNeural"),
                    "de": ("de-DE-ConradNeural", "de-DE-KatjaNeural"),
                    "ru": ("ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"),
                    "ar": ("ar-SA-HamedNeural", "ar-SA-ZariyahNeural"),
                    "ja": ("ja-JP-KeitaNeural", "ja-JP-NanamiNeural"),
                }
                m_v, f_v = edge_voices_by_lang.get(clean_target_lang, ("fr-FR-HenriNeural", "fr-FR-DeniseNeural"))
                edge_v = f_v if diag["speaker"] == 2 else m_v
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
