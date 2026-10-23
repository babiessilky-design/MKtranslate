from tqdm import tqdm
from deep_translator import GoogleTranslator
from itertools import chain
import copy
from .language_configuration import fix_code_language, INVERTED_LANGUAGES
from .logging_setup import logger
import re
import json
import time
import os

# --- IMPORTS ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    pass
try:
    from openai import OpenAI
except ImportError:
    pass
try:
    from huggingface_hub import InferenceClient
except ImportError:
    pass
try:
    import httpx
except ImportError:
    pass

TRANSLATION_PROCESS_OPTIONS = [
    "gemini_flash",
    "gemini_pro",
    "groq_llama3",
    "hf_zephyr_7b_beta",
    "google_translator_batch",
    "google_translator",
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "disable_translation",
]

DOCS_TRANSLATION_PROCESS_OPTIONS = [
    "gemini_flash",
    "gemini_pro",
    "groq_llama3",
    "hf_zephyr_7b_beta",
    "google_translator",
    "disable_translation",
]

# ==============================================================================
# PROMPT CONTEXTUEL DYNAMIQUE (MULTILINGUE)
# ==============================================================================
def get_system_prompt(target_lang):
    """Génère un prompt dynamique adapté à la langue cible choisie par l'utilisateur."""
    is_fish_emotion = os.getenv("FISH_AUTO_EMOTION", "False").lower() in ["true", "1", "yes"]

    if is_fish_emotion:
        return f"""Tu es un traducteur expert en doublage vidéo dynamique pour YouTube et le cinéma.

RÈGLES OBLIGATOIRES :
1. Traduis chaque ligne fidèlement VERS LA LANGUE SUIVANTE : {target_lang}.
2. Le style doit être naturel, fluide, dynamique et parlé (adapté à un public jeune).
3. Conserve impérativement le sens exact, les nuances affectives et le tutoiement/langage familier si applicable.
4. Traduis TOUTES les phrases sans exception.
5. DIRECTIVE OBLIGATOIRE ÉMOTIONS ET JEU D'ACTEUR (FISH AUDIO) :
Pour CHAQUE réplique expressive, insère impérativement au tout début la balise d'émotion Fish Audio correspondante parmi :
- [excited] pour toute phrase dynamique, enthousiaste, joyeuse ou avec un point d'exclamation (ex: [excited] Regarde ce qu'on a trouvé !)
- [laughing] pour tout rire, réplique amusante, blague ou amusement (ex: [laughing] C'est trop drôle haha !)
- [whisper] pour tout chuchotement, confidence, secret ou voix basse (ex: [whisper] Chut, fais pas de bruit...)
- [angry] pour toute réplique énervée, cri, dispute, ordre sec ou tension (ex: [angry] Arrête tout de suite !)
- [surprised] pour toute surprise, question choquée ou stupéfaction (ex: [surprised] Quoi ?! Tu es sérieux ?)
- [sad] pour toute tristesse, regret, désolation ou compassion (ex: [sad] C'est vraiment triste...)
- [sigh] pour un soupir, un soulagement ou de la fatigue (ex: [sigh] Ouf, enfin réussi...)
- [calm] pour une explication calme, posée ou rassurante
Laisse sans balise uniquement les énoncés factuels sans aucune charge affective.
6. Renvoie STRICTEMENT un tableau JSON de chaînes de caractères.
Exemple :
["[excited] C'est incroyable, on a enfin réussi !", "[whisper] Fais attention, ils nous cherchent...", "[surprised] Attends, tu plaisantes ?"]"""

    return f"""Tu es un traducteur expert en doublage vidéo pour YouTube.

RÈGLES OBLIGATOIRES :
1. Traduis chaque ligne fidèlement VERS LA LANGUE SUIVANTE : {target_lang}.
2. Le style doit être naturel, fluide et parlé (adapté à un public jeune).
3. Le texte en {target_lang} doit être court et percutant pour coller au rythme de la vidéo.
4. Conserve impérativement le sens exact, les émotions et le niveau de langage (tutoiement/familier si applicable).
5. Traduis TOUTES les phrases sans exception.
6. Renvoie la traduction STRICTEMENT sous forme d'un tableau JSON de chaînes de caractères.
Exemple :
["Traduction 1", "Traduction 2", "Traduction 3"]"""

# ==============================================================================
# FILTRE ANTI-PAGE D'ERREUR SERVEUR
# ==============================================================================
def is_corrupted_translation(text):
    """Détecte si un traducteur a renvoyé une page d'erreur HTTP au lieu d'une traduction."""
    if not text or not isinstance(text, str):
        return True
    bad_patterns = [
        "error 500", "server error", "that’s an error", "that's an error",
        "please try again later", "that's all we know", "html", "<body", "500."
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in bad_patterns)

# ==============================================================================
# PARSEUR UNIVERSEL DE RÉPONSES LLM
# ==============================================================================
def parse_llm_response(raw_text, expected_len):
    """Extrait proprement les traductions qu'elles soient en liste JSON, dictionnaire ou texte numéroté."""
    if not raw_text or not isinstance(raw_text, str):
        return None
    
    clean = re.sub(r'^```(?:json)?\s*', '', raw_text.strip())
    clean = re.sub(r'\s*```$', '', clean).strip()

    # 1. Tentative de décodage JSON
    try:
        data = json.loads(clean)
        if isinstance(data, list):
            res = [str(x).strip() for x in data if str(x).strip()]
            if res:
                return res
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    res = [str(x).strip() for x in v if str(x).strip()]
                    if res:
                        return res
            def get_num(key):
                nums = re.findall(r'\d+', str(key))
                return int(nums[0]) if nums else 9999
            sorted_keys = sorted(data.keys(), key=get_num)
            res = [str(data[k]).strip() for k in sorted_keys if str(data[k]).strip()]
            if res:
                return res
    except Exception:
        pass

    # 2. Fallback de parsing ligne par ligne numérotée
    lines = []
    for line in clean.split('\n'):
        l_str = line.strip()
        if not l_str or l_str in ['[', ']', '{', '}']:
            continue
        clean_line = re.sub(r'^\s*[\d]+[\.\)\-\:\s]+\s*', '', l_str).strip()
        clean_line = re.sub(r'^["\']|["\',]$', '', clean_line).strip()
        if clean_line and not is_corrupted_translation(clean_line):
            lines.append(clean_line)

    return lines if lines else None

def _single_translate(text, target_lang):
    """Traduction de secours sécurisée d'une phrase unique avec plusieurs essais."""
    orig_text = str(text).strip()
    if not orig_text:
        return ""
    target_clean = fix_code_language(target_lang)
    for _ in range(3):
        try:
            tr = GoogleTranslator(source='auto', target=target_clean)
            res = tr.translate(orig_text)
            if res and not is_corrupted_translation(res) and res.strip() != orig_text:
                return res.strip()
        except Exception:
            time.sleep(1.0)
    return orig_text

# ==============================================================================
# BATCHING CONTEXTUEL SÉCURISÉ AVEC GESTION DU RATE-LIMIT
# ==============================================================================
def _batch_with_context(segments, batch_size, translate_func, desc, target_lang):
    translated = copy.deepcopy(segments)
    progress = tqdm(total=len(segments), desc=desc)
    context = []
    
    # On génère le prompt dynamique pour la langue cible
    sys_prompt = get_system_prompt(target_lang)

    start = 0
    while start < len(segments):
        end = min(start + batch_size, len(segments))
        batch = translated[start:end]
        batch_len = len(batch)

        previous = "\n".join([f"Précédent {i+1}: {c}" for i, c in enumerate(context[-3:])])
        lines_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])

        full_prompt = (
            f"{sys_prompt}\n\n"
            f"Contexte précédent :\n{previous}\n\n"
            f"Lignes à traduire ({batch_len} phrases) :\n{lines_text}\n\n"
            f"Renvoie le tableau JSON des {batch_len} traductions :"
        )

        translated_lines = None
        for attempt in range(3):
            try:
                translated_lines = translate_func(full_prompt, batch_len)
                if translated_lines and len(translated_lines) >= batch_len:
                    break
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg or "503" in err_msg or "unavailable" in err_msg:
                    sleep_time = 15.0 * (attempt + 1)
                    logger.warning(f"Quota/Saturation API au batch {start}-{end}. Pause de {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Tentative {attempt+1} échouée pour le batch {start}-{end}: {e}")
                    time.sleep(1.5 * (attempt + 1))

        # Application intelligente des traductions
        if translated_lines and len(translated_lines) > 0:
            for j in range(batch_len):
                if j < len(translated_lines):
                    clean = re.sub(r'^\s*[\d]+[\.\)\-\:\s]+', '', str(translated_lines[j])).strip()
                    clean = re.sub(r'^["\']|["\']$', '', clean).strip()
                    if clean and not is_corrupted_translation(clean):
                        translated[start + j]["text"] = clean
                    else:
                        translated[start + j]["text"] = _single_translate(batch[j]["text"], target_lang)
                else:
                    translated[start + j]["text"] = _single_translate(batch[j]["text"], target_lang)
            context.extend(translated_lines[:batch_len])
            progress.update(batch_len)
            start += batch_size
            time.sleep(1.0)
        else:
            logger.warning(f"Paquet {start}-{end} basculé sur secours individuel...")
            for j, seg in enumerate(batch):
                translated[start + j]["text"] = _single_translate(seg["text"], target_lang)
                time.sleep(0.3)
            progress.update(batch_len)
            start += batch_size
            time.sleep(1.0)

    try: progress.close()
    except: pass
    return translated

# ==============================================================================
# CONFIGURATION SÉCURITÉ GEMINI (DÉBLOCAGE DE LA CENSURE DE DIALOGUES)
# ==============================================================================
def get_gemini_safety_settings():
    """Désactive le blocage des insultes de fiction/dialogues pour ne jamais censurer de phrases."""
    try:
        return [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
    except Exception:
        return None

# ==============================================================================
# GEMINI (CASCADE DYNAMIQUE & MÉMORISATION SANS CENSURE)
# ==============================================================================
_ACTIVE_GEMINI_INDEX = 0

def gemini_translate(segments, target, source=None, mode="flash"):
    global _ACTIVE_GEMINI_INDEX
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key") or ""
    if not api_key:
        logger.error("❌ GEMINI: Clé GOOGLE_API_KEY manquante dans l'environnement !")
        return translate_iterative(segments, target, source)

    client = genai.Client(api_key=api_key)

    # Modèles actifs de secours
    if mode == "pro":
        candidates = ["gemini-3.1-pro-preview", "gemini-3.1-pro"]
    else:
        candidates = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.7-flash"]

    # Prompt système dynamique par langue
    sys_prompt = get_system_prompt(target)

    config_kwargs = {
        "temperature": 0.2,
        "max_output_tokens": 3000,
        "response_mime_type": "application/json",
        "system_instruction": sys_prompt
    }
    
    safety = get_gemini_safety_settings()
    if safety:
        config_kwargs["safety_settings"] = safety

    config = types.GenerateContentConfig(**config_kwargs)

    def call_gemini(full_prompt, batch_len):
        global _ACTIVE_GEMINI_INDEX
        last_err = None

        num_models = len(candidates)
        for offset in range(num_models):
            idx = (_ACTIVE_GEMINI_INDEX + offset) % num_models
            m_id = candidates[idx]
            try:
                response = client.models.generate_content(model=m_id, contents=full_prompt, config=config)
                res = parse_llm_response(response.text, batch_len)
                if res and len(res) > 0:
                    _ACTIVE_GEMINI_INDEX = idx
                    return res
            except Exception as e:
                last_err = e
                logger.warning(f"Modèle '{m_id}' saturé/indisponible ({e}). Bascule immédiate sur le modèle suivant...")
                continue

        if last_err:
            raise last_err
        return None

    return _batch_with_context(segments, 20, call_gemini, f"Translating (Gemini {mode.upper()} Uncensored)", target)

# ==============================================================================
# GROQ (LLaMA-3.3 70B)
# ==============================================================================
def groq_translate(segments, target, source=None):
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key") or ""
    if not api_key:
        logger.error("❌ GROQ: Clé GROQ_API_KEY manquante !")
        return translate_iterative(segments, target, source)

    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, http_client=httpx.Client(timeout=60))
    sys_prompt = get_system_prompt(target)

    def call_groq(full_prompt, batch_len):
        chat = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": full_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        return parse_llm_response(chat.choices[0].message.content, batch_len)

    return _batch_with_context(segments, 25, call_groq, "Translating (Groq LLaMA-3.3 Batch)", target)

# ==============================================================================
# ZEPHYR (Hugging Face Inference)
# ==============================================================================
def hf_zephyr_translate(segments, target, source=None, batch_size=15):
    hf_token = os.getenv("HF_TOKEN") or os.getenv("YOUR_HF_TOKEN") or ""
    if not hf_token or not hf_token.startswith("hf_"):
        logger.error("❌ ZEPHYR: HF_TOKEN manquant !")
        return translate_iterative(segments, target, source)

    client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=hf_token)
    sys_prompt = get_system_prompt(target)

    def call_zephyr(full_prompt, batch_len):
        prompt = f"<|system|>\n{sys_prompt}</s>\n<|user|>\n{full_prompt}</s>\n<|assistant|>"
        response = client.text_generation(prompt, max_new_tokens=1500, temperature=0.3, return_full_text=False)
        return parse_llm_response(response, batch_len)

    return _batch_with_context(segments, batch_size, call_zephyr, "Translating (Zephyr Batch)", target)

# ==============================================================================
# GOOGLE TRANSLATE FALLBACK
# ==============================================================================
def translate_iterative(segments, target, source=None):
    segments_ = copy.deepcopy(segments)
    if not source: source = "auto"
    target_clean = fix_code_language(target)
    translator = GoogleTranslator(source=source, target=target_clean)
    for line in tqdm(range(len(segments_)), desc="Translating (Iterative)"):
        text = segments_[line]["text"]
        try:
            res = translator.translate(text.strip())
            if res and not is_corrupted_translation(res):
                segments_[line]["text"] = res
        except Exception as e:
            logger.error(f"Error google iterative: {e}")
    return segments_

def translate_batch(segments, target, chunk_size=2000, source=None):
    segments_copy = copy.deepcopy(segments)
    if not source: source = "auto"
    text_lines = [seg["text"].strip() for seg in segments_copy]
    text_merge = []
    actual_chunk = ""
    global_text_list = []
    actual_text_list = []
    for one_line in text_lines:
        one_line = " " if not one_line else one_line
        if (len(actual_chunk) + len(one_line)) <= chunk_size:
            if actual_chunk: actual_chunk += " ||||| "
            actual_chunk += one_line
            actual_text_list.append(one_line)
        else:
            text_merge.append(actual_chunk)
            actual_chunk = one_line
            global_text_list.append(actual_text_list)
            actual_text_list = [one_line]
    if actual_chunk:
        text_merge.append(actual_chunk)
        global_text_list.append(actual_text_list)

    progress_bar = tqdm(total=len(segments), desc="Translating (Google Batch)")
    translator = GoogleTranslator(source=source, target=target)
    split_list = []
   
    try:
        for text, text_iterable in zip(text_merge, global_text_list):
            translated_line = translator.translate(text.strip())
            if is_corrupted_translation(translated_line):
                raise ValueError("Page d'erreur détectée dans Google Translate")
            split_text = translated_line.split("|||||")
            if len(split_text) == len(text_iterable):
                try: progress_bar.update(len(split_text))
                except: pass
            else:
                split_text = []
                for txt_iter in text_iterable:
                    translated_txt = translator.translate(txt_iter.strip())
                    if is_corrupted_translation(translated_txt):
                        translated_txt = txt_iter
                    split_text.append(translated_txt)
                    try: progress_bar.update(1)
                    except: pass
            split_list.append(split_text)
        try: progress_bar.close()
        except: pass
    except Exception:
        try: progress_bar.close()
        except: pass
        return translate_iterative(segments, target, source)
        
    translated_lines = list(chain.from_iterable(split_list))
    return verify_translate(segments, segments_copy, translated_lines, target, source)

def verify_translate(segments, segments_copy, translated_lines, target, source):
    if len(segments) == len(translated_lines):
        for line in range(len(segments_copy)):
            clean_text = translated_lines[line].replace("\t", "").replace("\n", "").strip()
            if not is_corrupted_translation(clean_text):
                segments_copy[line]["text"] = clean_text
        return segments_copy
    else:
        return translate_iterative(segments, target, source)

def gpt_sequential(segments, model, target, source=None):
    return translate_iterative(segments, target, source)

def gpt_batch(segments, model, target, token_batch_limit=900, source=None):
    return translate_iterative(segments, target, source)

# ==============================================================================
def apply_rule_based_emotions(segments):
    """
    Rattrapage et enrichissement automatique d'émotions pour Fish Audio :
    Analyse sémantique et ponctuation pour garantir que chaque réplique expressive
    possède une balise d'émotion reconnue par Fish Audio.
    """
    import re
    valid_tags_pattern = re.compile(
        r'\[\s*(?:whisper|laughing|angry|excited|sad|surprised|sigh|crying|pause|calm)\s*\]', 
        re.IGNORECASE
    )

    laugh_patterns = re.compile(r'\b(haha+|héhé+|hihi+|mdr|lol|ptdr|ahah+|marrant|drôle|rigole|blague|fou rire|hilarious|funny|joke)\b', re.IGNORECASE)
    whisper_patterns = re.compile(r'\b(chut|psst|silence|tout bas|en secret|discrètement|baisse d\'un ton|pas un mot|whisper|hush|quietly|shh)\b', re.IGNORECASE)
    angry_patterns = re.compile(r'\b(merde|putain|bordel|fais chier|dégage|ferme[- ]la|tais[- ]toi|idiot|crétin|imbécile|enfoiré|stop|arrête|fous le camp|colère|hurle|damn|fuck|shit|shut up)\b', re.IGNORECASE)
    surprised_patterns = re.compile(r'(\?!|!\?|\b(quoi\s*\?|comment\s*\?|attends quoi|impossible|sérieux\s*\?|vraiment\s*\?|c\'est pas vrai|non\s*\?|incroyable|ouah|wow|omg|mon dieu|what\s*\?|no way|seriously)\b)', re.IGNORECASE)
    sad_patterns = re.compile(r'\b(triste|malheureusement|hélas|désolé|pardon|pauvre|pleure|sanglot|larme|douleur|douloureux|horrible|adieu|sad|sorry|crying|painful|grief)\b', re.IGNORECASE)
    sigh_patterns = re.compile(r'\b(ouf|pff+|enfin|soulagement|fatigué|marre|j\'en peux plus|ras le bol|sigh|finally|exhausted)\b', re.IGNORECASE)
    excited_patterns = re.compile(r'\b(génial|super|fantastique|magnifique|trop bien|let\'s go|on y va|regarde|viens voir|victoire|bravo|dingue|extraordinaire|succès|champion|yes|awesome|amazing|hurray|look at)\b', re.IGNORECASE)

    tagged_count = 0
    for idx, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        # Si une balise valide est déjà présente (mise par le LLM), on la conserve
        if valid_tags_pattern.search(text):
            tagged_count += 1
            logger.info(f"[Auto-Emotion] Segment {idx} (LLM) : '{text}'")
            continue

        chosen_tag = None
        # Priorité 1 : Rires
        if laugh_patterns.search(text):
            chosen_tag = "[laughing]"
        # Priorité 2 : Chuchotements
        elif whisper_patterns.search(text):
            chosen_tag = "[whisper]"
        # Priorité 3 : Colère / Tension
        elif angry_patterns.search(text) or (text.endswith("!!") and len(text) > 3):
            chosen_tag = "[angry]"
        # Priorité 4 : Surprise / Choc
        elif surprised_patterns.search(text):
            chosen_tag = "[surprised]"
        # Priorité 5 : Tristesse
        elif sad_patterns.search(text):
            chosen_tag = "[sad]"
        # Priorité 6 : Soupir / Soulagement
        elif sigh_patterns.search(text):
            chosen_tag = "[sigh]"
        # Priorité 7 : Enthousiasme / Joie (ou toute phrase se terminant par '!')
        elif excited_patterns.search(text) or text.endswith("!"):
            chosen_tag = "[excited]"
        
        if chosen_tag:
            seg["text"] = f"{chosen_tag} {text}"
            tagged_count += 1
            logger.info(f"[Auto-Emotion] Segment {idx} enrichi avec {chosen_tag} : '{seg['text']}'")

    logger.info(f"[Auto-Emotion] Total segments enrichis avec succès : {tagged_count}/{len(segments)}")
    return segments


# ==============================================================================
# DISPATCHER PRINCIPAL
# ==============================================================================
def translate_text(
    segments,
    target,
    translation_process="gemini_flash",
    chunk_size=4500,
    source=None,
    token_batch_limit=1000,
):
    target_clean = fix_code_language(target)
    source_clean = fix_code_language(source) if source else "auto"

    match translation_process:
        case "gemini_flash":
            translated = gemini_translate(segments, target, source, mode="flash")
        case "gemini_pro":
            translated = gemini_translate(segments, target, source, mode="pro")
        case "groq_llama3":
            translated = groq_translate(segments, target, source)
        case "hf_zephyr_7b_beta":
            translated = hf_zephyr_translate(segments, target, source)
        case "google_translator_batch":
            translated = translate_batch(segments, target_clean, chunk_size, source_clean)
        case "google_translator":
            translated = translate_iterative(segments, target_clean, source_clean)
        case model if "gpt" in model:
            translated = translate_iterative(segments, target_clean, source_clean)
        case "disable_translation":
            translated = segments
        case _:
            translated = translate_iterative(segments, target_clean, source_clean)

    # Enrichissement automatique avec les émotions Fish Audio si l'option est activée
    is_fish_emotion = os.getenv("FISH_AUTO_EMOTION", "False").lower() in ["true", "1", "yes"]
    if is_fish_emotion and translated:
        translated = apply_rule_based_emotions(translated)
        logger.info("Auto-Emotion Fish Audio: Balises d'émotions appliquées sur les segments traduits.")

    return translated
