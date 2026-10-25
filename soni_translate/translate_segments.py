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

    clean_code = fix_code_language(target_lang)
    full_name = INVERTED_LANGUAGES.get(clean_code, target_lang)
    lang_instruction = f"{full_name} (code ISO: '{clean_code}')"

    if is_fish_emotion:
        return f"""Tu es un traducteur expert en doublage vidéo dynamique pour YouTube et le cinéma.

RÈGLES OBLIGATOIRES :
1. Traduis chaque réplique fidèlement VERS LA LANGUE SUIVANTE : {lang_instruction}.
2. Le style doit être naturel, fluide, dynamique et parlé (adapté à un public jeune).
3. Conserve impérativement le sens exact, les nuances affectives et le tutoiement/langage familier si applicable.
4. Traduis TOUTES les phrases sans exception, sans aucun commentaire ni note.
5. DIRECTIVE OBLIGATOIRE ÉMOTIONS ET JEU D'ACTEUR (FISH AUDIO) :
Pour CHAQUE réplique expressive, insère impérativement au tout début la balise d'émotion Fish Audio correspondante parmi :
- [excited], [laughing], [whisper], [angry], [surprised], [sad], [sigh], [calm]
Laisse sans balise uniquement les énoncés factuels sans aucune charge affective.
6. Renvoie STRICTEMENT un objet JSON avec les numéros de ligne sous forme de clés ('1', '2', ...) associées à leur traduction.
Exemple pour 3 phrases :
{{
  "1": "[excited] C'est incroyable, on a enfin réussi !",
  "2": "[whisper] Fais attention, ils nous cherchent...",
  "3": "[surprised] Attends, tu plaisantes ?"
}}"""

    return f"""Tu es un traducteur expert en doublage vidéo pour YouTube.

RÈGLES OBLIGATOIRES :
1. Traduis chaque réplique fidèlement VERS LA LANGUE SUIVANTE : {lang_instruction}.
2. Le style doit être naturel, fluide et parlé (adapté à un public jeune).
3. Le texte en {lang_instruction} doit être court et percutant pour coller au rythme de la vidéo.
4. Conserve impérativement le sens exact, les émotions et le niveau de langage (tutoiement/familier si applicable).
5. Traduis TOUTES les phrases sans exception, sans aucun commentaire ou texte hors du JSON.
6. Renvoie STRICTEMENT un objet JSON avec les numéros de ligne sous forme de clés ('1', '2', ...) associées à leur traduction.
Exemple pour 3 phrases :
{{
  "1": "Traduction phrase 1",
  "2": "Traduction phrase 2",
  "3": "Traduction phrase 3"
}}"""

# ==============================================================================
# FILTRE ANTI-PAGE D'ERREUR SERVEUR
# ==============================================================================
def is_corrupted_translation(text):
    """Détecte si un traducteur a renvoyé une page d'erreur HTTP au lieu d'une traduction."""
    if not text or not isinstance(text, str):
        return True
    bad_patterns = [
        "error 500", "server error", "that’s an error", "that's an error",
        "please try again later", "that's all we know", "html", "<body", "500.",
        "here is the translation", "here are the translations", "voici la traduction",
        "voici les traductions", "certainly!", "sure, here", "note:"
    ]
    text_lower = text.lower().strip()
    return any(p in text_lower for p in bad_patterns)

# ==============================================================================
# PARSEUR UNIVERSEL DE RÉPONSES LLM
# ==============================================================================
def parse_llm_response(raw_text, expected_len=None):
    """Extrait proprement les traductions qu'elles soient en liste JSON, dictionnaire ou texte numéroté."""
    if not raw_text or not isinstance(raw_text, str):
        return None
    
    clean = raw_text.strip()

    # 1. Tentative d'extraction d'un bloc JSON explicite ```json ... ``` ou [ ... ] / { ... }
    json_candidate = None
    json_block = re.search(r'```(?:json)?\s*([\[\{].*?[\]\}])\s*```', clean, re.DOTALL)
    if json_block:
        json_candidate = json_block.group(1).strip()
    else:
        array_match = re.search(r'(\[[\s\S]*\])', clean)
        if array_match:
            json_candidate = array_match.group(1).strip()
        else:
            dict_match = re.search(r'(\{[\s\S]*\})', clean)
            if dict_match:
                json_candidate = dict_match.group(1).strip()

    if not json_candidate:
        json_candidate = re.sub(r'^```(?:json)?\s*', '', clean)
        json_candidate = re.sub(r'\s*```$', '', json_candidate).strip()

    try:
        data = json.loads(json_candidate)
        if isinstance(data, list):
            res = []
            for x in data:
                if isinstance(x, dict):
                    val = x.get("text") or x.get("translation") or x.get("translated") or list(x.values())[-1]
                    res.append(str(val).strip())
                else:
                    res.append(str(x).strip())
            res = [x for x in res if x and not is_corrupted_translation(x)]
            if res and (expected_len is None or len(res) == expected_len):
                return res
            elif res and len(res) >= (expected_len or 1) * 0.7:
                return res
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    res = [str(x).strip() for x in v if str(x).strip() and not is_corrupted_translation(str(x))]
                    if res:
                        return res
            def get_num(key):
                nums = re.findall(r'\d+', str(key))
                return int(nums[0]) if nums else 9999
            sorted_keys = sorted(data.keys(), key=get_num)
            res = []
            for k in sorted_keys:
                val = data[k]
                if isinstance(val, dict):
                    val = val.get("text") or val.get("translation") or list(val.values())[-1]
                res.append(str(val).strip())
            res = [x for x in res if x and not is_corrupted_translation(x)]
            if res:
                return res
    except Exception:
        pass

    # 2. Fallback de parsing ligne par ligne
    lines = []
    line_prefix_regex = re.compile(
        r'^\s*[\*\-•]?\s*(?:\*\*)?(?:Line|Ligne|Phrase|Sentence)?\s*\[?\d+\]?[\.\)\:\-\s]*(?:\*\*)?[\:\-\s]*',
        re.IGNORECASE
    )
    for line in clean.split('\n'):
        l_str = line.strip()
        if not l_str or l_str in ['[', ']', '{', '}', '```', '```json']:
            continue
        l_lower = l_str.lower()
        if l_lower.startswith(('here is', 'here are', 'voici', 'note:', 'certainly', 'sure')):
            continue
        clean_line = line_prefix_regex.sub('', l_str).strip()
        clean_line = re.sub(r'^["\']|["\',]$', '', clean_line).strip()
        if clean_line and not is_corrupted_translation(clean_line):
            lines.append(clean_line)

    return lines if lines else None

def _single_translate(text, target_lang):
    """Traduction de secours sécurisée d'une phrase unique avec plusieurs essais et temporisation anti-ban."""
    orig_text = str(text).strip()
    if not orig_text:
        return ""
    target_clean = fix_code_language(target_lang)
    for attempt in range(3):
        try:
            tr = GoogleTranslator(source='auto', target=target_clean)
            res = tr.translate(orig_text)
            if res and not is_corrupted_translation(res):
                return res.strip()
        except Exception as e:
            logger.warning(f"[_single_translate] Erreur tentative {attempt+1} pour '{orig_text[:25]}' vers '{target_clean}': {e}")
            time.sleep(1.5 * (attempt + 1))
    return orig_text

def _safe_batch_translate(batch, target_lang):
    """
    Traduction de secours groupée via translate_batch :
    Envoie l'ensemble des phrases du lot en UN SEUL appel HTTP à Google Translate.
    Évite complètement le blocage IP de Google (5 req/sec) et traduit instantanément le batch complet.
    """
    target_clean = fix_code_language(target_lang)
    texts = [seg["text"].strip() if seg.get("text") else "" for seg in batch]
    non_empty = [t for t in texts if t]
    if not non_empty:
        return texts

    # 1. Tentative groupée en 1 seule requête pour tout le lot (ultra-rapide)
    try:
        tr = GoogleTranslator(source='auto', target=target_clean)
        res_list = tr.translate_batch(texts)
        if res_list and len(res_list) == len(texts):
            clean_res = []
            for r, orig in zip(res_list, texts):
                txt = str(r).strip() if r else orig
                clean_res.append(txt if not is_corrupted_translation(txt) else orig)
            logger.info(f"Secours groupé Google Translate réussi ({len(clean_res)} phrases en 1 seul appel).")
            return clean_res
    except Exception as e:
        logger.warning(f"translate_batch groupé direct a échoué ({e}), passage en secours individuel temporisé...")

    # 2. Secours individuel sécurisé avec temporisation anti-ban (> 1 seconde d'écart)
    clean_res = []
    for s in texts:
        if s:
            clean_res.append(_single_translate(s, target_clean))
            time.sleep(1.0)
        else:
            clean_res.append("")
    return clean_res

# ==============================================================================
# BATCHING CONTEXTUEL SÉCURISÉ AVEC GESTION DU RATE-LIMIT
# ==============================================================================
def _batch_with_context(segments, batch_size, translate_func, desc, target_lang):
    translated = copy.deepcopy(segments)
    progress = tqdm(total=len(segments), desc=desc)
    context = []
    
    # On génère le prompt dynamique pour la langue cible
    sys_prompt = get_system_prompt(target_lang)
    target_clean = fix_code_language(target_lang)

    start = 0
    while start < len(segments):
        end = min(start + batch_size, len(segments))
        batch = translated[start:end]
        batch_len = len(batch)

        previous = "\n".join([f"Précédent {i+1}: {c}" for i, c in enumerate(context[-3:])])
        lines_text = "\n".join([f"{i+1}. {seg['text'].strip()}" for i, seg in enumerate(batch)])

        context_part = f"Contexte précédent :\n{previous}\n\n" if context else ""
        full_prompt = (
            f"{sys_prompt}\n\n"
            f"{context_part}"
            f"Lignes à traduire ({batch_len} phrases) :\n{lines_text}\n\n"
            f"Renvoie STRICTEMENT un objet JSON avec les clés de '1' à '{batch_len}' associées à leur traduction respective :"
        )

        translated_lines = None
        for attempt in range(3):
            try:
                translated_lines = translate_func(full_prompt, batch_len)
                if translated_lines and len(translated_lines) == batch_len:
                    break
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg or "503" in err_msg or "unavailable" in err_msg:
                    sleep_time = 12.0 * (attempt + 1)
                    logger.warning(f"Quota/Saturation API au batch {start}-{end}. Pause de {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Tentative {attempt+1} échouée pour le batch {start}-{end}: {e}")
                    time.sleep(1.0 * (attempt + 1))

        # Application intelligente et sécurisée des traductions
        clean_regex = re.compile(
            r'^\s*[\*\-•]?\s*(?:\*\*)?(?:Line|Ligne|Phrase|Sentence)?\s*\[?\d+\]?[\.\)\:\-\s]*(?:\*\*)?[\:\-\s]*',
            re.IGNORECASE
        )

        if translated_lines and len(translated_lines) == batch_len:
            # 1. Correspondance 1:1 exacte et parfaite
            for j in range(batch_len):
                clean = clean_regex.sub('', str(translated_lines[j])).strip()
                clean = re.sub(r'^["\']|["\']$', '', clean).strip()
                is_untranslated = (target_clean != "en" and clean.lower() == batch[j]["text"].strip().lower() and len(clean) > 5)
                if clean and not is_corrupted_translation(clean) and not is_untranslated:
                    translated[start + j]["text"] = clean
                else:
                    translated[start + j]["text"] = _single_translate(batch[j]["text"], target_lang)
            context.extend([translated[start + j]["text"] for j in range(batch_len)])
            progress.update(batch_len)
            start += batch_size
            time.sleep(0.5)
        elif translated_lines and len(translated_lines) >= int(batch_len * 0.7):
            # 2. Le modèle a renvoyé au moins 70% des lignes
            for j in range(batch_len):
                if j < len(translated_lines):
                    clean = clean_regex.sub('', str(translated_lines[j])).strip()
                    clean = re.sub(r'^["\']|["\']$', '', clean).strip()
                    is_untranslated = (target_clean != "en" and clean.lower() == batch[j]["text"].strip().lower() and len(clean) > 5)
                    if clean and not is_corrupted_translation(clean) and not is_untranslated:
                        translated[start + j]["text"] = clean
                    else:
                        translated[start + j]["text"] = _single_translate(batch[j]["text"], target_lang)
                else:
                    translated[start + j]["text"] = _single_translate(batch[j]["text"], target_lang)
            context.extend([translated[start + j]["text"] for j in range(batch_len)])
            progress.update(batch_len)
            start += batch_size
            time.sleep(0.5)
        else:
            # 3. Moins de 70% ou échec complet : bascule groupée de sécurité (1 seul appel pour tout le batch !)
            logger.warning(f"Batch {start}-{end} incomplet ({len(translated_lines) if translated_lines else 0}/{batch_len} lignes). Secours groupé activé...")
            fallback_res = _safe_batch_translate(batch, target_lang)
            for j in range(batch_len):
                if j < len(fallback_res) and fallback_res[j]:
                    translated[start + j]["text"] = fallback_res[j]
                else:
                    translated[start + j]["text"] = _single_translate(batch[j]["text"], target_lang)
            context.extend([translated[start + j]["text"] for j in range(batch_len)])
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
# GESTION DES CLÉS & RATE LIMITING INTELLIGENT GEMINI (15 RPM / QUOTA PROTÉGÉ)
# ==============================================================================
_ACTIVE_GEMINI_INDEX = 0
_ACTIVE_KEY_INDEX = 0
_LAST_GEMINI_CALL_TIME = 0.0

def get_gemini_api_keys():
    """Récupère une ou plusieurs clés d'API Google Gemini (supporte le multi-clés séparé par virgule)."""
    raw = os.getenv("GOOGLE_API_KEY") or os.getenv("google_api_key") or ""
    keys = []
    for part in re.split(r'[,;\n]+', raw):
        k = part.strip()
        if k and k not in keys:
            keys.append(k)
    for i in range(1, 10):
        extra = os.getenv(f"GOOGLE_API_KEY_{i}") or os.getenv(f"google_api_key_{i}")
        if extra and extra.strip() and extra.strip() not in keys:
            keys.append(extra.strip())
    return keys

def _wait_for_gemini_pacing(min_interval=4.3):
    """
    Régulateur de débit préventif :
    Google AI Studio autorise 15 requêtes/minute (1 requête toutes les 4 secondes).
    Cet espacement garantit qu'on ne dépasse JAMAIS les 15 RPM, même sur 1 heure de vidéo !
    """
    global _LAST_GEMINI_CALL_TIME
    now = time.time()
    elapsed = now - _LAST_GEMINI_CALL_TIME
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_GEMINI_CALL_TIME = time.time()

def gemini_translate(segments, target, source=None, mode="flash"):
    global _ACTIVE_GEMINI_INDEX, _ACTIVE_KEY_INDEX
    keys = get_gemini_api_keys()
    if not keys:
        logger.error("❌ GEMINI: Clé GOOGLE_API_KEY manquante dans l'environnement !")
        return translate_iterative(segments, target, source)

    # Modèles officiels actifs recommandés par Google (zéro modèle 404 obsolète)
    if mode == "pro":
        candidates = ["gemini-3.1-pro-preview", "gemini-3.1-pro"]
    else:
        candidates = [
            "gemini-3.8-flash",       # Modèle officiel actif recommandé par Google
            "gemini-3.5-flash",       # Modèle officiel actif
            "gemini-3.5-flash-lite",  # Modèle officiel actif ultra-rapide
        ]

    # Prompt système dynamique par langue
    sys_prompt = get_system_prompt(target)

    config_kwargs = {
        "temperature": 0.2,
        "max_output_tokens": 4000,
        "response_mime_type": "application/json",
        "system_instruction": sys_prompt
    }

    # Désactiver les tokens de pensée (thinking) pour économiser 90% des tokens et accélérer
    try:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    safety = get_gemini_safety_settings()
    if safety:
        config_kwargs["safety_settings"] = safety

    config = types.GenerateContentConfig(**config_kwargs)

    def call_gemini(full_prompt, batch_len):
        global _ACTIVE_GEMINI_INDEX, _ACTIVE_KEY_INDEX
        last_err = None

        current_keys = get_gemini_api_keys() or keys
        num_keys = len(current_keys)
        num_models = len(candidates)

        for key_offset in range(num_keys):
            k_idx = (_ACTIVE_KEY_INDEX + key_offset) % num_keys
            current_key = current_keys[k_idx]
            current_client = genai.Client(api_key=current_key)

            for offset in range(num_models):
                idx = (_ACTIVE_GEMINI_INDEX + offset) % num_models
                m_id = candidates[idx]

                # Régulation stricte avant chaque tentative pour ne jamais dépasser 15 RPM
                _wait_for_gemini_pacing(min_interval=4.3)

                try:
                    response = current_client.models.generate_content(
                        model=m_id, 
                        contents=full_prompt, 
                        config=config
                    )
                    res = parse_llm_response(response.text, batch_len)
                    if res and len(res) > 0:
                        _ACTIVE_GEMINI_INDEX = idx
                        _ACTIVE_KEY_INDEX = k_idx
                        return res
                except Exception as e:
                    last_err = e
                    err_str = str(e).lower()
                    if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                        logger.warning(f"Quota/Rate-limit atteint sur '{m_id}' (Clé #{k_idx+1}).")
                        if num_keys > 1 and key_offset < num_keys - 1:
                            logger.info("Bascule immédiate sur la clé API Gemini suivante...")
                            break
                        else:
                            time.sleep(12.0)
                            continue
                    elif "404" in err_str or "not_found" in err_str:
                        logger.warning(f"Modèle '{m_id}' indisponible (404). Essai du modèle suivant...")
                        continue
                    else:
                        logger.warning(f"Modèle '{m_id}' indisponible ({e}). Essai du suivant...")
                        continue

        if last_err:
            raise last_err
        return None

    # Paquets de 25 phrases : réduit le nombre de requêtes de 60% (seulement ~40 requêtes pour 1 heure de vidéo !)
    return _batch_with_context(segments, 25, call_gemini, f"Translating (Gemini {mode.upper()} Optimized)", target)

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

    return _batch_with_context(segments, 12, call_groq, "Translating (Groq LLaMA-3.3 Batch)", target)

# ==============================================================================
# ZEPHYR (Hugging Face Inference)
# ==============================================================================
def hf_zephyr_translate(segments, target, source=None, batch_size=10):
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
    target_clean = fix_code_language(target)
    source_clean = fix_code_language(source) if source and source != "auto" else "auto"
    try:
        translator = GoogleTranslator(source=source_clean, target=target_clean)
    except Exception as e:
        logger.error(f"Erreur init GoogleTranslator (source={source_clean}, target={target_clean}): {e}")
        translator = GoogleTranslator(source='auto', target='en')
    for line in tqdm(range(len(segments_)), desc="Translating (Iterative)"):
        text = segments_[line]["text"]
        try:
            res = translator.translate(text.strip())
            if res and not is_corrupted_translation(res):
                segments_[line]["text"] = res
        except Exception as e:
            logger.error(f"Error google iterative ligne {line}: {e}")
    return segments_

def translate_batch(segments, target, chunk_size=2000, source=None):
    segments_copy = copy.deepcopy(segments)
    target_clean = fix_code_language(target)
    source_clean = fix_code_language(source) if source and source != "auto" else "auto"
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
    try:
        translator = GoogleTranslator(source=source_clean, target=target_clean)
    except Exception as e:
        logger.warning(f"GoogleTranslator batch init error ({e}), bascule vers iterative...")
        return translate_iterative(segments, target_clean, source_clean)
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
    except Exception as e:
        logger.warning(f"Bascule de translate_batch vers translate_iterative: {e}")
        try: progress_bar.close()
        except: pass
        return translate_iterative(segments, target_clean, source_clean)
        
    translated_lines = list(chain.from_iterable(split_list))
    return verify_translate(segments, segments_copy, translated_lines, target_clean, source_clean)

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
            try:
                translated = gemini_translate(segments, target, source, mode="flash")
            except Exception as e:
                logger.error(f"Erreur globale Gemini Flash ({e}). Bascule de secours automatique...")
                translated = translate_batch(segments, target_clean, chunk_size, source_clean)
        case "gemini_pro":
            try:
                translated = gemini_translate(segments, target, source, mode="pro")
            except Exception as e:
                logger.error(f"Erreur globale Gemini Pro ({e}). Bascule de secours automatique...")
                translated = translate_batch(segments, target_clean, chunk_size, source_clean)
        case "groq_llama3":
            try:
                translated = groq_translate(segments, target, source)
            except Exception as e:
                logger.error(f"Erreur globale Groq ({e}). Bascule de secours automatique...")
                translated = translate_batch(segments, target_clean, chunk_size, source_clean)
        case "hf_zephyr_7b_beta":
            try:
                translated = hf_zephyr_translate(segments, target, source)
            except Exception as e:
                logger.error(f"Erreur globale Zephyr ({e}). Bascule de secours automatique...")
                translated = translate_batch(segments, target_clean, chunk_size, source_clean)
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
