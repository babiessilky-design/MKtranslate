import os
import re
import shutil

def patch():
    candidates = [
        "/content/SoniTranslate/app_rvc.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_rvc.py"),
        "app_rvc.py"
    ]
    target_file = None
    for cand in candidates:
        if os.path.exists(cand):
            target_file = os.path.abspath(cand)
            break
            
    if not target_file:
        print("[MKtranslate Patch] app_rvc.py non trouvé.")
        return

    with open(target_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # 1. Définition et injection de is_custom_ai_voice
    custom_func_code = '''    def is_custom_ai_voice(v):
        if not isinstance(v, str):
            return False
        v_low = v.lower()
        return (
            v_low.startswith("kokoro/") or 
            v_low.startswith("gemini/") or 
            v_low.startswith("elevenlabs/") or 
            v_low.startswith("custom/") or
            v_low.startswith("chatterbox/") or
            v_low.startswith("omnivoice/")
        )

    # On regarde si au moins une voix "externe/custom" est demandée
    has_custom_tts = any(is_custom_ai_voice(v) for v in tts_voices)'''

    # Remplacement de l'ancien bloc has_custom_tts
    old_has_block_pattern = r'# On regarde si au moins une voix "externe/custom" est demandée\s+has_custom_tts = any\([^)]+\s+for v in tts_voices\s+\)'
    if re.search(old_has_block_pattern, content):
        content = re.sub(old_has_block_pattern, custom_func_code, content)
    elif "def is_custom_ai_voice" not in content:
        # Fallback de remplacement par chaîne
        content = content.replace(
            'has_custom_tts = any(',
            custom_func_code + '\n    _dummy = ('
        )

    # 2. Remplacement de l'ancien is_custom_voice dans la boucle par is_custom_ai_voice(voice)
    old_is_custom_pattern = r'is_custom_voice = isinstance\(voice, str\) and \([^)]+\)\s+if not is_custom_voice:\s+continue'
    content = re.sub(old_is_custom_pattern, 'if not is_custom_ai_voice(voice):\n            continue', content)

    # 3. Prise en charge de OmniVoice dans le elif de génération
    content = content.replace(
        'elif voice.startswith("Custom/") or voice.startswith("ChatterBox/") or voice.startswith("Chatterbox/"):',
        'elif voice.startswith("Custom/") or voice.startswith("ChatterBox/") or voice.startswith("Chatterbox/") or voice.startswith("OmniVoice/") or voice.startswith("Omnivoice/"):'
    )

    # 4. Détermination du moteur de clonage OmniVoice
    if 'voice.startswith("OmniVoice/")' not in content:
        content = content.replace(
            'if voice.startswith("ChatterBox/") or voice.startswith("Chatterbox/"):',
            'if voice.startswith("OmniVoice/") or voice.startswith("Omnivoice/"):\n                    cloning_engine = "OmniVoice"\n                elif voice.startswith("ChatterBox/") or voice.startswith("Chatterbox/"):')

    # 5. Moteur par défaut = OmniVoice
    content = content.replace('os.getenv("CUSTOM_CLONING_ENGINE", "F5-TTS")', 'os.getenv("CUSTOM_CLONING_ENGINE", "OmniVoice")')
    content = content.replace('custom_cloning_engine="F5-TTS"', 'custom_cloning_engine="OmniVoice"')
    content = content.replace('choices=["F5-TTS", "ChatterBox Multilingual", "OmniVoice"]', 'choices=["OmniVoice", "ChatterBox Multilingual", "F5-TTS"]')
    content = content.replace('value="F5-TTS"', 'value="OmniVoice"')

    # 6. Titre MKtranslate
    content = content.replace("📽️ SoniTranslate 🈷️", "📽️ MKtranslate 🈷️")
    content = content.replace('with gr.Blocks(theme=theme) as app:', 'with gr.Blocks(theme=theme, title="MKtranslate") as app:')

    # 7. Présence de OmniVoice et cosy dans la liste des voix de la bibliothèque
    if '"OmniVoice/cosy"' not in content:
        content = content.replace(
            'list_custom_library = []',
            'list_custom_library = ["OmniVoice/ref_homme_fr", "OmniVoice/ref_femme_fr", "OmniVoice/cosy", "Custom/ref_homme_fr", "Custom/ref_femme_fr", "Custom/cosy"]'
        )

    # 8. Synchronisation automatique min_speakers = max_speakers pour forcer PyAnote
    if "SYNCHRONISATION AUTOMATIQUE : MIN_SPEAKERS SUIT MAX_SPEAKERS" not in content:
        content = content.replace(
            'if not YOUR_HF_TOKEN:',
            '# SYNCHRONISATION AUTOMATIQUE : MIN_SPEAKERS SUIT MAX_SPEAKERS\n        if max_speakers > 1:\n            min_speakers = max_speakers\n            logger.info(f"Auto-Sync Speakers: min_speakers set to {min_speakers} (matching max_speakers={max_speakers})")\n        else:\n            min_speakers = 1\n\n        if not YOUR_HF_TOKEN:'
        )

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[MKtranslate Patch] {target_file} patché avec succès pour OmniVoice & MKtranslate !")

    # 8. Copie de cosy.wav dans voice_library si nécessaire
    base_dir = os.path.dirname(target_file)
    for vdir_name in ["voice_library", os.path.join("soni_translate", "voice_library")]:
        vdir = os.path.join(base_dir, vdir_name)
        if os.path.exists(vdir):
            f5_cosy = os.path.join(vdir, "f5-tts_cosy.wav")
            cosy = os.path.join(vdir, "cosy.wav")
            if os.path.exists(f5_cosy) and not os.path.exists(cosy):
                try:
                    shutil.copy(f5_cosy, cosy)
                    print(f"[MKtranslate Patch] cosy.wav créé à partir de f5-tts_cosy.wav dans {vdir}")
                except Exception as e:
                    print(f"[MKtranslate Patch] Erreur copie cosy.wav: {e}")

    # 9. Correction du bug zero-size array dans omnivoice
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith("omnivoice.py"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f_omni:
                        omni_code = f_omni.read()
                    changed = False
                    if "peak = np.abs(generated_audio).max()" in omni_code and "if generated_audio.size > 0:" not in omni_code:
                        omni_code = omni_code.replace(
                            "peak = np.abs(generated_audio).max()",
                            "peak = np.abs(generated_audio).max() if generated_audio.size > 0 else 0.0"
                        )
                        changed = True
                    if "postprocess_output=True" in omni_code:
                        omni_code = omni_code.replace("postprocess_output=True", "postprocess_output=False")
                        changed = True
                    if changed:
                        with open(fpath, "w", encoding="utf-8") as f_omni:
                            f_omni.write(omni_code)
                        print(f"[MKtranslate Patch] {fpath} patché contre le bug zero-size array !")
                except Exception:
                    pass

if __name__ == "__main__":
    patch()
