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

    # 0. S'assurer que 'import re' est bien présent
    if "\nimport re\n" not in content and "import re" not in content[:500]:
        content = content.replace("import sys\n", "import sys\nimport re\n")
    if "def sanitize_tts_text(t):\n        if not t:\n            return \"\"\n        import re" not in content:
        content = content.replace(
            "def sanitize_tts_text(t):\n        if not t:\n            return \"\"\n",
            "def sanitize_tts_text(t):\n        if not t:\n            return \"\"\n        import re\n"
        )

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

    if 'os.environ["MAX_SPEAKERS"] = str(max_speakers)' not in content:
        content = content.replace(
            'os.environ["CUSTOM_CLONING_ENGINE"] = custom_cloning_engine',
            'os.environ["CUSTOM_CLONING_ENGINE"] = custom_cloning_engine\n        os.environ["MAX_SPEAKERS"] = str(max_speakers)'
        )

    if 'max_speakers=max_speakers,' not in content:
        content = content.replace(
            'segment_duration_limit,\n                    )',
            'segment_duration_limit,\n                        max_speakers=max_speakers,\n                    )'
        )

    if 'reencode_speakers(copy.deepcopy(self.result), max_speakers=max_speakers)' not in content:
        content = content.replace(
            'self.result_diarize = reencode_speakers(copy.deepcopy(self.result))',
            'self.result_diarize = reencode_speakers(copy.deepcopy(self.result), max_speakers=max_speakers)'
        )

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[MKtranslate Patch] {target_file} patché avec succès pour OmniVoice & MKtranslate !")

    # 9. Copie de cosy.wav et cosy.txt dans voice_library si nécessaire
    base_dir = os.path.dirname(target_file)
    cosy_text_content = "It is during our darkest moments that we must focus to see the light."
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
            
            for c_txt in ["cosy.txt", "f5-tts_cosy.txt"]:
                c_path = os.path.join(vdir, c_txt)
                if not os.path.exists(c_path):
                    try:
                        with open(c_path, "w", encoding="utf-8") as f_ctxt:
                            f_ctxt.write(cosy_text_content)
                        print(f"[MKtranslate Patch] {c_txt} créé dans {vdir}")
                    except Exception as e_txt:
                        print(f"[MKtranslate Patch] Erreur création {c_txt}: {e_txt}")

    # 10. Correction du bug zero-size array et anti-hallucination dans omnivoice
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
                    if "OMNIVOICE_GUIDANCE_SCALE\", 2.6)" in omni_code:
                        omni_code = omni_code.replace("OMNIVOICE_GUIDANCE_SCALE\", 2.6)", "OMNIVOICE_GUIDANCE_SCALE\", 1.9)")
                        changed = True
                    if "num_step = int(kwargs.get(\"num_step\", 32))" in omni_code:
                        omni_code = omni_code.replace(
                            "num_step = int(kwargs.get(\"num_step\", 32))",
                            "num_step = int(kwargs.get(\"num_step\", os.getenv(\"OMNIVOICE_NUM_STEP\", 48)))"
                        )
                        changed = True
                    if changed:
                        with open(fpath, "w", encoding="utf-8") as f_omni:
                            f_omni.write(omni_code)
                        print(f"[MKtranslate Patch] {fpath} patché pour OmniVoice (48 steps / guidance 1.9) !")
                except Exception:
                    pass

    # 11. Correction du bug de répétition Chatterbox (Bug #549) et suppression du tatouage Perth
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname == "alignment_stream_analyzer.py":
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f_asa:
                        asa_code = f_asa.read()
                    if "len(set(self.generated_tokens[-2:])) == 1" in asa_code:
                        asa_code = asa_code.replace(
                            "len(set(self.generated_tokens[-2:])) == 1",
                            "len(self.generated_tokens) >= 4 and len(set(self.generated_tokens[-4:])) == 1"
                        )
                        with open(fpath, "w", encoding="utf-8") as f_asa:
                            f_asa.write(asa_code)
                        print(f"[MKtranslate Patch] {fpath} patché contre le bug de répétition Chatterbox !")
                except Exception:
                    pass
            elif fname in ["mtl_tts.py", "tts.py", "tts_turbo.py", "vc.py"]:
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f_mtl:
                        mtl_code = f_mtl.read()
                    changed_mtl = False
                    if "temperature=0.8," in mtl_code:
                        mtl_code = mtl_code.replace("temperature=0.8,", "temperature=0.35,")
                        changed_mtl = True
                    if "self.watermarker.apply_watermark(wav" in mtl_code:
                        mtl_code = mtl_code.replace(
                            "watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)",
                            "watermarked_wav = wav"
                        )
                        changed_mtl = True
                    if changed_mtl:
                        with open(fpath, "w", encoding="utf-8") as f_mtl:
                            f_mtl.write(mtl_code)
                        print(f"[MKtranslate Patch] {fpath} patché (pureté audio sans watermark / temperature=0.35) !")
                except Exception:
                    pass
            elif fname == "speech_segmentation.py":
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f_ss:
                        ss_code = f_ss.read()
                    if '"gemini-2.0-flash"' in ss_code:
                        ss_code = ss_code.replace('"gemini-2.0-flash"', '"gemini-3.8-flash"')
                        with open(fpath, "w", encoding="utf-8") as f_ss:
                            f_ss.write(ss_code)
                        print(f"[MKtranslate Patch] {fpath} patché : gemini-2.0-flash banni au profit de gemini-3.8-flash !")
                except Exception:
                    pass

if __name__ == "__main__":
    patch()
