import logging

class BaseTTSProvider:
    def __init__(self):
        self.logger = logging.getLogger("soni_translate")

    def generate(self, text, voice, target_lang, output_file, **kwargs):
        """
        Génère un fichier audio à partir d'un texte.
        Doit être implémentée par les classes enfants.
        """
        raise NotImplementedError("La méthode generate() doit être implémentée par le provider.")
