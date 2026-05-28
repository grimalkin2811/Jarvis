import queue
import re
import threading
import time
import winreg
from collections.abc import Iterator
from typing import Final

from RealtimeTTS import SystemEngine, TextToAudioStream


DEFAULT_VOICE_HINTS: Final[tuple[str, ...]] = (
    "Paul",
    "Microsoft Paul",
    "French",
    "Hortense",
)
DEFAULT_RATE: Final[int] = 175
DEFAULT_VOLUME: Final[float] = 1.0
DEFAULT_BUFFER_THRESHOLD_SECONDS: Final[float] = 1.2
_STOP = object()


def _normalize_text(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = normalized.replace("...", ".")
    normalized = normalized.replace(";", ", ")
    normalized = normalized.replace(" - ", ", ")
    return normalized.strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _get_absolute_onecore_voices() -> dict[str, str]:
    """Scanne le registre Windows pour associer le nom d'une voix OneCore à son ID absolu."""
    voices_map = {}
    path = r"SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens"
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
        for i in range(winreg.QueryInfoKey(key)[0]):
            subkey_name = winreg.EnumKey(key, i)
            subkey = winreg.OpenKey(key, subkey_name)
            try:
                voice_name = winreg.QueryValueEx(subkey, "")[0]
                full_id = rf"HKEY_LOCAL_MACHINE\{path}\{subkey_name}"
                voices_map[voice_name] = full_id
            except:
                continue
    except Exception:
        pass
    return voices_map


def _pick_voice(engine: SystemEngine, preferred: str | None = None) -> str:
    # 1. On récupère d'abord les voix classiques de l'engine (SAPI5, ou OneCore unifiées)
    engine_voices = engine.get_voices()
    
    # 2. On récupère aussi les voix OneCore cachées via le registre
    onecore_voices = _get_absolute_onecore_voices()

    hints = [preferred] if preferred else list(DEFAULT_VOICE_HINTS)

    # Étape A : On cherche d'abord dans les voix actives de l'engine (SAPI5 classiques, ou OneCore unifiées)
    for hint in hints:
        if not hint:
            continue
        hint_lower = hint.lower()
        for v in engine_voices:
            if hint_lower in v.name.lower() or hint_lower in v.id.lower():
                return v.id

    # Étape B : Si pas trouvé dans SAPI5, on cherche dans les voix OneCore cachées (Paul, Julie...)
    for hint in hints:
        if not hint:
            continue
        hint_lower = hint.lower()
        for name, full_id in onecore_voices.items():
            if hint_lower in name.lower():
                return full_id

    # Étape C : Fallback ultime
    if engine_voices:
        return engine_voices[0].id
    return ""


def available_voices() -> list[str]:
    engine = SystemEngine()
    try:
        # Affiche à la fois les voix classiques et détecte les OneCore dans le registre
        onecore = list(_get_absolute_onecore_voices().keys())
        classic = [voice.name for voice in engine.get_voices()]
        return list(set(onecore + classic))
    finally:
        engine.shutdown()


class StreamingSpeaker:
    """
    Speaker basé sur RealtimeTTS + SystemEngine (avec bypass Registre OneCore).
    """

    def __init__(
        self,
        voice: str | None = None,
        rate: int = DEFAULT_RATE,
        volume: float = DEFAULT_VOLUME,
        buffer_threshold_seconds: float = DEFAULT_BUFFER_THRESHOLD_SECONDS,
    ) -> None:
        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = False
        self._engine = SystemEngine()
        
        # 1. Recherche de l'ID de la voix (Paul ciblé via son chemin absolu)
        self.voice_id = _pick_voice(self._engine, voice)
        print(f"[TTS] ID Voix appliqué : {self.voice_id}")
        
        # 2. Création du stream
        self._stream = TextToAudioStream(
            self._engine,
            language="fr",
            tokenizer="nltk",
            muted=False,
        )
        
        # 3. On force l'ID absolu sur le moteur après l'init du stream
        try:
            self._engine.set_voice(self.voice_id)
        except Exception as e:
            print(f"[TTS] Erreur lors du set_voice initial, tentative de forçage direct : {e}")
            
        self._engine.set_voice_parameters(rate=rate, volume=volume)
        
        self._buffer_threshold_seconds = buffer_threshold_seconds
        self._playback_started = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _text_iterator(self) -> Iterator[str]:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return

                text = _normalize_text(str(item))
                if text:
                    yield text
            finally:
                self._queue.task_done()

    def _run(self) -> None:
        iterator = self._text_iterator()
        self._stream.feed(iterator)
        
        # On remet une couche juste avant le play pour s'assurer que l'engine n'a pas bougé
        try:
            self._engine.set_voice(self.voice_id)
        except:
            pass

        self._stream.play_async(
            fast_sentence_fragment=False,
            fast_sentence_fragment_allsentences=False,
            buffer_threshold_seconds=self._buffer_threshold_seconds,
            minimum_sentence_length=18,
            minimum_first_fragment_length=18,
            tokenizer="nltk",
            tokenize_sentences=_split_sentences,
            language="fr",
            sentence_fragment_delimiters=".?!;:\n",
            comma_silence_duration=0.05,
            sentence_silence_duration=0.12,
            default_silence_duration=0.05,
        )
        self._playback_started.set()

        while self._stream.is_playing():
            time.sleep(0.05)

    def speak(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("Le speaker est déjà fermé.")
        self._queue.put(text)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._queue.put(_STOP)
        self._queue.join()
        self._playback_started.wait(timeout=5)
        self._worker.join(timeout=60)
        try:
            self._stream.stop()
        except Exception:
            pass
        self._engine.shutdown()


def parler(
    texte: str,
    voice: str | None = None,
    rate: int = DEFAULT_RATE,
    volume: float = DEFAULT_VOLUME,
) -> None:
    """
    Lit un texte complet de façon bloquante avec RealtimeTTS/SystemEngine.
    """
    speaker = StreamingSpeaker(voice=voice, rate=rate, volume=volume)
    try:
        speaker.speak(texte)
    finally:
        speaker.close()


if __name__ == "__main__":
    print("Voix globales détectées (Classiques + Modernes) :", available_voices())
    print("-" * 50)
    
    parler("Bonjour Simon ! Ceci est un test avec le script mis à jour. Normalement, c'est bien la voix de Paul qui doit s'activer maintenant.")