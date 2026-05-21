import asyncio
import queue
import threading
from pathlib import Path
from typing import Final

import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro, SAMPLE_RATE


MODEL_PATH: Final[Path] = Path(__file__).with_name("kokoro-v1.0.onnx")
VOICES_PATH: Final[Path] = Path(__file__).with_name("voices-v1.0.bin")
DEFAULT_VOICE: Final[str] = "ff_siwis"
DEFAULT_LANG: Final[str] = "fr-fr"
DEFAULT_SPEED: Final[float] = 1.0
_STOP = object()


_kokoro_lock = threading.Lock()
_kokoro_instance: Kokoro | None = None


def _get_kokoro() -> Kokoro:
    global _kokoro_instance

    if _kokoro_instance is None:
        with _kokoro_lock:
            if _kokoro_instance is None:
                if not MODEL_PATH.exists():
                    raise FileNotFoundError(f"Modele Kokoro introuvable: {MODEL_PATH}")
                if not VOICES_PATH.exists():
                    raise FileNotFoundError(f"Voices Kokoro introuvable: {VOICES_PATH}")
                _kokoro_instance = Kokoro(str(MODEL_PATH), str(VOICES_PATH))

    return _kokoro_instance


class StreamingSpeaker:
    """
    Speaker Kokoro avec synthese et lecture audio en continu.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        lang: str = DEFAULT_LANG,
        speed: float = DEFAULT_SPEED,
        blocksize: int = 2048,
    ) -> None:
        self.voice = voice
        self.lang = lang
        self.speed = speed
        self.blocksize = blocksize

        self._text_queue: queue.Queue[object] = queue.Queue()
        self._audio_queue: queue.Queue[object] = queue.Queue(maxsize=64)
        self._current_audio = np.zeros(0, dtype=np.float32)
        self._current_offset = 0
        self._audio_finished = False

        self._synth_thread = threading.Thread(target=self._run_synth, daemon=True)
        self._play_thread = threading.Thread(target=self._run_playback, daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    async def _synthesize_text(self, text: str) -> None:
        kokoro = _get_kokoro()
        async for samples, _sample_rate in kokoro.create_stream(
            text=text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        ):
            self._audio_queue.put(np.asarray(samples, dtype=np.float32))

    def _run_synth(self) -> None:
        while True:
            item = self._text_queue.get()
            try:
                if item is _STOP:
                    self._audio_queue.put(_STOP)
                    return

                text = str(item).strip()
                if text:
                    asyncio.run(self._synthesize_text(text))
            finally:
                self._text_queue.task_done()

    def _pull_audio(self, frames: int) -> np.ndarray:
        output = np.zeros(frames, dtype=np.float32)
        written = 0

        while written < frames:
            if self._current_offset >= len(self._current_audio):
                if self._audio_finished:
                    break

                try:
                    item = self._audio_queue.get_nowait()
                except queue.Empty:
                    break

                if item is _STOP:
                    self._audio_finished = True
                    self._audio_queue.task_done()
                    break

                self._current_audio = np.asarray(item, dtype=np.float32).flatten()
                self._current_offset = 0
                self._audio_queue.task_done()

                if len(self._current_audio) == 0:
                    continue

            remaining_chunk = len(self._current_audio) - self._current_offset
            remaining_output = frames - written
            take = min(remaining_chunk, remaining_output)

            output[written : written + take] = self._current_audio[
                self._current_offset : self._current_offset + take
            ]
            self._current_offset += take
            written += take

        return output

    def _run_playback(self) -> None:
        def callback(outdata, frames, _time, _status) -> None:
            samples = self._pull_audio(frames)
            outdata[:, 0] = samples

        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=callback,
        ):
            while True:
                if (
                    self._audio_finished
                    and self._current_offset >= len(self._current_audio)
                    and self._audio_queue.empty()
                ):
                    break
                threading.Event().wait(0.02)

    def speak(self, text: str) -> None:
        self._text_queue.put(text)

    def close(self) -> None:
        self._text_queue.put(_STOP)
        self._text_queue.join()
        self._synth_thread.join()
        self._play_thread.join()


def parler(
    texte: str,
    voice: str = DEFAULT_VOICE,
    lang: str = DEFAULT_LANG,
    speed: float = DEFAULT_SPEED,
) -> None:
    """
    Lit un texte complet de facon bloquante avec Kokoro.
    """
    speaker = StreamingSpeaker(voice=voice, lang=lang, speed=speed)
    try:
        speaker.speak(texte)
    finally:
        speaker.close()
