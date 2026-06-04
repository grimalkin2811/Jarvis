from __future__ import annotations

import queue
import re
import tarfile
import threading
import urllib.request
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import sherpa_onnx
import sounddevice as sd


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
MODELS_DIR: Final[Path] = BASE_DIR / "models" / "tts"
DEFAULT_MODEL_NAME: Final[str] = "vits-piper-fr_FR-siwis-medium"
DEFAULT_MODEL_ARCHIVE_URL: Final[str] = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    f"{DEFAULT_MODEL_NAME}.tar.bz2"
)
DEFAULT_MODEL_DIR: Final[Path] = MODELS_DIR / DEFAULT_MODEL_NAME
DEFAULT_MODEL_PATH: Final[Path] = DEFAULT_MODEL_DIR / "fr_FR-siwis-medium.onnx"
DEFAULT_TOKENS_PATH: Final[Path] = DEFAULT_MODEL_DIR / "tokens.txt"
DEFAULT_DATA_DIR: Final[Path] = DEFAULT_MODEL_DIR / "espeak-ng-data"
DEFAULT_SPEAKER_ID: Final[int] = 0
DEFAULT_RATE: Final[int] = 175
DEFAULT_VOLUME: Final[float] = 1.0
DEFAULT_BUFFER_THRESHOLD_SECONDS: Final[float] = 1.2
_STOP = object()

_VOICE_ALIASES: Final[tuple[str, ...]] = (
    DEFAULT_MODEL_NAME,
    "default",
    "fr",
    "french",
    "francais",
    "gilles",
    "siwis",
)


def _normalize_text(text: str) -> str:
    normalized = " ".join(text.split())
    normalized = normalized.replace("...", ".")
    normalized = normalized.replace(";", ", ")
    normalized = normalized.replace(" - ", ", ")
    return normalized.strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;:])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _rate_to_speed(rate: int) -> float:
    if rate <= 0:
        return 0.75
    return max(0.5, min(1.8, rate / float(DEFAULT_RATE)))


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()

    def _is_within_destination(path: Path) -> bool:
        try:
            path.resolve().relative_to(destination)
            return True
        except ValueError:
            return False

    with tarfile.open(archive_path, mode="r:bz2") as tar:
        for member in tar.getmembers():
            member_path = destination / member.name
            if not _is_within_destination(member_path):
                raise RuntimeError(
                    f"Archive TTS invalide: chemin inattendu '{member.name}'."
                )
        tar.extractall(path=destination)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response, destination.open(
        "wb"
    ) as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _model_assets_ready() -> bool:
    return (
        DEFAULT_MODEL_PATH.exists()
        and DEFAULT_TOKENS_PATH.exists()
        and DEFAULT_DATA_DIR.exists()
    )


def _ensure_model_assets() -> Path:
    if _model_assets_ready():
        return DEFAULT_MODEL_DIR

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = MODELS_DIR / f"{DEFAULT_MODEL_NAME}.tar.bz2"

    if not archive_path.exists():
        print(f"[TTS] Telechargement du modele {DEFAULT_MODEL_NAME}...")
        _download_file(DEFAULT_MODEL_ARCHIVE_URL, archive_path)

    print(f"[TTS] Extraction du modele {DEFAULT_MODEL_NAME}...")
    _safe_extract_tar(archive_path, MODELS_DIR)

    try:
        archive_path.unlink()
    except OSError:
        pass

    if not _model_assets_ready():
        raise RuntimeError(
            "Le modele sherpa-onnx n'a pas pu etre prepare correctement."
        )

    return DEFAULT_MODEL_DIR


@lru_cache(maxsize=1)
def _build_tts_engine() -> sherpa_onnx.OfflineTts:
    model_dir = _ensure_model_assets()

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model_dir / "fr_FR-siwis-medium.onnx"),
                tokens=str(model_dir / "tokens.txt"),
                data_dir=str(model_dir / "espeak-ng-data"),
            ),
            num_threads=1,
            debug=False,
        )
    )
    if not config.validate():
        raise RuntimeError("La configuration sherpa-onnx est invalide.")

    return sherpa_onnx.OfflineTts(config)


def _pick_voice(preferred: str | None = None) -> int:
    if not preferred:
        return DEFAULT_SPEAKER_ID

    normalized = preferred.strip().lower()
    if normalized in _VOICE_ALIASES:
        return DEFAULT_SPEAKER_ID

    return DEFAULT_SPEAKER_ID


def available_voices() -> list[str]:
    return [
        "fr_FR-siwis-medium",
        "gilles",
        "siwis",
        "french",
        "default",
    ]


class StreamingSpeaker:
    """
    Speaker basé sur sherpa_onnx OfflineTts.

    Le moteur n'est pas natif streaming, donc on synthétise chaque chunk
    séparément puis on le lit dans l'ordre au fil de l'eau.
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
        self._tts = _build_tts_engine()
        self._speaker_id = _pick_voice(voice)
        self._speed = _rate_to_speed(rate)
        self._volume = max(0.0, volume)
        self._buffer_threshold_seconds = buffer_threshold_seconds
        self.voice_id = self._speaker_id
        self.voice_name = DEFAULT_MODEL_NAME
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

    def _play_audio(self, samples: list[float], sample_rate: int) -> None:
        audio = np.asarray(samples, dtype=np.float32)
        if audio.size == 0:
            return

        if self._volume != 1.0:
            audio = np.clip(audio * self._volume, -1.0, 1.0)

        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    def _run(self) -> None:
        for text in self._text_iterator():
            try:
                generated = self._tts.generate(
                    text=text,
                    sid=self._speaker_id,
                    speed=self._speed,
                )
                self._play_audio(generated.samples, generated.sample_rate)
            except Exception as exc:
                print(f"[TTS] Erreur pendant la synthèse ou la lecture: {exc}")

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
        self._worker.join(timeout=60)
        if self._worker.is_alive():
            try:
                sd.stop()
            except Exception:
                pass
            self._worker.join(timeout=5)


def parler(
    texte: str,
    voice: str | None = None,
    rate: int = DEFAULT_RATE,
    volume: float = DEFAULT_VOLUME,
) -> None:
    """
    Lit un texte complet de façon bloquante avec sherpa_onnx.
    """
    speaker = StreamingSpeaker(voice=voice, rate=rate, volume=volume)
    try:
        speaker.speak(texte)
    finally:
        speaker.close()


if __name__ == "__main__":
    print("Voix détectées :", available_voices())
    print("-" * 50)
    parler(
        "Bonjour Simon. Ceci est un test avec sherpa-onnx et le nouveau moteur TTS."
    )
