from __future__ import annotations

import argparse
import queue
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque

import numpy as np
import sounddevice as sd
import voice_brain
from openwakeword import Model as WakeWordModel
from openwakeword.utils import download_models
from faster_whisper import WhisperModel


WAKEWORD_NAME = "hey_jarvis"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_SIZE = 1280
DEFAULT_WAKEWORD_THRESHOLD = 0.2
DEFAULT_PRE_ROLL_SECONDS = 0.8
DEFAULT_SILENCE_SECONDS = 1.0
DEFAULT_MIN_SPEECH_SECONDS = 0.4
DEFAULT_MAX_UTTERANCE_SECONDS = 15.0
DEFAULT_SPEECH_RMS_THRESHOLD = 0.01
DEFAULT_WAKEWORD_LOCKOUT_SECONDS = 5.0
DEFAULT_TRANSCRIPTION_LANGUAGE = "fr"
DEFAULT_WHISPER_MODEL = "base"


@dataclass(slots=True)
class ListenerConfig:
    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_size: int = DEFAULT_BLOCK_SIZE
    wakeword_threshold: float = DEFAULT_WAKEWORD_THRESHOLD
    pre_roll_seconds: float = DEFAULT_PRE_ROLL_SECONDS
    silence_seconds: float = DEFAULT_SILENCE_SECONDS
    min_speech_seconds: float = DEFAULT_MIN_SPEECH_SECONDS
    max_utterance_seconds: float = DEFAULT_MAX_UTTERANCE_SECONDS
    speech_rms_threshold: float = DEFAULT_SPEECH_RMS_THRESHOLD
    wakeword_lockout_seconds: float = DEFAULT_WAKEWORD_LOCKOUT_SECONDS
    whisper_language: str = DEFAULT_TRANSCRIPTION_LANGUAGE
    whisper_model_size: str = DEFAULT_WHISPER_MODEL
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32), dtype=np.float32)))


def _to_pcm16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def _seconds_to_blocks(seconds: float, sample_rate: int, block_size: int) -> int:
    return max(1, int(round(seconds * sample_rate / block_size)))


def _resolve_wakeword_model_path() -> str:
    import openwakeword
    from pathlib import Path

    models_dir = Path(openwakeword.__file__).resolve().parent / "resources" / "models"
    model_path = models_dir / "hey_jarvis_v0.1.onnx"

    if not model_path.exists():
        download_models(model_names=["hey_jarvis"], target_directory=str(models_dir))

    if not model_path.exists():
        raise RuntimeError(
            "Le modele openWakeWord hey_jarvis_v0.1.onnx est introuvable apres telechargement."
        )

    return str(model_path)


class PermanentSpeechListener:
    def __init__(
        self,
        config: ListenerConfig,
        presence_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self._presence_hook = presence_hook
        wakeword_model_path = _resolve_wakeword_model_path()
        self._wakeword_model = WakeWordModel(
            wakeword_models=[wakeword_model_path],
            inference_framework="onnx",
        )
        self._wakeword_key = next(iter(self._wakeword_model.models.keys()))
        self._whisper_model: WhisperModel | None = None
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stop_requested = False
        self._pre_roll: Deque[np.ndarray] = deque(
            maxlen=_seconds_to_blocks(
                self.config.pre_roll_seconds,
                self.config.sample_rate,
                self.config.block_size,
            )
        )
        self._wakeword_hits = 0
        self._last_block_report = 0.0
        self._wakeword_cooldown_until = 0.0

    def _load_whisper(self) -> WhisperModel:
        if self._whisper_model is None:
            self._whisper_model = WhisperModel(
                self.config.whisper_model_size,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute_type,
            )
        return self._whisper_model

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            pass

        block = np.asarray(indata[:, 0], dtype=np.float32).copy()
        try:
            self._audio_queue.put_nowait(block)
        except queue.Full:
            pass

    def _transcribe(self, audio: np.ndarray) -> str:
        whisper_model = self._load_whisper()
        segments, _info = whisper_model.transcribe(
            audio,
            language=self.config.whisper_language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
        )

        transcript = " ".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        )
        return _clean_text(transcript)

    def _ask_brain(self, question: str) -> str:
        answer_text = voice_brain.ask_and_speak(
            question=question,
            system_prompt=(
                "Tu es Jarvis. Reponds en francais, de facon directe, courte et utile."
            ),
            temperature=0.0,
            presence_hook=self._presence_hook,
        )

        answer_text = _clean_text(answer_text)
        return answer_text

    def _handle_idle_block(self, block: np.ndarray) -> bool:
        now = time.monotonic()
        if now < self._wakeword_cooldown_until:
            return False

        self._pre_roll.append(block)
        wakeword_block = _to_pcm16(block)
        scores = self._wakeword_model.predict(wakeword_block)
        score = float(scores.get(self._wakeword_key, 0.0))

        if now - self._last_block_report >= 5.0:
            self._last_block_report = now

        if score >= self.config.wakeword_threshold:
            self._wakeword_hits += 1
        else:
            self._wakeword_hits = 0

        if self._wakeword_hits >= 2:
            self._wakeword_hits = 0
            self._wakeword_model.reset()
            self._wakeword_cooldown_until = now + self.config.wakeword_lockout_seconds
            if self._presence_hook is not None:
                self._presence_hook("listening")
            return True

        return False

    def _drain_audio_queue(self) -> int:
        drained_blocks = 0
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
            else:
                drained_blocks += 1
        return drained_blocks

    def _collect_until_silence(self, first_block: np.ndarray) -> None:
        self._wakeword_cooldown_until = time.monotonic() + self.config.wakeword_lockout_seconds
        drained_blocks = self._drain_audio_queue()

        captured_blocks: list[np.ndarray] = list(self._pre_roll)
        captured_blocks.append(first_block)
        self._pre_roll.clear()

        start_time = time.monotonic()
        speech_detected = _audio_rms(first_block) >= self.config.speech_rms_threshold
        last_voice_time = start_time if speech_detected else 0.0

        while not self._stop_requested:
            try:
                block = self._audio_queue.get(timeout=0.2)
            except queue.Empty:
                block = None

            now = time.monotonic()
            if block is not None:
                captured_blocks.append(block)
                rms = _audio_rms(block)
                if rms >= self.config.speech_rms_threshold:
                    last_voice_time = now
                    speech_detected = True

            elapsed = now - start_time
            silence_elapsed = now - last_voice_time if speech_detected else 0.0

            if speech_detected and elapsed >= self.config.min_speech_seconds and silence_elapsed >= self.config.silence_seconds:
                break

            if elapsed >= self.config.max_utterance_seconds:
                break

        audio = np.concatenate(captured_blocks)
        try:
            text = self._transcribe(audio)
        except Exception:
            if self._presence_hook is not None:
                self._presence_hook("hide_overlay")
            return

        if text:
            try:
                if self._presence_hook is not None:
                    self._presence_hook("thinking")
                self._ask_brain(text)
            except Exception:
                pass
        elif self._presence_hook is not None:
            self._presence_hook("hide_overlay")

        self._wakeword_model.reset()
        self._wakeword_cooldown_until = time.monotonic() + self.config.wakeword_lockout_seconds
        drained_blocks = self._drain_audio_queue()

    def run(self) -> None:
        with sd.InputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.block_size,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        ):
            while not self._stop_requested:
                if time.monotonic() < self._wakeword_cooldown_until:
                    try:
                        self._audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        pass
                    continue

                try:
                    block = self._audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if self._handle_idle_block(block):
                    self._collect_until_silence(block)

    def stop(self) -> None:
        self._stop_requested = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ecoute permanente avec Hey Jarvis puis transcription Whisper."
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help="Nom du modele faster-whisper a charger.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_TRANSCRIPTION_LANGUAGE,
        help="Code langue pour Whisper, par exemple fr ou en.",
    )
    parser.add_argument(
        "--wakeword-threshold",
        type=float,
        default=DEFAULT_WAKEWORD_THRESHOLD,
        help="Seuil de declenchement pour openWakeWord.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = ListenerConfig(
        wakeword_threshold=args.wakeword_threshold,
        whisper_model_size=args.whisper_model,
        whisper_language=args.language,
    )
    listener = PermanentSpeechListener(config)

    try:
        listener.run()
    except KeyboardInterrupt:
        listener.stop()


if __name__ == "__main__":
    main()
