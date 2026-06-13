import base64
import queue
import re
import subprocess
import threading
from typing import Callable, Optional
from urllib.parse import quote

import brain


_TOKEN_RE = re.compile(r"\S+\s*", re.DOTALL)
_STRONG_PUNCTUATION_ENDINGS = (".", "!", "?", ";", ":")
_SOFT_PUNCTUATION_ENDINGS = (",",)


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _split_ready_chunks(
    buffer: str,
    min_words: int = 18,
    max_words: int = 32,
) -> tuple[list[str], str]:
    ready: list[str] = []
    matches = list(_TOKEN_RE.finditer(buffer))

    if not matches:
        return ready, buffer

    complete_upto = len(matches)
    last_match = matches[-1]
    last_token = last_match.group(0)

    # Garde le dernier token en attente s'il semble encore incomplet.
    if (
        not last_token[-1].isspace()
        and not last_token.rstrip().endswith(_STRONG_PUNCTUATION_ENDINGS + _SOFT_PUNCTUATION_ENDINGS)
    ):
        complete_upto -= 1

    if complete_upto <= 0:
        return ready, buffer

    segment_start = 0
    words_in_segment = 0
    residual_start = 0

    for index in range(complete_upto):
        token_text = matches[index].group(0)
        stripped = token_text.strip()
        if not stripped:
            continue

        words_in_segment += _count_words(stripped)
        should_flush = False

        if stripped.endswith(_STRONG_PUNCTUATION_ENDINGS):
            should_flush = words_in_segment >= 6
        elif stripped.endswith(_SOFT_PUNCTUATION_ENDINGS):
            should_flush = words_in_segment >= min_words
        elif words_in_segment >= max_words:
            should_flush = True

        if should_flush:
            chunk = buffer[segment_start:matches[index].end()].strip()
            if chunk:
                ready.append(chunk)
            segment_start = matches[index].end()
            residual_start = segment_start
            words_in_segment = 0

    if segment_start == 0:
        return ready, buffer

    return ready, buffer[residual_start:]


def _extract_first_sentence(buffer: str) -> tuple[str | None, str]:
    trimmed = buffer.strip()
    if not trimmed:
        return None, buffer

    sentence_match = re.search(r"^(.+?[.!?;:])(?:\s|$)", trimmed, re.DOTALL)
    if not sentence_match:
        return None, buffer

    sentence = sentence_match.group(1).strip()
    remainder = trimmed[sentence_match.end():].lstrip()
    return sentence, remainder


def _powershell_speak(text: str) -> None:
    encoded_text = quote(text)
    script = f"""
Add-Type -AssemblyName System.Speech
$text = [System.Uri]::UnescapeDataString('{encoded_text}')
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Rate = 0
$synth.Volume = 100
$synth.Speak($text)
"""
    encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    subprocess.run(
        [
            "powershell",
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-EncodedCommand",
            encoded_script,
        ],
        check=False,
        capture_output=True,
    )


class _FallbackSpeaker:
    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()
        self._closed = False
        self._stop_token = object()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._stop_token:
                    return
                text = str(item).strip()
                if text:
                    print(f"[jarvis] tts fallback: '{text}'", flush=True)
                    _powershell_speak(text)
            finally:
                self._queue.task_done()

    def speak(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("Le speaker est déjà fermé.")
        self._queue.put(text)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._stop_token)
        self._queue.join()
        self._worker.join(timeout=10)


def _build_speaker():
    try:
        from tts import StreamingSpeaker
        print("[jarvis] tts: moteur sherpa prêt", flush=True)
    except Exception:
        print("[jarvis] tts: fallback Windows", flush=True)
        return _FallbackSpeaker()
    return StreamingSpeaker()


def ask_and_speak(
    question: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    echo: bool = False,
    lead_chunks: int = 1,
    presence_hook: Callable[[str], None] | None = None,
) -> str:
    """
    Stream la reponse de brain.py et la lit a voix haute.

    On attend juste la fin de la première phrase avant de lancer sherpa,
    puis on reprend le découpage classique pour le reste.
    """
    speaker = _build_speaker()
    answer_parts: list[str] = []
    chunk_buffer = ""
    pending_chunks: list[str] = []
    speech_started = False

    try:
        if presence_hook is not None:
            presence_hook("thinking")
        print("[jarvis] cerveau: réflexion", flush=True)
        for chunk in brain.stream_ask(
            question=question,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            answer_parts.append(chunk)
            chunk_buffer += chunk

            if not speech_started:
                first_sentence, chunk_buffer = _extract_first_sentence(chunk_buffer)
                if first_sentence:
                    if presence_hook is not None:
                        presence_hook("speaking")
                    print("[jarvis] cerveau: début parole", flush=True)
                    if speaker is not None:
                        speaker.speak(first_sentence)
                    speech_started = True

            ready_chunks, chunk_buffer = _split_ready_chunks(chunk_buffer)
            pending_chunks.extend(ready_chunks)

            if speech_started:
                while pending_chunks:
                    text_chunk = pending_chunks.pop(0)
                    if speaker is not None:
                        speaker.speak(text_chunk)

        trailing = chunk_buffer.strip()
        if trailing:
            if not speech_started:
                if presence_hook is not None:
                    presence_hook("speaking")
                if speaker is not None:
                    speaker.speak(trailing)
                speech_started = True
            else:
                pending_chunks.append(trailing)

        if pending_chunks:
            for text_chunk in pending_chunks:
                if speaker is not None:
                    speaker.speak(text_chunk)

        full_answer = "".join(answer_parts).strip()
        print("[jarvis] cerveau: réponse complète reçue", flush=True)
        return full_answer
    finally:
        try:
            if speaker is not None:
                speaker.close()
        finally:
            if presence_hook is not None:
                presence_hook("hide_overlay")
            print("[jarvis] cerveau: fin", flush=True)


if __name__ == "__main__":
    ask_and_speak("Explique en trois phrases ce qu'est l'outil Ollama.")
