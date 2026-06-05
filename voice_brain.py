import re
from typing import Callable, Optional

import brain
import tts


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
    speaker = tts.StreamingSpeaker()
    answer_parts: list[str] = []
    chunk_buffer = ""
    pending_chunks: list[str] = []
    speech_started = False

    try:
        if presence_hook is not None:
            presence_hook("thinking")
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
                    speaker.speak(first_sentence)
                    speech_started = True

            ready_chunks, chunk_buffer = _split_ready_chunks(chunk_buffer)
            pending_chunks.extend(ready_chunks)

            if speech_started:
                while pending_chunks:
                    speaker.speak(pending_chunks.pop(0))

        trailing = chunk_buffer.strip()
        if trailing:
            if not speech_started:
                if presence_hook is not None:
                    presence_hook("speaking")
                speaker.speak(trailing)
                speech_started = True
            else:
                pending_chunks.append(trailing)

        if pending_chunks:
            for text_chunk in pending_chunks:
                speaker.speak(text_chunk)

        full_answer = "".join(answer_parts).strip()
        return full_answer
    finally:
        try:
            speaker.close()
        finally:
            if presence_hook is not None:
                presence_hook("hide_overlay")


if __name__ == "__main__":
    ask_and_speak("Explique en trois phrases ce qu'est l'outil Ollama.")
