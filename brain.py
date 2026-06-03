import json
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any, Iterator


OLLAMA_URL = "http://127.0.0.1:11434"


def _post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{OLLAMA_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama sur http://127.0.0.1:11434. "
            "Verifie que le service Ollama est demarre."
        ) from exc


def _get_json(path: str) -> dict[str, Any]:
    request = urllib.request.Request(url=f"{OLLAMA_URL}{path}", method="GET")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Impossible de recuperer la liste des modeles Ollama. "
            "Verifie que le service Ollama est demarre."
        ) from exc


@lru_cache(maxsize=1)
def _available_models() -> list[str]:
    data = _get_json("/api/tags")
    return [model["name"] for model in data.get("models", []) if model.get("name")]


def _find_model(candidates: list[str], available: list[str]) -> str | None:
    lowered = {name.lower(): name for name in available}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for candidate in candidates:
        prefix = candidate.lower().split(":")[0]
        for name in available:
            if name.lower().startswith(prefix):
                return name

    return None


@lru_cache(maxsize=1)
def _resolve_default_model() -> str:
    available = _available_models()
    if not available:
        raise RuntimeError("Aucun modele Ollama n'est installe.")

    qwen = _find_model(["qwen2.5:1.5b", "qwen2.5", "qwen"], available)
    if qwen:
        return qwen

    return available[0]


def stream_ask(
    question: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
) -> Iterator[str]:
    """
    Genere la reponse d'Ollama morceau par morceau.
    """
    if not question or not question.strip():
        raise ValueError("La question ne peut pas etre vide.")

    selected_model = _resolve_default_model()
    final_system = system_prompt or (
        "Tu es un assistant utile et precis. Reponds dans la langue de l'utilisateur. "
        "Ne rajoute pas de bla bla inutile, reponds simplement a la question. "
        "Si tu ne connais pas la reponse, dis que tu ne sais pas au lieu d'inventer une reponse. "
        "Utise des signes de ponctuation (virgules, points, deux-points, point-virgule...) pour segmenter tes reponses en petits morceaux. "
        "Quand la question concerne du code, donne une reponse concrete et exploitable."
    )

    body = json.dumps(
        {
            "model": selected_model,
            "stream": True,
            "messages": [
                {"role": "system", "content": final_system},
                {"role": "user", "content": question.strip()},
            ],
            "options": {
                "temperature": temperature,
            },
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        url=f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Impossible de contacter Ollama sur http://127.0.0.1:11434. "
            "Verifie que le service Ollama est demarre."
        ) from exc


def ask(
    question: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
    return_metadata: bool = False,
) -> str | dict[str, Any]:
    """
    Pose une question a Ollama.

    Pour l'instant, toutes les questions sont envoyees a qwen2.5:1.5b.
    """
    if not question or not question.strip():
        raise ValueError("La question ne peut pas etre vide.")

    selected_model = _resolve_default_model()
    final_system = system_prompt or (
        "Tu es un assistant utile et precis. Reponds dans la langue de l'utilisateur. "
        "Ne rajoute pas de bla bla inutile, réponds simplement a la question. "
        "Si tu ne connais pas la reponse, dis que tu ne sais pas au lieu d'inventer une reponse. "
        "Utise des signes de ponctuation (virgules, points, deux-points, point-virgule...) pour segmenter tes reponses en petits morceaux. "
        "Quand la question concerne du code, donne une reponse concrete et exploitable."
    )

    final_response = _post_json(
        "/api/chat",
        {
            "model": selected_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": final_system},
                {"role": "user", "content": question.strip()},
            ],
            "options": {
                "temperature": temperature,
            },
        },
    )

    answer = final_response.get("message", {}).get("content", "").strip()

    if return_metadata:
        return {
            "answer": answer,
            "selected_route": "qwen2.5:1.5b",
            "selected_model": selected_model,
            "routing_reason": "Routage temporairement desactive: qwen2.5:1.5b est force.",
        }

    return answer
