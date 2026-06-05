from __future__ import annotations

import json
import os
from dataclasses import dataclass


PERF_MODES = [
    {
        "label": "Lite",
        "model_label": "Qwen 2.5 0.5B",
        "candidates": ["qwen2.5:0.5b-instruct", "qwen2.5:0.5b", "qwen2.5"],
    },
    {
        "label": "Flash",
        "model_label": "Qwen 2.5 1.5B",
        "candidates": ["qwen2.5:1.5b-instruct", "qwen2.5:1.5b", "qwen2.5"],
    },
    {
        "label": "Medium",
        "model_label": "Phi 3",
        "candidates": ["phi3:mini-instruct", "phi3:mini", "phi3", "phi-3-mini"],
    },
    {
        "label": "Advanced",
        "model_label": "Mistral 7B",
        "candidates": ["mistral:7b-instruct", "mistral:7b", "mistral"],
    },
]


@dataclass
class SystemState:
    perf_mode_index: int = 3

    @property
    def perf_mode_label(self) -> str:
        return PERF_MODES[self.perf_mode_index % len(PERF_MODES)]["label"]

    @property
    def perf_mode_model_label(self) -> str:
        return PERF_MODES[self.perf_mode_index % len(PERF_MODES)]["model_label"]


def _state_payload(state: SystemState) -> dict[str, int]:
    return {"perf_mode_index": int(state.perf_mode_index)}


def _apply_payload(state: SystemState, payload: dict) -> None:
    state.perf_mode_index = int(payload.get("perf_mode_index", state.perf_mode_index)) % len(PERF_MODES)


def load_state(path: str) -> SystemState:
    state = SystemState()
    if not path or not os.path.exists(path):
        return state
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            _apply_payload(state, payload)
    except Exception:
        pass
    return state


def save_state(state: SystemState, path: str) -> None:
    if not path:
        return
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(_state_payload(state), handle, indent=2)
    except Exception:
        pass


def cycle_perf_mode(state: SystemState) -> None:
    state.perf_mode_index = (state.perf_mode_index + 1) % len(PERF_MODES)
    print(f"[system] perf_mode={state.perf_mode_label} -> {state.perf_mode_model_label}")


def perf_mode_info(state: SystemState) -> dict[str, str]:
    entry = PERF_MODES[state.perf_mode_index % len(PERF_MODES)]
    return {
        "label": entry["label"],
        "model_label": entry["model_label"],
        "model_candidates": entry["candidates"],
    }
