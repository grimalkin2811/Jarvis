from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Sequence

try:
    import sounddevice
except ImportError:
    pass


def _log(message: str) -> None:
    print(f"[jarvis] {message}", flush=True)


def _run_listen(args: argparse.Namespace) -> int:
    import wake_listener

    _log("mode listen: démarrage")
    config = wake_listener.ListenerConfig(
        whisper_model_size=args.whisper_model,
        whisper_language=args.language,
        wakeword_threshold=args.wakeword_threshold,
    )
    listener = wake_listener.PermanentSpeechListener(config)

    try:
        _log("listener prêt")
        listener.run()
    except KeyboardInterrupt:
        _log("arrêt demandé")
        listener.stop()
    return 0


def _run_say(args: argparse.Namespace) -> int:
    import voice_brain

    if not args.text.strip():
        raise SystemExit("Le mode 'say' demande un texte avec --text.")

    _log("mode say: envoi de la requête")
    voice_brain.ask_and_speak(
        question=args.text,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        echo=True,
    )
    _log("mode say: terminé")
    return 0


def _run_ui(_args: argparse.Namespace) -> int:
    from UI.jarvis_menu import main as ui_main

    return ui_main()


def _run_desktop(_args: argparse.Namespace) -> int:
    from PySide6.QtCore import QObject, Signal, Slot
    from PySide6.QtWidgets import QApplication

    from UI import appearance_actions
    from UI.screen_halo_overlay import ScreenHaloOverlay

    class PresenceBridge(QObject):
        presence_changed = Signal(str)

    class PresenceRouter(QObject):
        def __init__(self, overlay: ScreenHaloOverlay) -> None:
            super().__init__()
            self._overlay = overlay

        @Slot(str)
        def handle_presence_state(self, state: str) -> None:
            _log(f"état overlay -> {state}")
            if state == "listening":
                self._overlay.show_listening()
            elif state == "thinking":
                self._overlay.show_thinking()
            elif state == "speaking":
                self._overlay.show_speaking()
            else:
                self._overlay.hide_overlay()

    app = QApplication(sys.argv)
    _log("mode desktop: UI initialisée")
    appearance_state = appearance_actions.load_state(
        str(Path(__file__).resolve().parent / "UI" / "appearance_state.json")
    )
    overlay_window = ScreenHaloOverlay(appearance_state)
    presence_bridge = PresenceBridge()
    presence_router = PresenceRouter(overlay_window)
    presence_bridge.presence_changed.connect(presence_router.handle_presence_state)

    listener_holder: dict[str, object] = {}

    def _run_listener() -> None:
        try:
            _log("thread listener: import wake_listener")
            import wake_listener

            _log("thread listener: démarrage")
            config = wake_listener.ListenerConfig()
            _log("thread listener: création du listener")
            listener = wake_listener.PermanentSpeechListener(
                config=config,
                presence_hook=presence_bridge.presence_changed.emit,
            )
            listener_holder["listener"] = listener
            _log("thread listener: boucle audio lancée")
            listener.run()
            _log("thread listener: arrêt propre")
        except Exception as exc:
            listener_holder["error"] = exc
            print(f"[jarvis] listener error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    listener_thread = threading.Thread(target=_run_listener, daemon=True)

    def _shutdown() -> None:
        listener = listener_holder.get("listener")
        if listener is not None:
            listener.stop()
        if listener_thread.is_alive():
            listener_thread.join(timeout=2.0)

    app.aboutToQuit.connect(_shutdown)

    overlay_window.hide_overlay()
    _log("overlay masqué, écoute active")
    listener_thread.start()
    _log("thread listener: démarré")

    try:
        return app.exec()
    finally:
        _log("fermeture demandée")
        _shutdown()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lanceur principal de Jarvis.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    listen = subparsers.add_parser("listen", help="Ecoute le wake word puis repond.")
    listen.add_argument(
        "--whisper-model",
        default="base",
        help="Nom du modele faster-whisper a charger.",
    )
    listen.add_argument(
        "--language",
        default="fr",
        help="Code langue pour Whisper, par exemple fr ou en.",
    )
    listen.add_argument(
        "--wakeword-threshold",
        type=float,
        default=0.2,
        help="Seuil de declenchement pour openWakeWord.",
    )
    listen.set_defaults(func=_run_listen)

    say = subparsers.add_parser("say", help="Fait parler Jarvis sur un texte donne.")
    say.add_argument("--text", required=True, help="Texte a lire a voix haute.")
    say.add_argument(
        "--system-prompt",
        default=None,
        help="Prompt systeme optionnel pour le cerveau.",
    )
    say.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature du modele.",
    )
    say.set_defaults(func=_run_say)

    ui = subparsers.add_parser("ui", help="Ouvre l'interface Jarvis.")
    ui.set_defaults(func=_run_ui)

    desktop = subparsers.add_parser(
        "desktop",
        help="Ouvre seulement l'overlay de Jarvis.",
    )
    desktop.set_defaults(func=_run_desktop)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        argv = ["desktop"]
        _log("aucun argument fourni, bascule sur desktop")
    args = parser.parse_args(argv)
    _log(f"mode sélectionné: {args.mode}")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
