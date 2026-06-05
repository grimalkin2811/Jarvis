from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _run_listen(args: argparse.Namespace) -> int:
    import wake_listener

    config = wake_listener.ListenerConfig(
        whisper_model_size=args.whisper_model,
        whisper_language=args.language,
        wakeword_threshold=args.wakeword_threshold,
    )
    listener = wake_listener.PermanentSpeechListener(config)

    try:
        listener.run()
    except KeyboardInterrupt:
        listener.stop()
    return 0


def _run_say(args: argparse.Namespace) -> int:
    import voice_brain

    if not args.text.strip():
        raise SystemExit("Le mode 'say' demande un texte avec --text.")

    voice_brain.ask_and_speak(
        question=args.text,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        echo=True,
    )
    return 0


def _run_ui(_args: argparse.Namespace) -> int:
    from UI.jarvis_menu import main as ui_main

    return ui_main()


def _run_desktop(_args: argparse.Namespace) -> int:
    import sys

    from PySide6.QtWidgets import QApplication

    from UI.jarvis_menu import MorphingOrbWidget
    from UI.screen_halo_overlay import ScreenHaloOverlay

    app = QApplication(sys.argv)
    blob_window = MorphingOrbWidget()
    overlay_window = ScreenHaloOverlay(blob_window.appearance_state)

    blob_window.showFullScreen()
    overlay_window.show_overlay()
    overlay_window.hide_overlay()

    return app.exec()


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
        help="Ouvre le blob et l'overlay dans le même processus.",
    )
    desktop.set_defaults(func=_run_desktop)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    if argv is None and len(sys.argv) == 1:
        argv = ["listen"]
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
