from __future__ import annotations

import math
from typing import Callable

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    Property,
    QPropertyAnimation,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

from .appearance_actions import AppearanceState


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_presence_hook(overlay: "ScreenHaloOverlay") -> Callable[[str], None]:
    def _hook(state: str) -> None:
        action_map = {
            "listening": overlay.show_listening,
            "thinking": overlay.show_thinking,
            "speaking": overlay.show_speaking,
            "hide_overlay": overlay.hide_overlay,
            "hidden": overlay.hide_overlay,
        }
        action = action_map.get(state, overlay.hide_overlay)
        action()

    return _hook


class ScreenHaloOverlay(QWidget):
    overlayIntensityChanged = Signal(float)

    def __init__(self, appearance_state: AppearanceState | None = None) -> None:
        super().__init__()

        self.appearance_state = appearance_state or AppearanceState()
        self._overlay_intensity = 0.0
        self._presence_state = "hidden"
        self._animation_target = 0.0
        self._phase = 0.0

        window_flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        if hasattr(Qt, "WindowTransparentForInput"):
            window_flags |= Qt.WindowTransparentForInput
        if hasattr(Qt, "WindowDoesNotAcceptFocus"):
            window_flags |= Qt.WindowDoesNotAcceptFocus

        self.setWindowFlags(window_flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMouseTracking(False)

        self._fade_animation = QPropertyAnimation(self, b"overlay_intensity", self)
        self._fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_animation.setDuration(260)
        self._fade_animation.finished.connect(self._on_fade_finished)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick)
        self._pulse_timer.start(16)

    def _tick(self) -> None:
        self._phase = (self._phase + 0.045) % (math.tau * 8.0)
        if self.isVisible() or self._fade_animation.state() == QAbstractAnimation.State.Running:
            self.update()

    def _on_fade_finished(self) -> None:
        if self._animation_target <= 0.0 and self._overlay_intensity <= 0.001:
            super().hide()

    def _animate_to(self, target: float, duration: int = 260, presence_state: str | None = None) -> None:
        target = _clamp(target, 0.0, 1.0)
        self._animation_target = target
        if presence_state is not None:
            self._presence_state = presence_state

        if target > 0.0 and not self.isVisible():
            self.show_overlay()

        self._fade_animation.stop()
        self._fade_animation.setDuration(duration)
        self._fade_animation.setStartValue(self._overlay_intensity)
        self._fade_animation.setEndValue(target)
        self._fade_animation.start()

    def show_overlay(self) -> None:
        super().showFullScreen()
        self.raise_()

    def show_listening(self) -> None:
        self._animate_to(0.62, duration=220, presence_state="listening")

    def show_thinking(self) -> None:
        self._animate_to(0.78, duration=240, presence_state="thinking")

    def show_speaking(self) -> None:
        self._animate_to(1.0, duration=220, presence_state="speaking")

    def hide_overlay(self) -> None:
        self._animate_to(0.0, duration=280, presence_state="hidden")

    def get_overlay_intensity(self) -> float:
        return self._overlay_intensity

    def set_overlay_intensity(self, value: float) -> None:
        value = _clamp(float(value), 0.0, 1.0)
        if abs(self._overlay_intensity - value) < 1e-4:
            return
        self._overlay_intensity = value
        self.overlayIntensityChanged.emit(value)
        self.update()

    overlay_intensity = Property(
        float,
        get_overlay_intensity,
        set_overlay_intensity,
        notify=overlayIntensityChanged,
    )

    def _base_glow_color(self) -> QColor:
        return QColor(self.appearance_state.glow_color)

    def _accent_color(self) -> QColor:
        accent = QColor(self.appearance_state.text_color)
        accent.setAlpha(255)
        return accent

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(self.rect(), Qt.transparent)

            intensity = _clamp(self._overlay_intensity, 0.0, 1.0)
            if intensity <= 0.001:
                return

            width = float(self.width())
            height = float(self.height())
            edge_width = 22.0 + 52.0 * intensity
            corner_size = 120.0 + 90.0 * intensity
            pulse = 0.84 + 0.16 * math.sin(self._phase)
            base_color = self._base_glow_color()
            accent_color = self._accent_color()

            outer_alpha = int(32 + 90 * intensity * pulse)
            mid_alpha = int(78 + 120 * intensity * pulse)
            inner_alpha = int(18 + 36 * intensity)
            highlight_alpha = int(44 + 60 * intensity * pulse)

            outer = QColor(base_color)
            outer.setAlpha(outer_alpha)
            mid = QColor(base_color)
            mid.setAlpha(mid_alpha)
            inner = QColor(base_color)
            inner.setAlpha(inner_alpha)
            highlight = QColor(accent_color)
            highlight.setAlpha(highlight_alpha)

            transparent = QColor(0, 0, 0, 0)

            # Top and bottom halos.
            for y, is_top in ((0.0, True), (height - edge_width, False)):
                gradient = QLinearGradient(0.0, y, 0.0, y + edge_width)
                if is_top:
                    gradient.setColorAt(0.0, highlight)
                    gradient.setColorAt(0.22, outer)
                    gradient.setColorAt(0.58, mid)
                    gradient.setColorAt(1.0, transparent)
                else:
                    gradient.setColorAt(0.0, transparent)
                    gradient.setColorAt(0.42, mid)
                    gradient.setColorAt(0.78, outer)
                    gradient.setColorAt(1.0, highlight)
                painter.fillRect(0.0, y, width, edge_width, gradient)

            # Left and right halos.
            for x, is_left in ((0.0, True), (width - edge_width, False)):
                gradient = QLinearGradient(x, 0.0, x + edge_width, 0.0)
                if is_left:
                    gradient.setColorAt(0.0, highlight)
                    gradient.setColorAt(0.22, outer)
                    gradient.setColorAt(0.58, mid)
                    gradient.setColorAt(1.0, transparent)
                else:
                    gradient.setColorAt(0.0, transparent)
                    gradient.setColorAt(0.42, mid)
                    gradient.setColorAt(0.78, outer)
                    gradient.setColorAt(1.0, highlight)
                painter.fillRect(x, 0.0, edge_width, height, gradient)

            # Corners.
            corner_alpha = int(88 + 80 * intensity * pulse)
            corner_glow = QColor(base_color)
            corner_glow.setAlpha(corner_alpha)
            for center_x, center_y in (
                (0.0, 0.0),
                (width, 0.0),
                (0.0, height),
                (width, height),
            ):
                radial = QRadialGradient(center_x, center_y, corner_size)
                radial.setColorAt(0.0, corner_glow)
                radial.setColorAt(0.32, outer)
                radial.setColorAt(0.66, inner)
                radial.setColorAt(1.0, transparent)
                painter.setPen(Qt.NoPen)
                painter.setBrush(radial)
                painter.drawEllipse(
                    center_x - corner_size,
                    center_y - corner_size,
                    corner_size * 2.0,
                    corner_size * 2.0,
                )

            # Thin premium edge line.
            border_alpha = int(22 + 48 * intensity)
            border_color = QColor(base_color)
            border_color.setAlpha(border_alpha)
            painter.setPen(QPen(border_color, 1.2))
            inset = 1.0
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.rect().adjusted(int(inset), int(inset), -int(inset), -int(inset)))

            # Softer inner contour to keep the center empty but alive.
            contour_alpha = int(10 + 24 * intensity * pulse)
            contour_color = QColor(accent_color)
            contour_color.setAlpha(contour_alpha)
            painter.setPen(QPen(contour_color, 1.0))
            contour_margin = 14.0 + 20.0 * intensity
            painter.drawRect(
                self.rect().adjusted(
                    int(contour_margin),
                    int(contour_margin),
                    -int(contour_margin),
                    -int(contour_margin),
                )
            )
        finally:
            painter.end()
