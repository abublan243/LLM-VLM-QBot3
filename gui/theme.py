"""
QBot3 design system — colours, typography, spacing tokens + a comprehensive
QSS stylesheet for every widget the app uses.

Design language
---------------
* Premium dark theme. Deep near-black background with elevated surfaces.
* Two accents: purple `#6C63FF` (primary, for actions / active states) and
  teal `#00D4AA` (secondary, for data + telemetry highlights).
* Generous corner radii (8 px on buttons, 12 px on cards/panels) — feels
  modern but never childish.
* Subtle borders + focus glow rather than heavy outlines.
* Type scale: 13 px base, 11 px caption, 16 / 20 / 28 px headings.

Usage
-----
    from gui.theme import apply_theme, Tokens
    apply_theme(QApplication.instance())
    accent = Tokens.ACCENT_PRIMARY
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    Qt,
    pyqtProperty,
)
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QPainter, QPalette, QPen
from PyQt6.QtWidgets import QApplication, QGraphicsDropShadowEffect, QPushButton, QWidget


# =====================================================================
# Tokens
# =====================================================================


@dataclass(frozen=True)
class Tokens:
    """Design tokens shared by QSS and Python widgets."""

    # ---- Colour palette ----
    BG: str = "#0D0D0F"
    SURFACE: str = "#1A1A1F"
    SURFACE_ELEVATED: str = "#222228"
    SURFACE_HOVER: str = "#2A2A32"
    BORDER: str = "#2E2E38"
    BORDER_BRIGHT: str = "#3A3A48"

    ACCENT_PRIMARY: str = "#6C63FF"
    ACCENT_PRIMARY_HOVER: str = "#8079FF"
    ACCENT_PRIMARY_DIM: str = "#4F49B8"
    ACCENT_SECONDARY: str = "#00D4AA"
    ACCENT_SECONDARY_HOVER: str = "#1FE7BD"

    TEXT_PRIMARY: str = "#F0F0F5"
    TEXT_SECONDARY: str = "#8888A0"
    TEXT_MUTED: str = "#5A5A6E"

    SUCCESS: str = "#22C55E"
    WARNING: str = "#F59E0B"
    DANGER: str = "#EF4444"
    DANGER_HOVER: str = "#F26060"
    INFO: str = "#3B82F6"

    # ---- Spacing / radii ----
    RADIUS_SM: int = 6
    RADIUS_MD: int = 8
    RADIUS_LG: int = 12
    RADIUS_PILL: int = 22
    PAD_SM: int = 6
    PAD_MD: int = 12
    PAD_LG: int = 20

    # ---- Typography ----
    FONT_FAMILY: str = "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif"
    FONT_FAMILY_MONO: str = "'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace"
    FONT_BASE_PX: int = 13


# =====================================================================
# QSS template
# =====================================================================


def _build_qss(t: Tokens) -> str:
    """Return the full QSS string with tokens substituted in."""
    return f"""
* {{
    font-family: {t.FONT_FAMILY};
    font-size: {t.FONT_BASE_PX}px;
    color: {t.TEXT_PRIMARY};
    outline: 0;
}}

QMainWindow,
QDialog {{
    background-color: {t.BG};
}}

QWidget {{
    background-color: transparent;
}}

QLabel {{
    color: {t.TEXT_PRIMARY};
}}

QLabel[role="caption"] {{
    color: {t.TEXT_SECONDARY};
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}

QLabel[role="heading"] {{
    font-size: 16px;
    font-weight: 600;
    color: {t.TEXT_PRIMARY};
}}

QLabel[role="display"] {{
    font-size: 28px;
    font-weight: 700;
    color: {t.TEXT_PRIMARY};
    font-family: {t.FONT_FAMILY_MONO};
}}

QLabel[role="muted"] {{
    color: {t.TEXT_MUTED};
}}

/* ---- Card surfaces ---- */
QFrame[role="card"],
QWidget[role="card"] {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_LG}px;
}}

QFrame[role="card-elevated"] {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_LG}px;
}}

/* ---- Group boxes (cards with a title) ---- */
QGroupBox {{
    background-color: {t.SURFACE};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_LG}px;
    margin-top: 16px;
    padding-top: 16px;
    color: {t.TEXT_PRIMARY};
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: -2px;
    padding: 0 6px;
    color: {t.TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    background-color: {t.BG};
}}

/* ---- Buttons ---- */
QPushButton {{
    background-color: {t.SURFACE_ELEVATED};
    color: {t.TEXT_PRIMARY};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_MD}px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 16px;
}}
QPushButton:hover {{
    background-color: {t.SURFACE_HOVER};
    border-color: {t.BORDER_BRIGHT};
}}
QPushButton:pressed {{
    background-color: {t.SURFACE};
}}
QPushButton:disabled {{
    color: {t.TEXT_MUTED};
    border-color: {t.BORDER};
    background-color: {t.SURFACE};
}}

QPushButton[variant="primary"] {{
    background-color: {t.ACCENT_PRIMARY};
    color: white;
    border: none;
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover {{
    background-color: {t.ACCENT_PRIMARY_HOVER};
}}
QPushButton[variant="primary"]:pressed {{
    background-color: {t.ACCENT_PRIMARY_DIM};
}}
QPushButton[variant="primary"]:disabled {{
    background-color: {t.ACCENT_PRIMARY_DIM};
    color: rgba(255, 255, 255, 100);
}}

QPushButton[variant="secondary"] {{
    background-color: transparent;
    color: {t.ACCENT_SECONDARY};
    border: 1px solid {t.ACCENT_SECONDARY};
}}
QPushButton[variant="secondary"]:hover {{
    background-color: rgba(0, 212, 170, 30);
}}

QPushButton[variant="danger"] {{
    background-color: {t.DANGER};
    color: white;
    border: none;
    font-weight: 600;
}}
QPushButton[variant="danger"]:hover {{
    background-color: {t.DANGER_HOVER};
}}

QPushButton[variant="ghost"] {{
    background-color: transparent;
    border: 1px solid {t.BORDER};
    color: {t.TEXT_SECONDARY};
}}
QPushButton[variant="ghost"]:hover {{
    color: {t.TEXT_PRIMARY};
    border-color: {t.BORDER_BRIGHT};
    background-color: {t.SURFACE_HOVER};
}}

QPushButton[variant="mode"] {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_PILL}px;
    padding: 10px 22px;
    font-weight: 600;
    min-width: 96px;
}}
QPushButton[variant="mode"]:hover {{
    background-color: {t.SURFACE_HOVER};
    color: {t.TEXT_PRIMARY};
}}
QPushButton[variant="mode"][active="true"] {{
    background-color: {t.ACCENT_PRIMARY};
    border-color: {t.ACCENT_PRIMARY_HOVER};
    color: white;
}}

QPushButton[variant="estop"] {{
    background-color: {t.DANGER};
    color: white;
    border: 2px solid {t.DANGER_HOVER};
    border-radius: {t.RADIUS_PILL}px;
    padding: 12px 26px;
    font-weight: 800;
    letter-spacing: 1.5px;
    font-size: 14px;
}}
QPushButton[variant="estop"]:hover {{
    background-color: {t.DANGER_HOVER};
}}

QPushButton[variant="dpad"] {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_MD}px;
    font-size: 18px;
    font-weight: 700;
    min-width: 56px;
    min-height: 56px;
}}
QPushButton[variant="dpad"]:pressed,
QPushButton[variant="dpad"]:hover {{
    background-color: {t.ACCENT_PRIMARY};
    border-color: {t.ACCENT_PRIMARY_HOVER};
    color: white;
}}

/* ---- Inputs ---- */
QLineEdit,
QTextEdit,
QPlainTextEdit,
QSpinBox,
QDoubleSpinBox {{
    background-color: {t.SURFACE_ELEVATED};
    color: {t.TEXT_PRIMARY};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_MD}px;
    padding: 8px 10px;
    selection-background-color: {t.ACCENT_PRIMARY};
}}
QLineEdit:focus,
QTextEdit:focus,
QPlainTextEdit:focus,
QSpinBox:focus,
QDoubleSpinBox:focus {{
    border: 1px solid {t.ACCENT_PRIMARY};
}}

QTextEdit[role="log"],
QPlainTextEdit[role="log"] {{
    font-family: {t.FONT_FAMILY_MONO};
    font-size: 12px;
    color: {t.TEXT_SECONDARY};
    background-color: {t.BG};
    border: 1px solid {t.BORDER};
}}

QComboBox {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_MD}px;
    padding: 7px 12px;
    min-width: 100px;
    color: {t.TEXT_PRIMARY};
}}
QComboBox:hover {{
    border-color: {t.BORDER_BRIGHT};
}}
QComboBox:focus {{
    border-color: {t.ACCENT_PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {t.TEXT_SECONDARY};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER_BRIGHT};
    border-radius: {t.RADIUS_MD}px;
    padding: 4px;
    selection-background-color: {t.ACCENT_PRIMARY};
    selection-color: white;
}}

QCheckBox,
QRadioButton {{
    color: {t.TEXT_PRIMARY};
    spacing: 8px;
}}
QCheckBox::indicator,
QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER_BRIGHT};
}}
QCheckBox::indicator {{
    border-radius: 4px;
}}
QRadioButton::indicator {{
    border-radius: 8px;
}}
QCheckBox::indicator:checked,
QRadioButton::indicator:checked {{
    background-color: {t.ACCENT_PRIMARY};
    border-color: {t.ACCENT_PRIMARY};
}}

/* ---- Sliders ---- */
QSlider::groove:horizontal {{
    border: none;
    background: {t.SURFACE_ELEVATED};
    height: 6px;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {t.ACCENT_PRIMARY};
    border-radius: 3px;
}}
QSlider::add-page:horizontal {{
    background: {t.SURFACE_ELEVATED};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {t.TEXT_PRIMARY};
    border: 2px solid {t.ACCENT_PRIMARY};
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 9px;
}}
QSlider::handle:horizontal:hover {{
    background: {t.ACCENT_PRIMARY_HOVER};
}}

/* ---- Tabs ---- */
QTabWidget::pane {{
    border: none;
    background: transparent;
}}
QTabBar {{
    qproperty-drawBase: 0;
    background: transparent;
}}
QTabBar::tab {{
    background: transparent;
    color: {t.TEXT_SECONDARY};
    padding: 8px 16px;
    margin-right: 4px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}}
QTabBar::tab:hover {{
    color: {t.TEXT_PRIMARY};
}}
QTabBar::tab:selected {{
    color: {t.ACCENT_PRIMARY};
    border-bottom: 2px solid {t.ACCENT_PRIMARY};
}}

/* ---- Progress bar ---- */
QProgressBar {{
    background-color: {t.SURFACE_ELEVATED};
    border: none;
    border-radius: 4px;
    text-align: center;
    color: {t.TEXT_SECONDARY};
    height: 8px;
    font-size: 10px;
}}
QProgressBar::chunk {{
    background-color: {t.ACCENT_PRIMARY};
    border-radius: 4px;
}}

/* ---- Status pills (custom QLabel role) ---- */
QLabel[role="pill"] {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER};
    border-radius: 12px;
    padding: 4px 10px;
    color: {t.TEXT_SECONDARY};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}
QLabel[role="pill"][tone="ok"] {{ color: {t.SUCCESS}; border-color: {t.SUCCESS}; }}
QLabel[role="pill"][tone="warn"] {{ color: {t.WARNING}; border-color: {t.WARNING}; }}
QLabel[role="pill"][tone="bad"] {{ color: {t.DANGER}; border-color: {t.DANGER}; }}
QLabel[role="pill"][tone="info"] {{ color: {t.ACCENT_SECONDARY}; border-color: {t.ACCENT_SECONDARY}; }}
QLabel[role="pill"][tone="primary"] {{ color: {t.ACCENT_PRIMARY}; border-color: {t.ACCENT_PRIMARY}; }}

/* ---- Pipeline-stage indicator dots ---- */
QLabel[role="stage"] {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER};
    border-radius: {t.RADIUS_MD}px;
    padding: 6px 12px;
    color: {t.TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
QLabel[role="stage"][active="true"] {{
    background-color: {t.ACCENT_PRIMARY};
    border-color: {t.ACCENT_PRIMARY_HOVER};
    color: white;
}}
QLabel[role="stage"][done="true"] {{
    background-color: transparent;
    border-color: {t.ACCENT_SECONDARY};
    color: {t.ACCENT_SECONDARY};
}}

/* ---- Scroll bars (slim, themed) ---- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {t.BORDER_BRIGHT};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: {t.TEXT_MUTED};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0; border: none;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {t.BORDER_BRIGHT};
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {t.TEXT_MUTED};
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0; border: none;
}}

/* ---- Status bar (bottom of main window) ---- */
QStatusBar {{
    background-color: {t.SURFACE};
    color: {t.TEXT_SECONDARY};
    border-top: 1px solid {t.BORDER};
}}
QStatusBar::item {{
    border: none;
}}

/* ---- Tooltips ---- */
QToolTip {{
    background-color: {t.SURFACE_ELEVATED};
    color: {t.TEXT_PRIMARY};
    border: 1px solid {t.BORDER_BRIGHT};
    padding: 6px 10px;
    border-radius: {t.RADIUS_SM}px;
}}

/* ---- Splitter handle ---- */
QSplitter::handle {{
    background-color: {t.BORDER};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QSplitter::handle:vertical {{
    height: 1px;
}}

/* ---- Menus ---- */
QMenuBar {{
    background-color: {t.SURFACE};
    color: {t.TEXT_PRIMARY};
}}
QMenuBar::item {{
    background: transparent;
    padding: 6px 10px;
}}
QMenuBar::item:selected {{
    background-color: {t.SURFACE_HOVER};
    border-radius: 4px;
}}
QMenu {{
    background-color: {t.SURFACE_ELEVATED};
    border: 1px solid {t.BORDER_BRIGHT};
    border-radius: {t.RADIUS_MD}px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 18px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {t.ACCENT_PRIMARY};
    color: white;
}}
"""


# =====================================================================
# Apply theme + helpers
# =====================================================================


def apply_theme(app: QApplication, tokens: Optional[Tokens] = None) -> None:
    """Apply the QBot3 dark theme to a running QApplication."""
    t = tokens or Tokens()

    # Core palette (so widgets without explicit QSS still pick up dark colours)
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(t.BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(t.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Base, QColor(t.SURFACE_ELEVATED))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(t.SURFACE))
    palette.setColor(QPalette.ColorRole.Text, QColor(t.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Button, QColor(t.SURFACE_ELEVATED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(t.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(t.ACCENT_PRIMARY))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("white"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(t.SURFACE_ELEVATED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(t.TEXT_PRIMARY))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(t.TEXT_MUTED))
    palette.setColor(QPalette.ColorRole.Link, QColor(t.ACCENT_PRIMARY))
    app.setPalette(palette)

    # Default app font — Qt picks the first available family from the stack
    font = QFont()
    font.setFamily("Inter")
    font.setPointSize(10)        # ≈ 13 px on 96-DPI displays
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    app.setStyleSheet(_build_qss(t))


def add_drop_shadow(widget: QWidget, *, radius: int = 24,
                    color: str = "#000000", alpha: int = 140,
                    x_offset: int = 0, y_offset: int = 6) -> None:
    """Apply a soft drop shadow to a widget — used on cards + dialogs."""
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(radius)
    c = QColor(color)
    c.setAlpha(alpha)
    eff.setColor(c)
    eff.setOffset(x_offset, y_offset)
    widget.setGraphicsEffect(eff)


# =====================================================================
# Animated mode button — pulsing accent halo when active
# =====================================================================


class ModeButton(QPushButton):
    """Pill-shaped mode button with a pulsing halo when active.

    The halo is a QGraphicsDropShadowEffect whose blur radius is animated
    via a QPropertyAnimation. Cheap on the GPU, looks premium.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setProperty("variant", "mode")
        self.setProperty("active", "false")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._halo = QGraphicsDropShadowEffect(self)
        self._halo.setColor(QColor(Tokens.ACCENT_PRIMARY))
        self._halo.setOffset(0, 0)
        self._halo.setBlurRadius(0)
        self.setGraphicsEffect(self._halo)

        self._anim = QPropertyAnimation(self._halo, b"blurRadius", self)
        self._anim.setDuration(1400)
        self._anim.setStartValue(8)
        self._anim.setEndValue(28)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._anim.setLoopCount(-1)
        self._anim_reverse = False

    def set_active(self, active: bool) -> None:
        self.setProperty("active", "true" if active else "false")
        self.style().unpolish(self)
        self.style().polish(self)
        if active:
            self._halo.setColor(QColor(Tokens.ACCENT_PRIMARY))
            self._anim.start()
        else:
            self._anim.stop()
            self._halo.setBlurRadius(0)


# =====================================================================
# StatusPill — small coloured-dot label, used in the status bar
# =====================================================================


class StatusPill(QObject):
    """Helper that toggles a QLabel's `tone` property to switch colours."""

    @staticmethod
    def set_tone(label, tone: str) -> None:
        """tone in {ok, warn, bad, info, primary, ''} — empty resets to neutral."""
        label.setProperty("role", "pill")
        label.setProperty("tone", tone or "")
        st = label.style()
        if st is not None:
            st.unpolish(label)
            st.polish(label)


# =====================================================================
# Loading spinner — used during YOLO load / VLM call
# =====================================================================


class Spinner(QWidget):
    """Light-weight animated spinner — eight rotating dots, accent colour."""

    def __init__(self, parent: Optional[QWidget] = None, *, size: int = 28,
                 color: str = Tokens.ACCENT_PRIMARY) -> None:
        super().__init__(parent)
        self._size = size
        self._color = QColor(color)
        self._angle = 0
        self.setFixedSize(size, size)
        self._anim = QPropertyAnimation(self, b"angle", self)
        self._anim.setDuration(1200)
        self._anim.setStartValue(0)
        self._anim.setEndValue(360)
        self._anim.setLoopCount(-1)

    def start(self) -> None:
        self._anim.start()

    def stop(self) -> None:
        self._anim.stop()

    def get_angle(self) -> int:
        return self._angle

    def set_angle(self, value: int) -> None:
        self._angle = value
        self.update()

    angle = pyqtProperty(int, fget=get_angle, fset=set_angle)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx = self.width() / 2
        cy = self.height() / 2
        radius = min(cx, cy) - 3
        dot_count = 8
        from math import cos, radians, sin
        for i in range(dot_count):
            theta = radians(self._angle + i * (360 / dot_count))
            x = cx + cos(theta) * radius
            y = cy + sin(theta) * radius
            alpha = int(60 + (255 - 60) * (i / (dot_count - 1)))
            c = QColor(self._color)
            c.setAlpha(alpha)
            p.setBrush(c)
            p.setPen(QPen(Qt.PenStyle.NoPen))
            p.drawEllipse(int(x - 2), int(y - 2), 4, 4)
        p.end()
