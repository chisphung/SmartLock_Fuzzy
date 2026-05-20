"""
oled_display.py – SSD1306 OLED SPI display controller for SmartLock.

Hardware: SSD1306 128x64 OLED via SPI.
Library:  luma.oled + Pillow for rendering.

Install:
    pip install luma.oled
"""

from __future__ import annotations

import logging
import threading
import time

from PIL import Image, ImageDraw, ImageFont
from luma.core.interface.serial import spi
from luma.oled.device import ssd1306

logger = logging.getLogger("OLEDDisplay")

_WIDTH = 128
_HEIGHT = 64
_SPI_PORT = 0
_SPI_DEVICE = 0
_GPIO_DC = 24
_GPIO_RST = 25
_DISPLAY_TIMEOUT = 30.0


class OLEDDisplay:
    """SSD1306 128x64 OLED display driver for SmartLock."""

    def __init__(
        self,
        spi_port: int = _SPI_PORT,
        spi_device: int = _SPI_DEVICE,
        gpio_dc: int = _GPIO_DC,
        gpio_rst: int = _GPIO_RST,
    ) -> None:
        self._device: ssd1306 | None = None
        self._spi_port = spi_port
        self._spi_device = spi_device
        self._gpio_dc = gpio_dc
        self._gpio_rst = gpio_rst
        self._lock = threading.Lock()
        self._running = False

        try:
            self._font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            self._font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            self._font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except OSError:
            self._font_large = ImageFont.load_default()
            self._font_medium = ImageFont.load_default()
            self._font_small = ImageFont.load_default()

    def start(self) -> None:
        """Initialize the OLED device."""
        if self._running:
            return
        try:
            serial = spi(
                port=self._spi_port,
                device=self._spi_device,
                gpio_DC=self._gpio_dc,
                gpio_RST=self._gpio_rst,
            )
            self._device = ssd1306(serial, width=_WIDTH, height=_HEIGHT)
            self._running = True
            logger.info("[OLED] Display initialized.")
            self.show_idle()
        except Exception as exc:
            logger.error(f"[OLED] Failed to initialize: {exc}")
            self._device = None

    def stop(self) -> None:
        """Turn off the display and clean up."""
        self._running = False
        if self._device:
            try:
                self._device.hide()
            except Exception:
                pass
            self._device = None
        logger.info("[OLED] Display stopped.")

    def _render(self, draw_fn) -> None:
        """Create image, call draw_fn(draw, width, height), push to device."""
        if not self._device:
            return
        with self._lock:
            try:
                img = Image.new("1", (_WIDTH, _HEIGHT), 0)
                draw = ImageDraw.Draw(img)
                draw_fn(draw, _WIDTH, _HEIGHT)
                self._device.display(img)
            except Exception as exc:
                logger.error(f"[OLED] Render error: {exc}")

    def show_idle(self) -> None:
        """Idle screen: SmartLock ready."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 12), "SmartLock", fill=1, font=self._font_large, anchor="mt")
            d.line([(10, 32), (w - 10, 32)], fill=1)
            d.text((w // 2, 42), "Ready", fill=1, font=self._font_medium, anchor="mt")
        self._render(draw)

    def show_enter_pin(self, digits_entered: int, total: int = 6) -> None:
        """Show PIN entry progress: filled dots and empty dots."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 8), "Enter PIN", fill=1, font=self._font_large, anchor="mt")

            dot_size = 8
            gap = 6
            total_width = total * dot_size + (total - 1) * gap
            start_x = (w - total_width) // 2
            y = 36

            for i in range(total):
                x = start_x + i * (dot_size + gap)
                if i < digits_entered:
                    d.ellipse([x, y, x + dot_size, y + dot_size], fill=1)
                else:
                    d.ellipse([x, y, x + dot_size, y + dot_size], outline=1)

            remaining = total - digits_entered
            if remaining > 0:
                d.text((w // 2, 54), f"{remaining} more", fill=1, font=self._font_small, anchor="mt")
        self._render(draw)

    def show_access_granted(self, source: str = "keypad") -> None:
        """Access granted screen."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 8), "ACCESS", fill=1, font=self._font_large, anchor="mt")
            d.text((w // 2, 28), "GRANTED", fill=1, font=self._font_large, anchor="mt")
            d.line([(10, 48), (w - 10, 48)], fill=1)
            label = "PIN" if source == "keypad" else "Face"
            d.text((w // 2, 54), f"via {label}", fill=1, font=self._font_small, anchor="mt")
        self._render(draw)

    def show_access_denied(self, message: str = "Wrong PIN") -> None:
        """Access denied screen."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 8), "ACCESS", fill=1, font=self._font_large, anchor="mt")
            d.text((w // 2, 28), "DENIED", fill=1, font=self._font_large, anchor="mt")
            d.line([(10, 48), (w - 10, 48)], fill=1)
            d.text((w // 2, 54), message, fill=1, font=self._font_small, anchor="mt")
        self._render(draw)

    def show_lockout(self, seconds_remaining: int) -> None:
        """Lockout warning screen."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 8), "LOCKED", fill=1, font=self._font_large, anchor="mt")
            d.line([(10, 28), (w - 10, 28)], fill=1)
            d.text((w // 2, 36), "Too many attempts", fill=1, font=self._font_small, anchor="mt")
            d.text((w // 2, 52), f"Wait {seconds_remaining}s", fill=1, font=self._font_medium, anchor="mt")
        self._render(draw)

    def show_door_open(self, seconds_remaining: int = 5) -> None:
        """Door open countdown screen."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 8), "DOOR OPEN", fill=1, font=self._font_large, anchor="mt")
            d.line([(10, 28), (w - 10, 28)], fill=1)
            d.text((w // 2, 40), f"Closing in {seconds_remaining}s", fill=1, font=self._font_medium, anchor="mt")
        self._render(draw)

    def show_face_detected(self, name: str, action: str, risk: float) -> None:
        """Face recognition result screen."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 6), name, fill=1, font=self._font_large, anchor="mt")
            d.line([(10, 24), (w - 10, 24)], fill=1)
            d.text((w // 2, 30), f"Action: {action.upper()}", fill=1, font=self._font_medium, anchor="mt")
            d.text((w // 2, 48), f"Risk: {risk:.2f}", fill=1, font=self._font_small, anchor="mt")
        self._render(draw)

    def show_message(self, title: str, subtitle: str = "") -> None:
        """Generic two-line message screen."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 18), title, fill=1, font=self._font_large, anchor="mt")
            if subtitle:
                d.line([(10, 36), (w - 10, 36)], fill=1)
                d.text((w // 2, 44), subtitle, fill=1, font=self._font_medium, anchor="mt")
        self._render(draw)

    def show_registration(self, name: str, accepted: int, required: int) -> None:
        """Face registration progress screen."""
        def draw(d: ImageDraw.Draw, w: int, h: int):
            d.text((w // 2, 6), "Registering", fill=1, font=self._font_large, anchor="mt")
            d.text((w // 2, 26), name, fill=1, font=self._font_medium, anchor="mt")
            d.line([(10, 40), (w - 10, 40)], fill=1)

            bar_x = 10
            bar_w = w - 20
            bar_y = 46
            bar_h = 10
            d.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], outline=1)
            fill_w = int(bar_w * accepted / required) if required > 0 else 0
            if fill_w > 0:
                d.rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], fill=1)

            d.text((w // 2, 60), f"{accepted}/{required}", fill=1, font=self._font_small, anchor="mt")
        self._render(draw)

    def clear(self) -> None:
        """Clear the display."""
        if self._device:
            with self._lock:
                try:
                    self._device.clear()
                except Exception:
                    pass
