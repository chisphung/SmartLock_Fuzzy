"""
hardware_io.py – Interrupt-driven keypad 4x4 & servo controller for SmartLock.

Hardware wiring (Raspberry Pi GPIO BCM):
    Keypad 4x4:
        Row pins  : GPIO 17, 27, 22, 5   (output, driven HIGH for scanning)
        Col pins  : GPIO 6, 13, 19, 26   (input, PULL_DOWN, interrupt RISING)
    Servo:
        PWM pin   : GPIO 18              (hardware PWM channel 0)

Keypad processing uses GPIO interrupts (NOT polling).
Software debounce (250 ms) is applied on top of RPi.GPIO bouncetime.
Password is 6 digits, stored as a SHA-256 hash.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Callable

import RPi.GPIO as GPIO

logger = logging.getLogger("HardwareIO")

_KEYMAP: list[list[str]] = [
    ["1", "2", "3", "A"],
    ["4", "5", "6", "B"],
    ["7", "8", "9", "C"],
    ["*", "0", "#", "D"],
]

_DEFAULT_ROW_PINS = [17, 27, 22, 5]
_DEFAULT_COL_PINS = [6, 13, 19, 26]
_DEFAULT_SERVO_PIN = 18

_PASSWORD_LENGTH = 6
_DEFAULT_PASSWORD = "123456"
_DEBOUNCE_MS = 250
_DEBOUNCE_THRESHOLD = 0.25
_INPUT_TIMEOUT = 10.0
_SERVO_FREQ = 50
_SERVO_UNLOCK_DUTY = 7.5
_SERVO_LOCK_DUTY = 2.5
_UNLOCK_DURATION = 5.0


def _hash_pin(pin: str) -> str:
    """Return SHA-256 hex digest of a PIN string."""
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


class SmartLockHardware:
    """
    Interrupt-driven keypad scanner + servo door lock controller.

    Usage::

        hw = SmartLockHardware()
        hw.on_unlock = lambda src, ts: print(f"Unlocked by {src}")
        hw.on_keypad_event = lambda evt: print(evt)
        hw.start()
        ...
        hw.stop()
    """

    def __init__(
        self,
        default_password: str = _DEFAULT_PASSWORD,
        unlock_duration: float = _UNLOCK_DURATION,
        row_pins: list[int] | None = None,
        col_pins: list[int] | None = None,
        servo_pin: int = _DEFAULT_SERVO_PIN,
    ) -> None:
        self.ROW_PINS = row_pins or list(_DEFAULT_ROW_PINS)
        self.COL_PINS = col_pins or list(_DEFAULT_COL_PINS)
        self.SERVO_PIN = servo_pin

        self._password_hash = _hash_pin(default_password)

        self.unlock_duration = unlock_duration
        self._servo_pwm: GPIO.PWM | None = None
        self._door_lock = threading.Lock()

        self._key_buffer: list[str] = []
        self._last_key_time: float = 0.0
        self._last_interrupt_time: float = 0.0
        self._keypad_lock = threading.Lock()

        self._running = False

        self.on_unlock: Callable[[str, float], None] | None = None
        self.on_keypad_event: Callable[[dict], None] | None = None

        self._last_event: dict | None = None
        self._event_lock = threading.Lock()

        self._failed_attempts = 0
        self._lockout_until: float = 0.0
        self._MAX_FAILED = 5
        self._LOCKOUT_SECONDS = 30.0

    def start(self) -> None:
        """Configure GPIO pins and attach column interrupts."""
        if self._running:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        for pin in self.ROW_PINS:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.HIGH)

        for pin in self.COL_PINS:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        GPIO.setup(self.SERVO_PIN, GPIO.OUT)
        self._servo_pwm = GPIO.PWM(self.SERVO_PIN, _SERVO_FREQ)
        self._servo_pwm.start(_SERVO_LOCK_DUTY)
        time.sleep(0.3)
        self._servo_pwm.ChangeDutyCycle(0)

        for col_pin in self.COL_PINS:
            GPIO.add_event_detect(
                col_pin,
                GPIO.RISING,
                callback=self._col_interrupt,
                bouncetime=_DEBOUNCE_MS,
            )

        self._running = True
        logger.info("[Hardware] GPIO initialized – keypad interrupts active.")

    def stop(self) -> None:
        """Remove interrupts, stop PWM, clean up GPIO."""
        self._running = False

        if self._servo_pwm:
            self._servo_pwm.stop()
            self._servo_pwm = None

        for col in self.COL_PINS:
            try:
                GPIO.remove_event_detect(col)
            except Exception:
                pass

        GPIO.cleanup()
        logger.info("[Hardware] GPIO cleaned up.")

    def _col_interrupt(self, col_pin: int) -> None:
        """ISR callback – RISING edge on a column pin."""
        if not self._running:
            return

        now = time.time()

        if now - self._last_interrupt_time < _DEBOUNCE_THRESHOLD:
            return

        with self._keypad_lock:
            pressed_key = self._scan_key(col_pin)
            if pressed_key is None:
                return

            self._last_interrupt_time = now
            logger.info(f"[Keypad] Key pressed: '{pressed_key}'")

            if (
                self._key_buffer
                and now - self._last_key_time > _INPUT_TIMEOUT
            ):
                logger.info("[Keypad] Input timeout – buffer cleared.")
                self._key_buffer.clear()

            self._last_key_time = now

            if pressed_key == "*":
                self._key_buffer.clear()
                self._emit_event("buffer_cleared", "Buffer cleared by user")
                return

            if pressed_key == "#":
                self._submit_password()
                return

            if not pressed_key.isdigit():
                self._emit_event(
                    "key_ignored",
                    f"Non-digit key '{pressed_key}' ignored",
                )
                return

            self._key_buffer.append(pressed_key)
            remaining = _PASSWORD_LENGTH - len(self._key_buffer)
            self._emit_event(
                "digit_entered",
                f"Digit entered ({len(self._key_buffer)}/{_PASSWORD_LENGTH})",
                extra={"buffer_length": len(self._key_buffer), "remaining": remaining},
            )

            if len(self._key_buffer) >= _PASSWORD_LENGTH:
                self._submit_password()

    def _scan_key(self, col_pin: int) -> str | None:
        """Pull rows LOW one at a time to identify the pressed key."""
        col_idx = None
        for idx, pin in enumerate(self.COL_PINS):
            if pin == col_pin:
                col_idx = idx
                break
        if col_idx is None:
            return None

        for row_pin in self.ROW_PINS:
            GPIO.output(row_pin, GPIO.LOW)

        pressed_key = None
        for row_idx, row_pin in enumerate(self.ROW_PINS):
            GPIO.output(row_pin, GPIO.HIGH)
            time.sleep(0.005)
            if GPIO.input(col_pin):
                pressed_key = _KEYMAP[row_idx][col_idx]
                GPIO.output(row_pin, GPIO.LOW)
                break
            GPIO.output(row_pin, GPIO.LOW)

        for row_pin in self.ROW_PINS:
            GPIO.output(row_pin, GPIO.HIGH)

        return pressed_key

    def _submit_password(self) -> None:
        """Validate the buffered PIN and trigger unlock or deny."""
        entered = "".join(self._key_buffer)
        self._key_buffer.clear()

        now = time.time()

        if now < self._lockout_until:
            remaining = int(self._lockout_until - now)
            self._emit_event(
                "lockout",
                f"Keypad locked out. Try again in {remaining}s.",
                extra={"lockout_remaining": remaining},
            )
            logger.warning(f"[Keypad] Lockout active – {remaining}s remaining.")
            return

        if len(entered) != _PASSWORD_LENGTH:
            self._emit_event(
                "password_invalid",
                f"Password must be {_PASSWORD_LENGTH} digits (got {len(entered)})",
            )
            return

        entered_hash = _hash_pin(entered)

        if entered_hash == self._password_hash:
            self._failed_attempts = 0
            logger.info("[Keypad] Correct password – unlocking door.")
            self._emit_event("password_correct", "Correct password – door unlocking")
            threading.Thread(
                target=self.unlock_door,
                args=("keypad",),
                daemon=True,
            ).start()
        else:
            self._failed_attempts += 1
            remaining_attempts = self._MAX_FAILED - self._failed_attempts
            logger.warning(
                f"[Keypad] Wrong password "
                f"({self._failed_attempts}/{self._MAX_FAILED})"
            )

            if self._failed_attempts >= self._MAX_FAILED:
                self._lockout_until = now + self._LOCKOUT_SECONDS
                self._failed_attempts = 0
                self._emit_event(
                    "lockout",
                    f"Too many failed attempts – locked for {int(self._LOCKOUT_SECONDS)}s",
                    extra={"lockout_seconds": self._LOCKOUT_SECONDS},
                )
            else:
                self._emit_event(
                    "password_wrong",
                    f"Wrong password ({remaining_attempts} attempts left)",
                    extra={
                        "failed_attempts": self._failed_attempts,
                        "remaining_attempts": remaining_attempts,
                    },
                )

    def unlock_door(self, source: str = "unknown") -> None:
        """Actuate the servo to unlock, wait, then re-lock."""
        if not self._door_lock.acquire(blocking=False):
            logger.info("[Hardware] Door already unlocking – skipped.")
            return

        try:
            logger.info(f"[Hardware] Unlocking door (source: {source})")

            if self._servo_pwm:
                self._servo_pwm.ChangeDutyCycle(_SERVO_UNLOCK_DUTY)
                time.sleep(0.5)
                self._servo_pwm.ChangeDutyCycle(0)

            unlock_time = time.time()

            if self.on_unlock:
                try:
                    self.on_unlock(source, unlock_time)
                except Exception as exc:
                    logger.error(f"[Hardware] on_unlock callback error: {exc}")

            time.sleep(self.unlock_duration)

            logger.info("[Hardware] Locking door.")
            if self._servo_pwm:
                self._servo_pwm.ChangeDutyCycle(_SERVO_LOCK_DUTY)
                time.sleep(0.5)
                self._servo_pwm.ChangeDutyCycle(0)

        finally:
            self._door_lock.release()

    def set_password(self, current_pin: str, new_pin: str) -> dict:
        """Change the door password."""
        if len(new_pin) != _PASSWORD_LENGTH or not new_pin.isdigit():
            return {
                "success": False,
                "message": f"New password must be exactly {_PASSWORD_LENGTH} digits.",
            }

        if _hash_pin(current_pin) != self._password_hash:
            return {"success": False, "message": "Current password is incorrect."}

        self._password_hash = _hash_pin(new_pin)
        logger.info("[Hardware] Password changed successfully.")
        return {"success": True, "message": "Password updated."}

    @property
    def is_keypad_active(self) -> bool:
        """True if the user is currently entering digits on the keypad."""
        with self._keypad_lock:
            if not self._key_buffer:
                return False
            if time.time() - self._last_key_time > _INPUT_TIMEOUT:
                self._key_buffer.clear()
                return False
            return True

    def status(self) -> dict:
        """Return current hardware status for API responses."""
        with self._keypad_lock:
            buffer_len = len(self._key_buffer)

        now = time.time()
        lockout_remaining = max(0, self._lockout_until - now)

        with self._event_lock:
            last_event = dict(self._last_event) if self._last_event else None

        return {
            "running": self._running,
            "keypad_active": self.is_keypad_active,
            "buffer_length": buffer_len,
            "password_length": _PASSWORD_LENGTH,
            "failed_attempts": self._failed_attempts,
            "lockout_remaining": round(lockout_remaining, 1),
            "last_event": last_event,
        }



    def _emit_event(
        self,
        event_type: str,
        message: str,
        extra: dict | None = None,
    ) -> None:
        """Store the latest event and invoke the callback if set."""
        event = {
            "type": event_type,
            "message": message,
            "timestamp": time.time(),
            **(extra or {}),
        }

        with self._event_lock:
            self._last_event = event

        if self.on_keypad_event:
            try:
                self.on_keypad_event(event)
            except Exception as exc:
                logger.error(f"[Hardware] on_keypad_event callback error: {exc}")

    def __repr__(self) -> str:
        return (
            f"SmartLockHardware(running={self._running}, "
            f"servo_pin={self.SERVO_PIN}, "
            f"rows={self.ROW_PINS}, cols={self.COL_PINS})"
        )
