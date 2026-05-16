"""
fuzzy_logic.py - Smart-lock fuzzy decision wrapper.
"""

from __future__ import annotations


try:
    from infra.fuzzy_controller import FuzzySecurityController as _FuzzyController
except Exception as exc:  # pragma: no cover - fallback for missing pyfuzzylite
    print(f"[Fuzzy] pyfuzzylite unavailable, using fallback decision logic: {exc}")

    class _FuzzyController:
        def evaluate(
            self,
            confidence: float,
            illumination: float,
            facial_angle: float,
        ) -> dict:
            risk = 1.0
            if confidence >= 75 and 60 <= illumination <= 195 and facial_angle <= 35:
                risk = 0.2
            elif confidence >= 55 and 35 <= illumination <= 230 and facial_angle <= 45:
                risk = 0.55
            elif confidence >= 35:
                risk = 0.8

            if risk < 0.3:
                action, details = "unlock", "Actuate Solenoid (Unlock)"
            elif risk < 0.6:
                action, details = "otp", "Request 2nd Factor (OTP)"
            elif risk < 0.85:
                action, details = "deny", "Deny Access & Log Event"
            else:
                action, details = "lockout", "Immediate Lockout & Alert"

            return {
                "security_risk": round(risk, 4),
                "action": action,
                "details": details,
                "inputs": {
                    "model_confidence": round(confidence, 2),
                    "illumination": round(illumination, 2),
                    "facial_angle": round(facial_angle, 2),
                },
            }


class SmartLockFuzzyDecision:
    def __init__(self) -> None:
        self._controller = _FuzzyController()

    def evaluate_detection(self, detection: dict | None) -> dict | None:
        if not detection:
            return {
                "security_risk": 1.0,
                "action": "deny",
                "details": "No face detected",
                "inputs": {},
            }

        if detection.get("name") in (None, "Unknown"):
            model_confidence = 0.0
        else:
            distance = float(detection.get("confidence", 100.0))
            model_confidence = max(0.0, min(100.0, 100.0 - distance))

        return self._controller.evaluate(
            confidence=model_confidence,
            illumination=float(detection.get("illumination", 128.0)),
            facial_angle=float(detection.get("facial_angle", 0.0)),
        )
