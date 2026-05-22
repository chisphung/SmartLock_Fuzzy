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
            print(f"[Fuzzy Fallback DEBUG] Evaluating Inputs:")
            print(f"  - confidence   : {confidence:.2f}")
            print(f"  - illumination : {illumination:.2f}")
            print(f"  - facial_angle : {facial_angle:.2f}")
            risk = 1.0
            rule_fired = "Default (LOW confidence / No Match)"
            if confidence < 30:
                risk = 0.9
                rule_fired = "Low confidence"
            elif confidence < 47:
                if 60 <= illumination <= 195 and facial_angle <= 35:
                    risk = 0.75
                    rule_fired = "Medium-Low confidence + Normal light + Frontal angle"
                else:
                    risk = 0.9
                    rule_fired = "Medium-Low confidence + Suboptimal light/angle"
            elif confidence < 66:
                if 60 <= illumination <= 195 and facial_angle <= 35:
                    risk = 0.5
                    rule_fired = "Medium confidence + Normal light + Frontal angle"
                else:
                    risk = 0.75
                    rule_fired = "Medium confidence + Suboptimal light/angle"
            else:  # confidence >= 66
                if 60 <= illumination <= 195 and facial_angle <= 35:
                    risk = 0.2
                    rule_fired = "High confidence + Normal light + Frontal angle"
                else:
                    risk = 0.5
                    rule_fired = "High confidence + Suboptimal light/angle"

            if risk < 0.3:
                action, details = "unlock", "Actuate Solenoid (Unlock)"
            elif risk < 0.6:
                action, details = "otp", "Request 2nd Factor (OTP)"
            elif risk < 0.85:
                action, details = "deny", "Deny Access & Log Event"
            else:
                action, details = "lockout", "Immediate Lockout & Alert"

            print(f"[Fuzzy Fallback DEBUG] Result:")
            print(f"  - Rule Fired   : {rule_fired}")
            print(f"  - security_risk: {risk:.4f}")
            print(f"  - action       : {action} ({details})")

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
