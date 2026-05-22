"""
fuzzy_controller.py – Mamdani fuzzy logic controller for Smart Lock security.

Uses pyfuzzylite (https://github.com/fuzzylite/pyfuzzylite) to evaluate
three antecedent inputs and produce a security-risk score that maps to
an actuator decision.

Antecedents:
  1. Model Confidence (C)  – 0..100  (inverted LBPH distance)
  2. Illumination (I)      – 0..255  (mean brightness of face ROI)
  3. Facial Angle (θ)      – 0..90   (estimated yaw: 0 = frontal)

Consequent:
  Security Risk – 0..1  → mapped to one of four actions.
"""

from __future__ import annotations

import fuzzylite as fl


# ── Action thresholds (crisp output → decision) ─────────────────────────────

_ACTION_MAP = [
    (0.30, "unlock",  "Actuate Solenoid (Unlock)"),
    (0.60, "otp",     "Request 2nd Factor (OTP)"),
    (0.85, "deny",    "Deny Access & Log Event"),
    (1.01, "lockout", "Immediate Lockout & Alert"),
]


def _risk_to_action(risk: float) -> tuple[str, str]:
    """Map a crisp security-risk value to (action_code, description)."""
    for threshold, code, desc in _ACTION_MAP:
        if risk < threshold:
            return code, desc
    return "lockout", "Immediate Lockout & Alert"


# ── Fuzzy Engine ─────────────────────────────────────────────────────────────

class FuzzySecurityController:
    """
    Mamdani fuzzy controller for Smart Lock security-risk assessment.

    Usage::

        ctrl = FuzzySecurityController()
        result = ctrl.evaluate(confidence=85.0, illumination=130.0,
                               facial_angle=5.0)
        print(result)
        # {"security_risk": 0.12, "action": "unlock",
        #  "details": "Actuate Solenoid (Unlock)"}
    """

    def __init__(self) -> None:
        self._engine = self._build_engine()
        # cache variable references for fast access
        self._v_confidence   = self._engine.input_variable("model_confidence")
        self._v_illumination = self._engine.input_variable("illumination")
        self._v_angle        = self._engine.input_variable("facial_angle")
        self._v_risk         = self._engine.output_variable("security_risk")

    # ── engine construction ──────────────────────────────────────────────

    @staticmethod
    def _build_engine() -> fl.Engine:
        """Construct and return a fully-wired Mamdani fuzzy engine."""

        engine = fl.Engine(
            name="SmartLockSecurity",
            input_variables=[
                # ── Antecedent 1: Model Confidence ──────────────────
                # Inverted LBPH distance: 0 = no confidence, 100 = perfect
                fl.InputVariable(
                    name="model_confidence",
                    minimum=0.0,
                    maximum=85.0,
                    lock_range=True,
                    terms=[
                        fl.Gaussian("LOW", 0.0, 17.0),
                        fl.Gaussian("MEDIUM", 42.5, 10.0),
                        fl.Gaussian("HIGH", 85.0, 17.0),
                    ],
                ),
                # ── Antecedent 2: Illumination ──────────────────────
                # Mean brightness of face ROI (grayscale 0–255)
                fl.InputVariable(
                    name="illumination",
                    minimum=0.0,
                    maximum=255.0,
                    lock_range=True,
                    terms=[
                        fl.Gaussian("DARK", 0.0, 40.0),
                        fl.Gaussian("NORMAL", 128.0, 45.0),
                        fl.Gaussian("BRIGHT", 255.0, 40.0),
                    ],
                ),
                # ── Antecedent 3: Facial Angle ──────────────────────
                # Estimated yaw in degrees: 0 = frontal, 90 = profile
                fl.InputVariable(
                    name="facial_angle",
                    minimum=0.0,
                    maximum=90.0,
                    lock_range=True,
                    terms=[
                        fl.Gaussian("FRONTAL", 0.0, 20.0),
                        fl.Gaussian("MARGINAL", 90.0, 40.0),
                    ],
                ),
            ],
            output_variables=[
                # ── Consequent: Security Risk ───────────────────────
                fl.OutputVariable(
                    name="security_risk",
                    minimum=0.0,
                    maximum=1.0,
                    lock_range=True,
                    lock_previous=False,
                    default_value=1.0,          # fail-safe: max risk
                    aggregation=fl.Maximum(),
                    defuzzifier=fl.Centroid(resolution=200),
                    terms=[
                        fl.Gaussian("MINIMUM", 0.0, 0.1),
                        fl.Gaussian("AVERAGE", 0.5, 0.1),
                        fl.Gaussian("MAXIMUM", 1.0, 0.08),
                    ],
                ),
            ],
            rule_blocks=[
                fl.RuleBlock(
                    name="security_rules",
                    conjunction=fl.Minimum(),
                    disjunction=fl.Maximum(),
                    implication=fl.Minimum(),
                    activation=fl.General(),
                    rules=[
                        # LOW confidence -> MAXIMUM risk
                        fl.Rule.create(
                            "if model_confidence is LOW "
                            "then security_risk is MAXIMUM"
                        ),
                        # HIGH confidence rules
                        fl.Rule.create(
                            "if model_confidence is HIGH "
                            "and illumination is NORMAL "
                            "and facial_angle is FRONTAL "
                            "then security_risk is MINIMUM"
                        ),
                        fl.Rule.create(
                            "if model_confidence is HIGH "
                            "and illumination is NORMAL "
                            "and facial_angle is MARGINAL "
                            "then security_risk is AVERAGE"
                        ),
                        fl.Rule.create(
                            "if model_confidence is HIGH "
                            "and illumination is DARK "
                            "and facial_angle is FRONTAL "
                            "then security_risk is AVERAGE"
                        ),
                        fl.Rule.create(
                            "if model_confidence is HIGH "
                            "and illumination is DARK "
                            "and facial_angle is MARGINAL "
                            "then security_risk is AVERAGE"
                        ),
                        fl.Rule.create(
                            "if model_confidence is HIGH "
                            "and illumination is BRIGHT "
                            "and facial_angle is FRONTAL "
                            "then security_risk is AVERAGE"
                        ),
                        fl.Rule.create(
                            "if model_confidence is HIGH "
                            "and illumination is BRIGHT "
                            "and facial_angle is MARGINAL "
                            "then security_risk is AVERAGE"
                        ),
                        # MEDIUM confidence rules
                        fl.Rule.create(
                            "if model_confidence is MEDIUM "
                            "and illumination is NORMAL "
                            "and facial_angle is FRONTAL "
                            "then security_risk is AVERAGE"
                        ),
                        fl.Rule.create(
                            "if model_confidence is MEDIUM "
                            "and illumination is NORMAL "
                            "and facial_angle is MARGINAL "
                            "then security_risk is MAXIMUM"
                        ),
                        fl.Rule.create(
                            "if model_confidence is MEDIUM "
                            "and illumination is DARK "
                            "and facial_angle is FRONTAL "
                            "then security_risk is MAXIMUM"
                        ),
                        fl.Rule.create(
                            "if model_confidence is MEDIUM "
                            "and illumination is DARK "
                            "and facial_angle is MARGINAL "
                            "then security_risk is MAXIMUM"
                        ),
                        fl.Rule.create(
                            "if model_confidence is MEDIUM "
                            "and illumination is BRIGHT "
                            "and facial_angle is FRONTAL "
                            "then security_risk is AVERAGE"
                        ),
                        fl.Rule.create(
                            "if model_confidence is MEDIUM "
                            "and illumination is BRIGHT "
                            "and facial_angle is MARGINAL "
                            "then security_risk is AVERAGE"
                        ),
                    ],
                ),
            ],
        )

        return engine

    # ── public API ───────────────────────────────────────────────────────

    def evaluate(
        self,
        confidence: float,
        illumination: float,
        facial_angle: float,
    ) -> dict:
        """
        Run fuzzy inference and return the security decision.

        Parameters
        ----------
        confidence : float
            Inverted LBPH distance (0–100).  Higher = better match.
        illumination : float
            Mean brightness of the face ROI (0–255).
        facial_angle : float
            Estimated yaw angle in degrees (0 = frontal, 90 = profile).

        Returns
        -------
        dict with keys:
            security_risk  – defuzzified crisp value (0.0–1.0)
            action         – one of "unlock", "otp", "deny", "lockout"
            details        – human-readable description of the action
            inputs         – dict of the three input values used
        """
        print(f"[Fuzzy DEBUG] Evaluating Inputs:")
        print(f"  - confidence   : {confidence:.2f}")
        for term in self._v_confidence.terms:
            try:
                print(f"    * membership {term.name:<8}: {term.membership(confidence):.4f}")
            except Exception as e:
                print(f"    * membership {term.name:<8}: error ({e})")
        print(f"  - illumination : {illumination:.2f}")
        for term in self._v_illumination.terms:
            try:
                print(f"    * membership {term.name:<8}: {term.membership(illumination):.4f}")
            except Exception as e:
                print(f"    * membership {term.name:<8}: error ({e})")
        print(f"  - facial_angle : {facial_angle:.2f}")
        for term in self._v_angle.terms:
            try:
                print(f"    * membership {term.name:<8}: {term.membership(facial_angle):.4f}")
            except Exception as e:
                print(f"    * membership {term.name:<8}: error ({e})")

        self._v_confidence.value   = float(confidence)
        self._v_illumination.value = float(illumination)
        self._v_angle.value        = float(facial_angle)

        self._engine.process()

        risk = float(self._v_risk.value)
        action, details = _risk_to_action(risk)

        print(f"[Fuzzy DEBUG] Inference Output:")
        print(f"  - security_risk: {risk:.4f}")
        for term in self._v_risk.terms:
            try:
                print(f"    * membership {term.name:<8}: {term.membership(risk):.4f}")
            except Exception as e:
                print(f"    * membership {term.name:<8}: error ({e})")
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

    def __repr__(self) -> str:
        return f"FuzzySecurityController(engine={self._engine.name!r})"
