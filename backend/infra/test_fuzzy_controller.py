"""
test_fuzzy_controller.py – Validate each rule in the fuzzy security engine.
"""

from fuzzy_controller import FuzzySecurityController


def _banner(label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")


def test_rule1_unlock():
    """R1: HIGH confidence + NORMAL illumination + FRONTAL → MINIMUM risk → unlock"""
    ctrl = FuzzySecurityController()
    r = ctrl.evaluate(confidence=90.0, illumination=128.0, facial_angle=5.0)
    _banner("R1: HIGH / NORMAL / FRONTAL → unlock")
    print(f"  Inputs : {r['inputs']}")
    print(f"  Risk   : {r['security_risk']}")
    print(f"  Action : {r['action']} — {r['details']}")
    assert r["action"] == "unlock", f"Expected 'unlock', got '{r['action']}'"
    assert r["security_risk"] < 0.30
    print("  ✓ PASS")


def test_rule2_otp():
    """R2: HIGH confidence + DARK illumination + MARGINAL angle → AVERAGE risk → otp"""
    ctrl = FuzzySecurityController()
    r = ctrl.evaluate(confidence=85.0, illumination=30.0, facial_angle=50.0)
    _banner("R2: HIGH / DARK / MARGINAL → otp")
    print(f"  Inputs : {r['inputs']}")
    print(f"  Risk   : {r['security_risk']}")
    print(f"  Action : {r['action']} — {r['details']}")
    assert r["action"] == "otp", f"Expected 'otp', got '{r['action']}'"
    assert 0.30 <= r["security_risk"] < 0.60
    print("  ✓ PASS")


def test_rule3_otp():
    """R3: MEDIUM confidence + BRIGHT illumination + MARGINAL → AVERAGE risk → otp"""
    ctrl = FuzzySecurityController()
    r = ctrl.evaluate(confidence=50.0, illumination=230.0, facial_angle=50.0)
    _banner("R3: MEDIUM / BRIGHT / MARGINAL → otp")
    print(f"  Inputs : {r['inputs']}")
    print(f"  Risk   : {r['security_risk']}")
    print(f"  Action : {r['action']} — {r['details']}")
    assert r["action"] == "otp", f"Expected 'otp', got '{r['action']}'"
    assert 0.30 <= r["security_risk"] < 0.60
    print("  ✓ PASS")


def test_rule4_deny():
    """R4: MEDIUM confidence + DARK + FRONTAL → MAXIMUM risk → deny/lockout"""
    ctrl = FuzzySecurityController()
    r = ctrl.evaluate(confidence=50.0, illumination=25.0, facial_angle=10.0)
    _banner("R4: MEDIUM / DARK / FRONTAL → deny")
    print(f"  Inputs : {r['inputs']}")
    print(f"  Risk   : {r['security_risk']}")
    print(f"  Action : {r['action']} — {r['details']}")
    assert r["action"] in ("deny", "lockout"), f"Expected 'deny' or 'lockout', got '{r['action']}'"
    assert r["security_risk"] >= 0.60
    print("  ✓ PASS")


def test_rule5_lockout():
    """R5: LOW confidence (any illumination, any angle) → MAXIMUM risk → lockout"""
    ctrl = FuzzySecurityController()

    # Test with various illumination and angles
    test_cases = [
        (10.0, 128.0, 10.0, "normal/frontal"),
        (5.0,  30.0,  60.0, "dark/marginal"),
        (15.0, 220.0, 45.0, "bright/marginal"),
    ]

    _banner("R5: LOW / ANY / ANY → lockout")
    for conf, illum, angle, label in test_cases:
        r = ctrl.evaluate(confidence=conf, illumination=illum, facial_angle=angle)
        print(f"  [{label}]  risk={r['security_risk']:.4f}  action={r['action']}")
        assert r["action"] in ("deny", "lockout"), \
            f"[{label}] Expected 'deny'/'lockout', got '{r['action']}'"
        assert r["security_risk"] >= 0.60
    print("  ✓ PASS (all sub-cases)")


def test_edge_cases():
    """Boundary and extreme inputs."""
    ctrl = FuzzySecurityController()
    _banner("Edge cases")

    # Perfect recognition
    r = ctrl.evaluate(confidence=100.0, illumination=128.0, facial_angle=0.0)
    print(f"  Perfect:   risk={r['security_risk']:.4f}  action={r['action']}")
    assert r["action"] == "unlock"

    # Worst case
    r = ctrl.evaluate(confidence=0.0, illumination=0.0, facial_angle=90.0)
    print(f"  Worst:     risk={r['security_risk']:.4f}  action={r['action']}")
    assert r["action"] in ("deny", "lockout")

    print("  ✓ PASS")


if __name__ == "__main__":
    test_rule1_unlock()
    test_rule2_otp()
    test_rule3_otp()
    test_rule4_deny()
    test_rule5_lockout()
    test_edge_cases()
    print(f"\n{'='*60}")
    print("  ALL TESTS PASSED")
    print(f"{'='*60}")
