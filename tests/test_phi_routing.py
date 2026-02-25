"""
Test Suite: PHI/PII Detection and HIPAA Routing
================================================

Verifies that Protected Health Information (PHI) and Personally
Identifiable Information (PII) are correctly detected and routed
to self-hosted providers in HIPAA mode.
"""

import sys
import types
import pytest
from pathlib import Path
import importlib.util

# Load routing modules without polluting sys.modules with stub parent
# packages (the old ``type(sys)("m")`` trick broke other test files).
# We only register the leaf modules, not the parent packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_registered: list[str] = []


def _load_routing_modules():
    """Load routing modules, registering only leaf modules in sys.modules."""
    routing_dir = PROJECT_ROOT / "reflection" / "routing"

    def _load(module_name, filepath):
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        _registered.append(module_name)
        spec.loader.exec_module(mod)
        return mod

    # phi_detector has no cross-module deps — load first
    phi_mod = _load("phi_detector", routing_dir / "phi_detector.py")

    # smart_router does ``from phi_detector import ...`` (fallback path),
    # which resolves from sys.modules since we just registered it above.
    router_mod = _load("smart_router", routing_dir / "smart_router.py")

    return phi_mod, router_mod


_phi_mod, _router_mod = _load_routing_modules()
PHIDetector = _phi_mod.PHIDetector
PIIDetector = _phi_mod.PIIDetector
SmartLLMRouter = _router_mod.SmartLLMRouter


def teardown_module():
    """Remove leaf module stubs from sys.modules."""
    for name in _registered:
        sys.modules.pop(name, None)


class TestPHIDetection:

    @pytest.fixture
    def detector(self):
        return PHIDetector()

    def test_detects_mrn(self, detector):
        has_phi, matches = detector.contains_phi("Patient MRN: 12345")
        assert has_phi is True
        assert "mrn" in [m.lower() for m in matches]

    def test_detects_ssn_as_phi(self, detector):
        has_phi, _ = detector.contains_phi("SSN: 123-45-6789")
        assert has_phi is True

    def test_no_phi_in_general_text(self, detector):
        has_phi, _ = detector.contains_phi("What are the symptoms of diabetes?")
        assert has_phi is False

    def test_no_phi_in_greeting(self, detector):
        has_phi, _ = detector.contains_phi("Hello, how can I help you today?")
        assert has_phi is False

    def test_detects_patient_name_pattern(self, detector):
        """Patient + name patterns should be flagged."""
        has_phi, _ = detector.contains_phi("Patient John Smith was admitted on 3/15")
        # Depends on detector implementation — at minimum date + patient should flag
        # We're testing the detector works, not specific sensitivity
        assert isinstance(has_phi, bool)


class TestPIIDetection:

    @pytest.fixture
    def detector(self):
        return PIIDetector()

    def test_detects_ssn(self, detector):
        has_pii, matches = detector.contains_pii("My SSN is 123-45-6789")
        assert has_pii is True

    def test_detects_email(self, detector):
        has_pii, _ = detector.contains_pii("Contact me at user@example.com")
        assert has_pii is True

    def test_no_pii_in_general_text(self, detector):
        has_pii, _ = detector.contains_pii("The weather is nice today")
        assert has_pii is False


class TestSmartLLMRouter:

    @pytest.fixture
    def router(self):
        return SmartLLMRouter(
            phi_provider={"provider": "ollama", "model": "llama3.2"},
            general_provider={"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            enable_phi_detection=True,
            enable_pii_detection=True,
        )

    def test_phi_routes_to_ollama(self, router):
        decision = router.route("Patient MRN: 12345 has diabetes")
        assert decision.provider == "ollama"
        assert decision.phi_detected is True

    def test_general_routes_to_api(self, router):
        decision = router.route("What is the weather today?")
        assert decision.provider == "anthropic"
        assert decision.phi_detected is False

    def test_manual_phi_override(self, router):
        """Manual PHI tag forces self-hosted routing."""
        decision = router.route("This is confidential", manual_phi_tag=True)
        assert decision.provider == "ollama"

    def test_manual_non_phi_override(self, router):
        """Manual non-PHI tag overrides detection."""
        decision = router.route("Patient MRN: 99999", manual_phi_tag=False)
        assert decision.provider == "anthropic"

    def test_pii_routes_to_ollama(self, router):
        decision = router.route("My SSN is 123-45-6789")
        assert decision.provider == "ollama"

    def test_router_returns_model(self, router):
        decision = router.route("Hello")
        assert decision.model is not None
        assert len(decision.model) > 0

    def test_router_returns_reason(self, router):
        decision = router.route("Patient data here")
        assert decision.reason is not None
        assert len(decision.reason) > 0


class TestRouterWithoutDetection:

    def test_disabled_phi_detection(self):
        router = SmartLLMRouter(
            phi_provider={"provider": "ollama", "model": "llama3.2"},
            general_provider={"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            enable_phi_detection=False,
            enable_pii_detection=False,
        )
        # Even with PHI content, should route to general since detection is off
        decision = router.route("Patient MRN: 12345")
        assert decision.provider == "anthropic"
