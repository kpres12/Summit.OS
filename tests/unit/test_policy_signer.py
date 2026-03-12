"""Unit tests for the OPA policy signing/verification system."""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages"))


class TestPolicySigner:
    """Tests skip gracefully if PyNaCl is not installed."""

    @pytest.fixture
    def signer(self):
        from policy.signer import PolicySigner
        return PolicySigner(enforce=True)

    @pytest.fixture
    def signer_nonenforce(self):
        from policy.signer import PolicySigner
        return PolicySigner(enforce=False)

    @pytest.fixture
    def sample_policy(self, tmp_path):
        p = tmp_path / "test.rego"
        p.write_text('package test\ndefault allow := true\n')
        return str(p)

    def test_sign_creates_sig_file(self, signer, sample_policy):
        if not signer._available:
            pytest.skip("PyNaCl not installed")
        sig = signer.sign_file(sample_policy)
        assert sig is not None
        assert os.path.exists(f"{sample_policy}.sig")

    def test_verify_signed_file_passes(self, signer, sample_policy):
        if not signer._available:
            pytest.skip("PyNaCl not installed")
        signer.sign_file(sample_policy)
        assert signer.verify_file(sample_policy) is True

    def test_tampered_file_raises_in_enforce_mode(self, signer, sample_policy):
        if not signer._available:
            pytest.skip("PyNaCl not installed")
        from policy.signer import PolicyVerificationError
        signer.sign_file(sample_policy)
        # Tamper with the policy
        with open(sample_policy, "a") as f:
            f.write("\n# MALICIOUS CHANGE\n")
        with pytest.raises(PolicyVerificationError):
            signer.verify_file(sample_policy)

    def test_tampered_file_returns_false_in_nonenforce_mode(self, signer_nonenforce, sample_policy):
        if not signer_nonenforce._available:
            pytest.skip("PyNaCl not installed")
        signer_nonenforce.sign_file(sample_policy)
        with open(sample_policy, "a") as f:
            f.write("\n# tamper\n")
        result = signer_nonenforce.verify_file(sample_policy)
        assert result is False

    def test_missing_sig_file_raises_in_enforce_mode(self, signer, sample_policy):
        if not signer._available:
            pytest.skip("PyNaCl not installed")
        from policy.signer import PolicyVerificationError
        # No .sig file exists
        with pytest.raises(PolicyVerificationError):
            signer.verify_file(sample_policy)

    def test_missing_sig_file_passes_in_nonenforce_mode(self, signer_nonenforce, sample_policy):
        # No .sig file — should pass with warning
        result = signer_nonenforce.verify_file(sample_policy)
        assert result is True

    def test_sign_all_signs_rego_files(self, signer, tmp_path):
        if not signer._available:
            pytest.skip("PyNaCl not installed")
        (tmp_path / "a.rego").write_text("package a\ndefault allow := true\n")
        (tmp_path / "b.rego").write_text("package b\ndefault allow := false\n")
        (tmp_path / "not_a_policy.txt").write_text("ignored")
        signed = signer.sign_all(str(tmp_path))
        assert len(signed) == 2
        assert all(f.endswith(".rego") for f in signed)

    def test_public_key_export(self, signer):
        if not signer._available:
            pytest.skip("PyNaCl not installed")
        pub = signer.export_public_key_hex()
        assert pub is not None
        assert len(pub) == 64  # 32 bytes = 64 hex chars
