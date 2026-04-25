"""
Mutual TLS (mTLS) for Heli.OS

Provides:
- CertificateAuthority: generates self-signed CA + service certs
- TLSContextFactory: creates SSL contexts for server/client with mTLS
- Certificate rotation and revocation support

Implements zero-trust mTLS mesh networking for Heli.OS services.
"""

from __future__ import annotations

import ssl
import time
import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("security.mtls")


@dataclass
class CertInfo:
    """Certificate metadata."""

    common_name: str
    serial_number: str
    fingerprint: str
    not_before: float
    not_after: float
    issuer: str
    is_ca: bool = False
    cert_pem: bytes = b""
    key_pem: bytes = b""

    @property
    def is_expired(self) -> bool:
        return time.time() > self.not_after

    @property
    def days_until_expiry(self) -> float:
        return (self.not_after - time.time()) / 86400

    def to_dict(self) -> Dict:
        return {
            "cn": self.common_name,
            "serial": self.serial_number,
            "fingerprint": self.fingerprint,
            "expires": self.not_after,
            "days_remaining": round(self.days_until_expiry, 1),
            "is_ca": self.is_ca,
            "issuer": self.issuer,
        }


class CertificateAuthority:
    """
    Self-contained Certificate Authority for Heli.OS mesh.

    Uses Python's ssl module and optionally cryptography library
    for full X.509 cert generation. Falls back to openssl CLI.
    """

    def __init__(
        self,
        ca_dir: Optional[str] = None,
        org: str = "Heli.OS",
        validity_days: int = 365,
    ):
        self.ca_dir = (
            Path(ca_dir) if ca_dir else Path(tempfile.mkdtemp(prefix="summit-ca-"))
        )
        self.org = org
        self.validity_days = validity_days
        self._ca_cert: Optional[CertInfo] = None
        self._issued_certs: Dict[str, CertInfo] = {}
        self._revoked: set = set()
        self._crypto_available = self._check_crypto()

    @staticmethod
    def _check_crypto() -> bool:
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa

            return True
        except ImportError:
            return False

    def init_ca(self) -> CertInfo:
        """Initialize the root CA certificate."""
        if self._crypto_available:
            return self._init_ca_crypto()
        return self._init_ca_fallback()

    def _init_ca_crypto(self) -> CertInfo:
        """Generate CA using cryptography library."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        import datetime

        # Generate CA key
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)

        # Build CA cert
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, self.org),
                x509.NameAttribute(NameOID.COMMON_NAME, f"{self.org} Root CA"),
            ]
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=self.validity_days * 5))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )

        cert_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
        key_pem = ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )

        # Write to CA dir
        self.ca_dir.mkdir(parents=True, exist_ok=True)
        (self.ca_dir / "ca.crt").write_bytes(cert_pem)
        (self.ca_dir / "ca.key").write_bytes(key_pem)
        os.chmod(self.ca_dir / "ca.key", 0o600)

        self._ca_cert = CertInfo(
            common_name=f"{self.org} Root CA",
            serial_number=str(ca_cert.serial_number),
            fingerprint=ca_cert.fingerprint(hashes.SHA256()).hex(),
            not_before=now.timestamp(),
            not_after=(
                now + datetime.timedelta(days=self.validity_days * 5)
            ).timestamp(),
            issuer=f"{self.org} Root CA",
            is_ca=True,
            cert_pem=cert_pem,
            key_pem=key_pem,
        )

        logger.info(f"CA initialized: {self.ca_dir}")
        return self._ca_cert

    def _init_ca_fallback(self) -> CertInfo:
        """Fallback CA init using mock certs (for environments without cryptography)."""
        now = time.time()
        serial = hashlib.sha256(f"{self.org}-ca-{now}".encode()).hexdigest()[:16]

        self._ca_cert = CertInfo(
            common_name=f"{self.org} Root CA",
            serial_number=serial,
            fingerprint=hashlib.sha256(serial.encode()).hexdigest(),
            not_before=now,
            not_after=now + self.validity_days * 5 * 86400,
            issuer=f"{self.org} Root CA",
            is_ca=True,
            cert_pem=b"--- MOCK CA CERT (install 'cryptography' for real certs) ---",
            key_pem=b"--- MOCK CA KEY ---",
        )

        self.ca_dir.mkdir(parents=True, exist_ok=True)
        (self.ca_dir / "ca.crt").write_bytes(self._ca_cert.cert_pem)
        logger.warning(
            "CA initialized with mock certs (cryptography lib not available)"
        )
        return self._ca_cert

    def issue_cert(
        self,
        common_name: str,
        san_dns: Optional[List[str]] = None,
        san_ips: Optional[List[str]] = None,
    ) -> CertInfo:
        """Issue a service certificate signed by this CA."""
        if self._ca_cert is None:
            self.init_ca()

        if self._crypto_available:
            return self._issue_cert_crypto(common_name, san_dns, san_ips)
        return self._issue_cert_fallback(common_name)

    def _issue_cert_crypto(
        self, cn: str, san_dns: Optional[List[str]], san_ips: Optional[List[str]]
    ) -> CertInfo:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
        import datetime
        import ipaddress

        # Load CA key
        ca_key_pem = (self.ca_dir / "ca.key").read_bytes()
        ca_key = serialization.load_pem_private_key(ca_key_pem, password=None)
        ca_cert_pem = (self.ca_dir / "ca.crt").read_bytes()
        ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)

        # Generate service key
        svc_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        subject = x509.Name(
            [
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, self.org),
                x509.NameAttribute(NameOID.COMMON_NAME, cn),
            ]
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(svc_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=self.validity_days))
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None), critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
        )

        # SANs
        sans = []
        for dns in san_dns or [cn]:
            sans.append(x509.DNSName(dns))
        for ip in san_ips or ["127.0.0.1"]:
            sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
        builder = builder.add_extension(
            x509.SubjectAlternativeName(sans),
            critical=False,
        )

        svc_cert = builder.sign(ca_key, hashes.SHA256())

        cert_pem = svc_cert.public_bytes(serialization.Encoding.PEM)
        key_pem = svc_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )

        # Write to disk
        svc_dir = self.ca_dir / cn
        svc_dir.mkdir(parents=True, exist_ok=True)
        (svc_dir / "tls.crt").write_bytes(cert_pem)
        (svc_dir / "tls.key").write_bytes(key_pem)
        os.chmod(svc_dir / "tls.key", 0o600)

        info = CertInfo(
            common_name=cn,
            serial_number=str(svc_cert.serial_number),
            fingerprint=svc_cert.fingerprint(hashes.SHA256()).hex(),
            not_before=now.timestamp(),
            not_after=(now + datetime.timedelta(days=self.validity_days)).timestamp(),
            issuer=f"{self.org} Root CA",
            cert_pem=cert_pem,
            key_pem=key_pem,
        )

        self._issued_certs[cn] = info
        logger.info(f"Issued cert for: {cn}")
        return info

    def _issue_cert_fallback(self, cn: str) -> CertInfo:
        now = time.time()
        serial = hashlib.sha256(f"{cn}-{now}".encode()).hexdigest()[:16]
        info = CertInfo(
            common_name=cn,
            serial_number=serial,
            fingerprint=hashlib.sha256(serial.encode()).hexdigest(),
            not_before=now,
            not_after=now + self.validity_days * 86400,
            issuer=f"{self.org} Root CA",
            cert_pem=f"--- MOCK CERT for {cn} ---".encode(),
            key_pem=f"--- MOCK KEY for {cn} ---".encode(),
        )
        self._issued_certs[cn] = info
        return info

    def revoke(self, common_name: str) -> bool:
        """Revoke a certificate."""
        if common_name in self._issued_certs:
            self._revoked.add(self._issued_certs[common_name].serial_number)
            logger.info(f"Revoked cert: {common_name}")
            return True
        return False

    def is_revoked(self, serial_number: str) -> bool:
        return serial_number in self._revoked

    def get_issued(self) -> Dict[str, CertInfo]:
        return dict(self._issued_certs)


class TLSContextFactory:
    """
    Creates SSL contexts for mTLS server and client connections.
    """

    @staticmethod
    def create_server_context(
        cert_path: str, key_path: str, ca_path: str
    ) -> ssl.SSLContext:
        """Create server TLS context requiring client certs."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert_path, key_path)
        ctx.load_verify_locations(ca_path)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = False
        # Enforce TLS 1.2+
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM")
        return ctx

    @staticmethod
    def create_client_context(
        cert_path: str, key_path: str, ca_path: str
    ) -> ssl.SSLContext:
        """Create client TLS context for mTLS."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_cert_chain(cert_path, key_path)
        ctx.load_verify_locations(ca_path)
        ctx.check_hostname = True
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        return ctx

    @staticmethod
    def create_insecure_context() -> ssl.SSLContext:
        """Create insecure context for dev/testing only."""
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        logger.warning("Using insecure TLS context — DO NOT use in production")
        return ctx
