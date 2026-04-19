"""
Heli.OS Device Certificate Authority

Issues per-device X.509 certificates so every physical device connecting
to Heli.OS has a cryptographic identity — not just a network address.

When a Modbus PLC or MAVLink drone registers with Heli.OS, it receives
a unique certificate signed by the Heli CA. All subsequent connections
carry that certificate. If a device is compromised, its cert is revoked
without touching anything else.

This uses Python's `cryptography` library (already a common dep in
secure Python services). The CA key should live in Vault in production.

Usage:
    ca = DeviceCA()
    await ca.initialize()  # creates CA cert/key if not exists

    cert = await ca.issue_device_cert("drone-01", device_type="uav")
    # cert.cert_pem -> PEM string to send to device
    # cert.fingerprint -> store this in DeviceRegistry for validation
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("summit.identity.ca")

CA_CERT_PATH = os.getenv("SUMMIT_CA_CERT", "/certs/summit-ca.crt")
CA_KEY_PATH = os.getenv("SUMMIT_CA_KEY", "/certs/summit-ca.key")
CERT_VALIDITY_DAYS = int(os.getenv("DEVICE_CERT_VALIDITY_DAYS", "365"))
CA_VALIDITY_YEARS = int(os.getenv("CA_VALIDITY_YEARS", "10"))


@dataclass
class DeviceCert:
    device_id: str
    device_type: str
    cert_pem: str
    key_pem: str
    fingerprint: str  # SHA-256 of DER-encoded cert
    serial_number: int
    not_before: datetime.datetime
    not_after: datetime.datetime
    org_id: str = ""

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "fingerprint": self.fingerprint,
            "serial_number": self.serial_number,
            "not_before": self.not_before.isoformat(),
            "not_after": self.not_after.isoformat(),
            "org_id": self.org_id,
        }


class DeviceCA:
    """
    Self-signed Certificate Authority for Heli.OS device identity.

    In production, replace the CA key storage with Vault PKI secrets engine.
    Vault PKI provides automatic rotation, CRL, and OCSP — this implementation
    is intentionally simple to remove the Vault hard dependency.
    """

    def __init__(
        self,
        ca_cert_path: str = CA_CERT_PATH,
        ca_key_path: str = CA_KEY_PATH,
    ):
        self.ca_cert_path = ca_cert_path
        self.ca_key_path = ca_key_path
        self._ca_cert = None
        self._ca_key = None
        self._initialized = False

    async def initialize(self):
        """Load or create the CA certificate and key."""
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.x509.oid import NameOID
        except ImportError:
            logger.warning(
                "cryptography library not installed — device identity disabled. "
                "Run: pip install cryptography"
            )
            return

        if os.path.exists(self.ca_cert_path) and os.path.exists(self.ca_key_path):
            await self._load_ca()
        else:
            await self._create_ca()

        self._initialized = True
        logger.info(f"DeviceCA initialized (cert={self.ca_cert_path})")

    async def _load_ca(self):
        """Load existing CA cert and key from disk."""
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization

        with open(self.ca_cert_path, "rb") as f:
            self._ca_cert = x509.load_pem_x509_certificate(f.read())
        with open(self.ca_key_path, "rb") as f:
            self._ca_key = serialization.load_pem_private_key(f.read(), password=None)
        logger.info("Loaded existing Heli CA certificate")

    async def _create_ca(self):
        """Create a new Heli CA certificate and key."""
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID

        logger.info("Creating new Heli CA certificate")
        key = ec.generate_private_key(ec.SECP256R1())

        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Heli.OS"),
                x509.NameAttribute(NameOID.COMMON_NAME, "Heli.OS Device CA"),
            ]
        )

        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=CA_VALIDITY_YEARS * 365))
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
            .sign(key, hashes.SHA256())
        )

        self._ca_cert = cert
        self._ca_key = key

        # Persist
        os.makedirs(os.path.dirname(self.ca_cert_path), exist_ok=True)
        with open(self.ca_cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        with open(self.ca_key_path, "wb") as f:
            f.write(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
        logger.info(f"Heli CA certificate created at {self.ca_cert_path}")

    async def issue_device_cert(
        self,
        device_id: str,
        device_type: str = "device",
        org_id: str = "",
    ) -> Optional[DeviceCert]:
        """
        Issue a certificate for a device.

        Returns None if CA is not initialized (cryptography lib missing).
        The returned key_pem should be sent to the device ONCE and never stored.
        """
        if not self._initialized or self._ca_cert is None or self._ca_key is None:
            logger.warning(f"CA not initialized — cannot issue cert for {device_id}")
            return None

        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.x509.oid import NameOID

            key = ec.generate_private_key(ec.SECP256R1())

            subject = x509.Name(
                [
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                    x509.NameAttribute(
                        NameOID.ORGANIZATION_NAME, org_id or "Heli.OS"
                    ),
                    x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, device_type),
                    x509.NameAttribute(NameOID.COMMON_NAME, device_id),
                ]
            )

            now = datetime.datetime.now(datetime.timezone.utc)
            not_after = now + datetime.timedelta(days=CERT_VALIDITY_DAYS)
            serial = x509.random_serial_number()

            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(self._ca_cert.subject)
                .public_key(key.public_key())
                .serial_number(serial)
                .not_valid_before(now)
                .not_valid_after(not_after)
                .add_extension(
                    x509.BasicConstraints(ca=False, path_length=None), critical=True
                )
                .add_extension(
                    x509.SubjectAlternativeName(
                        [
                            x509.DNSName(device_id),
                            x509.DNSName(f"{device_type}.summit.local"),
                        ]
                    ),
                    critical=False,
                )
                .add_extension(
                    x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.CLIENT_AUTH]),
                    critical=False,
                )
                .sign(self._ca_key, hashes.SHA256())
            )

            cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
            key_pem = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode()

            fingerprint = hashlib.sha256(
                cert.public_bytes(serialization.Encoding.DER)
            ).hexdigest()

            logger.info(
                f"Issued cert for device '{device_id}' (fp={fingerprint[:16]}...)"
            )

            return DeviceCert(
                device_id=device_id,
                device_type=device_type,
                cert_pem=cert_pem,
                key_pem=key_pem,
                fingerprint=fingerprint,
                serial_number=serial,
                not_before=now,
                not_after=not_after,
                org_id=org_id,
            )

        except Exception as e:
            logger.error(f"Failed to issue cert for '{device_id}': {e}")
            return None

    def verify_cert(self, cert_pem: str) -> Optional[dict]:
        """
        Verify a device cert was signed by this CA.

        Returns device info dict if valid, None if invalid/expired.
        """
        if not self._initialized or self._ca_cert is None:
            return None

        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding
            from cryptography.x509.oid import NameOID

            cert = x509.load_pem_x509_certificate(cert_pem.encode())

            # Verify signature — EC and RSA keys require different verify() call signatures
            pub_key = self._ca_cert.public_key()
            hash_alg = cert.signature_hash_algorithm
            if isinstance(pub_key, ec.EllipticCurvePublicKey):
                pub_key.verify(
                    cert.signature, cert.tbs_certificate_bytes, ec.ECDSA(hash_alg)
                )
            elif isinstance(pub_key, rsa.RSAPublicKey):
                pub_key.verify(
                    cert.signature,
                    cert.tbs_certificate_bytes,
                    padding.PKCS1v15(),
                    hash_alg,
                )
            else:
                raise ValueError(f"Unsupported CA key type: {type(pub_key).__name__}")

            # Check expiry
            now = datetime.datetime.now(datetime.timezone.utc)
            if now < cert.not_valid_before_utc or now > cert.not_valid_after_utc:
                logger.warning(f"Certificate expired or not yet valid: {cert.subject}")
                return None

            subject = cert.subject
            fingerprint = hashlib.sha256(
                cert.public_bytes(
                    __import__(
                        "cryptography"
                    ).hazmat.primitives.serialization.Encoding.DER
                )
            ).hexdigest()

            return {
                "device_id": subject.get_attributes_for_oid(NameOID.COMMON_NAME)[
                    0
                ].value,
                "device_type": (
                    subject.get_attributes_for_oid(NameOID.ORGANIZATIONAL_UNIT_NAME)[
                        0
                    ].value
                    if subject.get_attributes_for_oid(NameOID.ORGANIZATIONAL_UNIT_NAME)
                    else "unknown"
                ),
                "org_id": subject.get_attributes_for_oid(NameOID.ORGANIZATION_NAME)[
                    0
                ].value,
                "fingerprint": fingerprint,
                "not_after": cert.not_valid_after_utc.isoformat(),
            }

        except Exception as e:
            logger.warning(f"Certificate verification failed: {e}")
            return None
