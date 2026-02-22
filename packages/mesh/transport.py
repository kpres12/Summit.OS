"""
Mesh Transport Layer for Summit.OS

Real asyncio UDP transport replacing mock/stub networking:
- UDPTransport: asyncio DatagramProtocol for peer-to-peer messaging
- FramedMessage: length-prefixed framing with message types
- EncryptionEnvelope: AES-GCM encryption for all mesh traffic
- TransportManager: manages connections and message routing

Designed for DDIL (Denied, Degraded, Intermittent, Limited) environments.
"""
from __future__ import annotations

import asyncio
import json
import time
import struct
import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple
from enum import IntEnum

logger = logging.getLogger("mesh.transport")


# ── Message Types ───────────────────────────────────────────

class MessageType(IntEnum):
    HEARTBEAT = 0x01
    SYNC_REQUEST = 0x02
    SYNC_RESPONSE = 0x03
    PING = 0x04
    PING_ACK = 0x05
    INDIRECT_PING = 0x06
    JOIN = 0x07
    LEAVE = 0x08
    BROADCAST = 0x09
    DATA = 0x0A
    ACK = 0x0B


# ── Framing ─────────────────────────────────────────────────

@dataclass
class FramedMessage:
    """
    Wire format for mesh messages.

    Layout (big-endian):
    [4B magic] [1B version] [1B msg_type] [4B payload_len] [16B nonce] [payload] [32B hmac]
    """
    MAGIC = b"SMSH"  # Summit Mesh
    VERSION = 1

    msg_type: MessageType
    payload: bytes
    nonce: bytes = b""
    sender_id: str = ""
    timestamp: float = 0.0

    def encode(self, hmac_key: bytes = b"") -> bytes:
        """Serialize to wire format."""
        if not self.nonce:
            self.nonce = os.urandom(16)
        if not self.timestamp:
            self.timestamp = time.time()

        header = struct.pack(
            ">4sBBI16s",
            self.MAGIC,
            self.VERSION,
            self.msg_type,
            len(self.payload),
            self.nonce,
        )

        data = header + self.payload

        if hmac_key:
            import hmac as hmac_mod
            mac = hmac_mod.new(hmac_key, data, hashlib.sha256).digest()
        else:
            mac = b"\x00" * 32

        return data + mac

    @classmethod
    def decode(cls, raw: bytes, hmac_key: bytes = b"") -> Optional["FramedMessage"]:
        """Deserialize from wire format."""
        # Minimum: header (26B) + hmac (32B) = 58B
        if len(raw) < 58:
            return None

        magic = raw[:4]
        if magic != cls.MAGIC:
            return None

        version, msg_type, payload_len = struct.unpack(">BBI", raw[4:10])
        nonce = raw[10:26]
        payload = raw[26:26 + payload_len]
        mac = raw[26 + payload_len:26 + payload_len + 32]

        if hmac_key:
            import hmac as hmac_mod
            expected_mac = hmac_mod.new(hmac_key, raw[:26 + payload_len], hashlib.sha256).digest()
            if not hmac_mod.compare_digest(mac, expected_mac):
                logger.warning("HMAC verification failed — message tampered or wrong key")
                return None

        return cls(
            msg_type=MessageType(msg_type),
            payload=payload,
            nonce=nonce,
        )


# ── Encryption Envelope ─────────────────────────────────────

class EncryptionEnvelope:
    """
    AES-256-GCM encryption for mesh traffic.

    Uses cryptography lib when available, falls back to XOR "encryption"
    for development/testing only.
    """

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or os.urandom(32)
        self._crypto_available = self._check_crypto()

    @staticmethod
    def _check_crypto() -> bool:
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            return True
        except ImportError:
            return False

    def encrypt(self, plaintext: bytes, nonce: bytes,
                aad: Optional[bytes] = None) -> bytes:
        """Encrypt plaintext with AES-256-GCM."""
        if self._crypto_available:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            # GCM needs 12-byte nonce
            gcm_nonce = nonce[:12] if len(nonce) >= 12 else nonce + b"\x00" * (12 - len(nonce))
            return aesgcm.encrypt(gcm_nonce, plaintext, aad)
        else:
            # XOR fallback — NOT SECURE, dev only
            key_stream = hashlib.sha256(self._key + nonce).digest()
            return bytes(p ^ key_stream[i % 32] for i, p in enumerate(plaintext))

    def decrypt(self, ciphertext: bytes, nonce: bytes,
                aad: Optional[bytes] = None) -> bytes:
        """Decrypt ciphertext with AES-256-GCM."""
        if self._crypto_available:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            gcm_nonce = nonce[:12] if len(nonce) >= 12 else nonce + b"\x00" * (12 - len(nonce))
            return aesgcm.decrypt(gcm_nonce, ciphertext, aad)
        else:
            key_stream = hashlib.sha256(self._key + nonce).digest()
            return bytes(c ^ key_stream[i % 32] for i, c in enumerate(ciphertext))


# ── UDP Protocol ────────────────────────────────────────────

class MeshUDPProtocol(asyncio.DatagramProtocol):
    """Asyncio UDP protocol for mesh communication."""

    def __init__(self, on_message: Callable, hmac_key: bytes = b"",
                 encryption: Optional[EncryptionEnvelope] = None):
        self.on_message = on_message
        self.hmac_key = hmac_key
        self.encryption = encryption
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.messages_received = 0
        self.messages_sent = 0
        self.bytes_received = 0
        self.bytes_sent = 0

    def connection_made(self, transport: asyncio.DatagramTransport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        self.messages_received += 1
        self.bytes_received += len(data)

        msg = FramedMessage.decode(data, self.hmac_key)
        if msg is None:
            logger.debug(f"Invalid message from {addr}")
            return

        # Decrypt if encryption is enabled
        payload = msg.payload
        if self.encryption and msg.msg_type != MessageType.JOIN:
            try:
                payload = self.encryption.decrypt(payload, msg.nonce)
            except Exception as e:
                logger.warning(f"Decryption failed from {addr}: {e}")
                return

        try:
            message_data = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            message_data = payload

        self.on_message(msg.msg_type, message_data, addr)

    def send(self, msg_type: MessageType, payload: Any,
             target: Tuple[str, int]) -> None:
        """Send a message to a target address."""
        if self.transport is None:
            return

        if isinstance(payload, dict):
            raw_payload = json.dumps(payload).encode()
        elif isinstance(payload, str):
            raw_payload = payload.encode()
        elif isinstance(payload, bytes):
            raw_payload = payload
        else:
            raw_payload = json.dumps(payload).encode()

        nonce = os.urandom(16)

        # Encrypt if enabled
        if self.encryption and msg_type != MessageType.JOIN:
            raw_payload = self.encryption.encrypt(raw_payload, nonce)

        msg = FramedMessage(msg_type=msg_type, payload=raw_payload, nonce=nonce)
        data = msg.encode(self.hmac_key)

        try:
            self.transport.sendto(data, target)
            self.messages_sent += 1
            self.bytes_sent += len(data)
        except Exception as e:
            logger.error(f"Send failed to {target}: {e}")

    def error_received(self, exc: Exception):
        logger.error(f"UDP error: {exc}")

    def connection_lost(self, exc: Optional[Exception]):
        if exc:
            logger.warning(f"Connection lost: {exc}")


# ── Transport Manager ──────────────────────────────────────

class TransportManager:
    """
    Manages the mesh transport layer.

    Handles:
    - Starting/stopping UDP listener
    - Message routing to handlers
    - Periodic heartbeat/sync scheduling
    - Connection metrics
    """

    def __init__(self, bind_host: str = "0.0.0.0", bind_port: int = 9100,
                 hmac_key: Optional[bytes] = None,
                 encryption_key: Optional[bytes] = None):
        self.bind_host = bind_host
        self.bind_port = bind_port
        self.hmac_key = hmac_key or b""
        self.encryption = EncryptionEnvelope(encryption_key) if encryption_key else None

        self._protocol: Optional[MeshUDPProtocol] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._handlers: Dict[MessageType, List[Callable]] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def register_handler(self, msg_type: MessageType,
                         handler: Callable) -> None:
        """Register a message handler."""
        if msg_type not in self._handlers:
            self._handlers[msg_type] = []
        self._handlers[msg_type].append(handler)

    async def start(self) -> None:
        """Start the transport layer."""
        if self._running:
            return

        loop = asyncio.get_event_loop()

        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: MeshUDPProtocol(
                on_message=self._dispatch,
                hmac_key=self.hmac_key,
                encryption=self.encryption,
            ),
            local_addr=(self.bind_host, self.bind_port),
        )

        self._running = True
        logger.info(f"Mesh transport started on {self.bind_host}:{self.bind_port}")

    async def stop(self) -> None:
        """Stop the transport layer."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        if self._transport:
            self._transport.close()
            self._transport = None

        logger.info("Mesh transport stopped")

    def send(self, msg_type: MessageType, payload: Any,
             target: Tuple[str, int]) -> None:
        """Send a message to a target peer."""
        if self._protocol:
            self._protocol.send(msg_type, payload, target)

    def broadcast(self, msg_type: MessageType, payload: Any,
                  targets: List[Tuple[str, int]]) -> None:
        """Send a message to multiple peers."""
        for target in targets:
            self.send(msg_type, payload, target)

    def _dispatch(self, msg_type: MessageType, data: Any,
                  addr: Tuple[str, int]) -> None:
        """Dispatch incoming message to registered handlers."""
        handlers = self._handlers.get(msg_type, [])
        for handler in handlers:
            try:
                handler(data, addr)
            except Exception as e:
                logger.error(f"Handler error for {msg_type}: {e}")

    def schedule_periodic(self, coro_factory: Callable[[], Coroutine],
                          interval: float) -> None:
        """Schedule a periodic coroutine."""
        async def _loop():
            while self._running:
                try:
                    await coro_factory()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Periodic task error: {e}")
                await asyncio.sleep(interval)

        task = asyncio.ensure_future(_loop())
        self._tasks.append(task)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_stats(self) -> Dict:
        """Get transport statistics."""
        if self._protocol:
            return {
                "running": self._running,
                "messages_sent": self._protocol.messages_sent,
                "messages_received": self._protocol.messages_received,
                "bytes_sent": self._protocol.bytes_sent,
                "bytes_received": self._protocol.bytes_received,
            }
        return {"running": False}
