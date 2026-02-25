# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Encryption Utilities

Provides encryption at rest using Fernet (AES-128-CBC + HMAC-SHA256).
Supports key derivation from passwords and key rotation.
"""

import base64
import hashlib
import logging
import secrets

logger = logging.getLogger(__name__)

# Try to import cryptography, graceful fallback if not available
try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore


# OWASP 2023 recommended iterations for PBKDF2-SHA256
PBKDF2_ITERATIONS = 480_000
SALT_LENGTH = 16


class EncryptionError(Exception):
    """Base exception for encryption errors."""

    pass


class DecryptionError(EncryptionError):
    """Failed to decrypt data."""

    pass


class KeyDerivationError(EncryptionError):
    """Failed to derive key."""

    pass


def generate_key() -> bytes:
    """
    Generate a new Fernet encryption key.

    Returns:
        32-byte key suitable for Fernet encryption.
    """
    if not CRYPTO_AVAILABLE:
        raise EncryptionError("cryptography package not installed")
    return Fernet.generate_key()


def derive_key_from_password(
    password: str | bytes, salt: bytes | None = None, iterations: int = PBKDF2_ITERATIONS
) -> tuple[bytes, bytes]:
    """
    Derive an encryption key from a password using PBKDF2.

    Args:
        password: The password to derive from
        salt: Optional salt (generated if not provided)
        iterations: PBKDF2 iterations (default: OWASP 2023 recommended)

    Returns:
        (derived_key, salt) tuple
    """
    if not CRYPTO_AVAILABLE:
        raise KeyDerivationError("cryptography package not installed")

    if isinstance(password, str):
        password = password.encode("utf-8")

    if salt is None:
        salt = secrets.token_bytes(SALT_LENGTH)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )

    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key, salt


class Encryptor:
    """
    Encryption wrapper for data at rest.

    Usage:
        encryptor = Encryptor(key)
        encrypted = encryptor.encrypt("sensitive data")
        decrypted = encryptor.decrypt(encrypted)
    """

    def __init__(self, key: str | bytes):
        """
        Initialize with an encryption key.

        Args:
            key: Fernet-compatible key (base64-encoded 32 bytes)
        """
        if not CRYPTO_AVAILABLE:
            raise EncryptionError("cryptography package not installed")

        if isinstance(key, str):
            key = key.encode("utf-8")

        self._fernet = Fernet(key)

    def encrypt(self, data: str | bytes) -> bytes:
        """
        Encrypt data.

        Args:
            data: String or bytes to encrypt

        Returns:
            Encrypted bytes (Fernet token)
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        return self._fernet.encrypt(data)

    def encrypt_string(self, data: str) -> str:
        """
        Encrypt a string and return base64-encoded result.

        Convenience method for storing encrypted strings.
        """
        encrypted = self.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode("ascii")

    def decrypt(self, token: str | bytes) -> bytes:
        """
        Decrypt data.

        Args:
            token: Fernet token to decrypt

        Returns:
            Decrypted bytes

        Raises:
            DecryptionError: If decryption fails
        """
        if isinstance(token, str):
            token = token.encode("utf-8")

        try:
            return self._fernet.decrypt(token)
        except InvalidToken as e:
            raise DecryptionError("Failed to decrypt: invalid token or key") from e

    def decrypt_string(self, encrypted: str) -> str:
        """
        Decrypt a base64-encoded encrypted string.

        Convenience method for retrieving encrypted strings.
        """
        token = base64.urlsafe_b64decode(encrypted.encode("ascii"))
        return self.decrypt(token).decode("utf-8")

    def rotate_key(self, new_key: str | bytes, data: bytes) -> bytes:
        """
        Re-encrypt data with a new key.

        Args:
            new_key: New encryption key
            data: Currently encrypted data

        Returns:
            Data encrypted with new key
        """
        # Decrypt with current key
        plaintext = self.decrypt(data)

        # Encrypt with new key
        new_encryptor = Encryptor(new_key)
        return new_encryptor.encrypt(plaintext)


class EncryptedField:
    """
    Descriptor for encrypted model fields.

    Usage:
        class User:
            ssn = EncryptedField()

            def __init__(self, encryptor: Encryptor):
                self._encryptor = encryptor
    """

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = f"_encrypted_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        encrypted = getattr(obj, self.private_name, None)
        if encrypted is None:
            return None

        encryptor = getattr(obj, "_encryptor", None)
        if encryptor is None:
            raise EncryptionError("No encryptor configured on object")

        return encryptor.decrypt_string(encrypted)

    def __set__(self, obj, value):
        if value is None:
            setattr(obj, self.private_name, None)
            return

        encryptor = getattr(obj, "_encryptor", None)
        if encryptor is None:
            raise EncryptionError("No encryptor configured on object")

        encrypted = encryptor.encrypt_string(value)
        setattr(obj, self.private_name, encrypted)


def hash_for_lookup(data: str, salt: str = "") -> str:
    """
    Create a deterministic hash for encrypted field lookups.

    Note: This enables searching encrypted fields but reduces security.
    Use sparingly and only when necessary.

    Args:
        data: Data to hash
        salt: Application-specific salt

    Returns:
        Hex-encoded SHA-256 hash
    """
    combined = f"{salt}:{data}".encode()
    return hashlib.sha256(combined).hexdigest()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "CRYPTO_AVAILABLE",
    "EncryptionError",
    "DecryptionError",
    "KeyDerivationError",
    "generate_key",
    "derive_key_from_password",
    "Encryptor",
    "EncryptedField",
    "hash_for_lookup",
]
