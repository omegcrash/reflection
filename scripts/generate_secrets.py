#!/usr/bin/env python3
# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Security Key Generator

Generates cryptographically secure keys for:
- JWT secret
- Master encryption key
- API keys

Usage:
    python scripts/generate_secrets.py
    python scripts/generate_secrets.py --env  # Output as .env format
    python scripts/generate_secrets.py --json # Output as JSON
"""

import argparse
import base64
import json
import secrets
import sys


def generate_jwt_secret(length: int = 64) -> str:
    """Generate a secure JWT secret key."""
    return secrets.token_urlsafe(length)


def generate_encryption_key() -> str:
    """Generate a 32-byte encryption key (base64 encoded)."""
    key = secrets.token_bytes(32)
    return base64.b64encode(key).decode('ascii')


def generate_api_key(prefix: str = "me_live_") -> str:
    """Generate an API key with prefix."""
    return f"{prefix}{secrets.token_urlsafe(32)}"


def generate_all_secrets() -> dict:
    """Generate all required secrets."""
    return {
        "SECURITY_JWT_SECRET_KEY": generate_jwt_secret(),
        "SECURITY_MASTER_ENCRYPTION_KEY": generate_encryption_key(),
        "SAMPLE_API_KEY": generate_api_key(),
    }


def format_env(secrets_dict: dict) -> str:
    """Format secrets as .env file content."""
    lines = [
        "# Reflection Security Configuration",
        "# Generated with scripts/generate_secrets.py",
        "# KEEP THESE VALUES SECRET - DO NOT COMMIT TO VERSION CONTROL",
        "",
    ]
    
    for key, value in secrets_dict.items():
        lines.append(f"{key}={value}")
    
    return "\n".join(lines)


def format_json(secrets_dict: dict) -> str:
    """Format secrets as JSON."""
    return json.dumps(secrets_dict, indent=2)


def format_plain(secrets_dict: dict) -> str:
    """Format secrets as plain text with descriptions."""
    output = [
        "=" * 60,
        "REFLECTION SECURITY KEYS",
        "=" * 60,
        "",
        "⚠️  IMPORTANT: Keep these values secret!",
        "    Do not commit to version control.",
        "    Store in a secure secrets manager.",
        "",
        "-" * 60,
    ]
    
    descriptions = {
        "SECURITY_JWT_SECRET_KEY": "JWT signing key (for authentication tokens)",
        "SECURITY_MASTER_ENCRYPTION_KEY": "Master encryption key (for tenant keys)",
        "SAMPLE_API_KEY": "Sample API key format",
    }
    
    for key, value in secrets_dict.items():
        desc = descriptions.get(key, "")
        output.extend([
            "",
            f"📌 {key}",
            f"   {desc}",
            f"   Value: {value}",
        ])
    
    output.extend([
        "",
        "-" * 60,
        "",
        "To use these values:",
        "",
        "1. Environment variables:",
        "   export SECURITY_JWT_SECRET_KEY='...'",
        "",
        "2. .env file:",
        "   SECURITY_JWT_SECRET_KEY=...",
        "",
        "3. Docker:",
        "   docker run -e SECURITY_JWT_SECRET_KEY='...'",
        "",
        "=" * 60,
    ])
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Generate secure keys for Reflection"
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="Output in .env file format"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    parser.add_argument(
        "--jwt-only",
        action="store_true",
        help="Generate only JWT secret"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Write output to file"
    )
    
    args = parser.parse_args()
    
    # Generate secrets
    if args.jwt_only:
        secrets_dict = {"SECURITY_JWT_SECRET_KEY": generate_jwt_secret()}
    else:
        secrets_dict = generate_all_secrets()
    
    # Format output
    if args.env:
        output = format_env(secrets_dict)
    elif args.json:
        output = format_json(secrets_dict)
    else:
        output = format_plain(secrets_dict)
    
    # Write or print
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"✅ Secrets written to {args.output}")
    else:
        print(output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
