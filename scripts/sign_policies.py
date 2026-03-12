#!/usr/bin/env python3
"""
Summit.OS Policy Signing Script

Signs all OPA .rego policy files in infra/policy/ with an Ed25519 key.
Run this whenever you add or update a policy file.

Usage:
    # Sign all policies (generates new key if POLICY_SIGNING_KEY not set)
    python scripts/sign_policies.py

    # Sign specific directory
    python scripts/sign_policies.py --dir infra/policy/

    # Verify signatures only (don't sign)
    python scripts/sign_policies.py --verify

    # Print the public key (to set POLICY_VERIFY_KEY on all Summit.OS services)
    python scripts/sign_policies.py --show-key

Environment:
    POLICY_SIGNING_KEY  - hex Ed25519 private key (32 bytes)
                          If not set, generates and prints a new key.
"""
import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "packages"))

from policy.signer import PolicySigner, PolicyVerificationError


def main():
    parser = argparse.ArgumentParser(description="Summit.OS Policy Signer")
    parser.add_argument("--dir", default=str(repo_root / "infra" / "policy"),
                        help="Directory containing .rego files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing signatures only (do not sign)")
    parser.add_argument("--show-key", action="store_true",
                        help="Print the public verification key and exit")
    args = parser.parse_args()

    signer = PolicySigner()

    if args.show_key:
        pub = signer.export_public_key_hex()
        if pub:
            print(f"\nPublic key (set as POLICY_VERIFY_KEY on all Summit.OS services):\n{pub}\n")
        else:
            print("Public key unavailable — is PyNaCl installed?")
        return

    policy_dir = args.dir
    if not os.path.isdir(policy_dir):
        print(f"Error: directory not found: {policy_dir}", file=sys.stderr)
        sys.exit(1)

    rego_files = sorted(Path(policy_dir).glob("*.rego"))
    if not rego_files:
        print(f"No .rego files found in {policy_dir}")
        return

    if args.verify:
        print(f"Verifying {len(rego_files)} policy files in {policy_dir}...")
        errors = []
        for f in rego_files:
            try:
                ok = signer.verify_file(str(f))
                status = "OK" if ok else "NO SIG"
                print(f"  {status:6}  {f.name}")
            except PolicyVerificationError as e:
                print(f"  FAIL    {f.name}: {e}")
                errors.append(f.name)

        if errors:
            print(f"\n{len(errors)} policy file(s) FAILED verification:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print(f"\nAll {len(rego_files)} policies verified successfully.")
    else:
        print(f"Signing {len(rego_files)} policy files in {policy_dir}...")
        signed = []
        for f in rego_files:
            result = signer.sign_file(str(f))
            if result:
                print(f"  SIGNED  {f.name}")
                signed.append(f.name)
            else:
                print(f"  SKIP    {f.name} (signing unavailable)")

        print(f"\nSigned {len(signed)}/{len(rego_files)} policy files.")
        pub = signer.export_public_key_hex()
        if pub:
            print(f"\nPublic verification key (set as POLICY_VERIFY_KEY):\n{pub}\n")


if __name__ == "__main__":
    main()
