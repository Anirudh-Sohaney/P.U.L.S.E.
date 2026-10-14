"""Reset one local account password and revoke all of its active sessions."""

from __future__ import annotations

import argparse
from getpass import getpass
import sys

from pydantic import ValidationError

from backend.routes.auth import reset_account_password


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set an account password locally and revoke its active sessions.")
    parser.add_argument("username", help="Account username to recover")
    args = parser.parse_args()

    confirmation = input(
        "A saved browser plan may remain encrypted with the old password. "
        "Unless the user knows that password, recovery requires retraining from "
        "their local files. Type RESET to continue: ")
    if confirmation != "RESET":
        print("Password reset cancelled.", file=sys.stderr)
        return 2

    password = getpass("New password (12–128 characters): ")
    password_confirmation = getpass("Confirm new password: ")
    if password != password_confirmation:
        print("Passwords do not match.", file=sys.stderr)
        return 2
    try:
        revoked = reset_account_password(args.username, password)
    except ValidationError as exc:
        messages = "; ".join(error["msg"] for error in exc.errors())
        print(f"Invalid account credentials: {messages}", file=sys.stderr)
        return 2
    except LookupError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Password reset for {args.username}; revoked {revoked} active session(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
