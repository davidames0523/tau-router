from __future__ import annotations

import sys

from tau_lattice.cli.infinite import main as infinite_main
from tau_lattice.cli.nano import main as nano_main


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python -m tau_lattice [infinite|nano] [args...]")
        return
    cmd = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if cmd == "infinite":
        infinite_main()
    elif cmd == "nano":
        nano_main()
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
