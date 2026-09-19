import sys
import os

# Ensure UTF-8 output on Windows terminals
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from arxiv_agent.cli import main

if __name__ == "__main__":
    main()
