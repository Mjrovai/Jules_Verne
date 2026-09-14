#!/usr/bin/env python3
"""
Start the Jules Verne Bot web demo.

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 8080
    python run.py --host 0.0.0.0  # reachable from other devices on your network
    python run.py --no-browser

The model is loaded once at startup (~0.1 s), so the page answers immediately.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Jules Verne Bot web demo.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (use 0.0.0.0 to share on your network).")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window automatically.")
    parser.add_argument("--reload", action="store_true",
                        help="Reload the server when source files change (development).")
    return parser.parse_args()


def preflight() -> None:
    """Fail early with a helpful message if the model or books are missing."""
    from vernebot.engine import BOOKS_DIR, DEFAULT_MODEL_PATH

    model_path = Path(os.environ.get("VERNE_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        sys.exit(
            f"\n[!] Model not found: {model_path}\n\n"
            "    The web app reads the trained weights from models/.\n"
            "    If your model file is named differently, point at it explicitly:\n\n"
            "        VERNE_MODEL_PATH=/path/to/verne_rnn_model.keras python run.py\n"
        )

    if not Path(os.environ.get("VERNE_BOOKS_DIR", BOOKS_DIR)).is_dir():
        sys.exit(
            "\n[!] books/ directory not found.\n\n"
            "    The character vocabulary is rebuilt from the same ten novels used\n"
            "    for training, so those .txt files must sit next to run.py.\n"
        )


def main() -> None:
    args = parse_args()
    preflight()

    import uvicorn

    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{display_host}:{args.port}"

    print("\n  Jules Verne Bot")
    print("  " + "-" * 46)
    print(f"  Serving at  {url}")
    if args.host == "0.0.0.0":
        print("  Bound to all interfaces - other devices on your network can reach it.")
    print("  Press Ctrl+C to stop.\n")

    if not args.no_browser:
        # Give uvicorn a moment to bind before opening the browser.
        def open_browser() -> None:
            time.sleep(1.2)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
