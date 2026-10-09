"""PyCharm-friendly launcher for the Streamlit product application."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "product_app" / "app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8502",
        "--server.address=127.0.0.1",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(streamlit_cli.main())
