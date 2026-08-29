"""
Vercel Serverless Function entrypoint for CADA FastAPI API.
"""

import sys
from pathlib import Path

# Add project root to path for serverless imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app

# Vercel looks for the ASGI `app` variable
__all__ = ["app"]
