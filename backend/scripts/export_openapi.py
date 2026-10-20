"""Write the API's OpenAPI schema; the frontend generates its types from it.

uv run python scripts/export_openapi.py ../frontend/src/api/openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from matchmade.api import create_app


def main() -> None:
    schema = create_app().openapi()
    Path(sys.argv[1]).write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    main()
