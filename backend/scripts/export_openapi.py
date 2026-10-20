"""Write the API's OpenAPI schema; `npm run gen:api` in frontend/ generates types from it.

uv run python scripts/export_openapi.py <path>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from matchmade.api import create_app


def main() -> None:
    schema = create_app().openapi()
    path = Path(sys.argv[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    main()
