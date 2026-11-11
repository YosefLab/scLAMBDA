"""Synchronize plugin metadata with the single Python release version."""
from pathlib import Path
import json
import runpy

root = Path(__file__).resolve().parents[1]
version = runpy.run_path(str(root / "src/sclambda/_version.py"))["__version__"]
for name in ("plugin", "marketplace"):
    path = root / ".claude-plugin" / f"{name}.json"
    data = json.loads(path.read_text())
    if name == "plugin":
        data["version"] = version
    else:
        data["metadata"]["version"] = version
        for plugin in data["plugins"]:
            plugin["version"] = version
    path.write_text(json.dumps(data, indent=2) + "\n")
print(version)
