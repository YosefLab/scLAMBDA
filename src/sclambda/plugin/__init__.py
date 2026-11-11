"""Locate the plugin bundled with a checkout or installed wheel."""
from pathlib import Path


def plugin_directory() -> Path:
    packaged = Path(__file__).resolve().parent
    checkout = packaged.parents[2]
    for root in (packaged, checkout):
        if (root / ".claude-plugin/plugin.json").is_file() and (root / "skills").is_dir():
            return root
    raise FileNotFoundError("Plugin assets are missing; reinstall sclambda from a complete release")
