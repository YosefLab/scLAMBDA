"""Include the repository's plugin assets in wheels without a second source copy."""
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py

class BuildWithPlugin(build_py):
    def run(self):
        super().run()
        root = Path(__file__).parent
        destination = Path(self.build_lib) / "sclambda" / "plugin"
        for name in ("skills", "hooks", ".claude-plugin", "bin"):
            self.copy_tree(str(root / name), str(destination / name))

setup(cmdclass={"build_py": BuildWithPlugin})
