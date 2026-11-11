"""Check wheel resources and imports independently of an editable checkout."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import zipfile


def check(wheel):
    wheel = Path(wheel).resolve()
    source = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="sclambda-wheel-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            for prompt in (source / 'src/sclambda/pipelines').rglob('prompts/*.md'):
                assert str(prompt.relative_to(source / 'src')) in names, prompt
            for folder in ('skills', 'hooks', '.claude-plugin', 'bin'):
                for asset in (source / folder).rglob('*'):
                    if asset.is_file():
                        assert 'sclambda/plugin/' + str(asset.relative_to(source)) in names, asset
            assert archive.getinfo('sclambda/plugin/bin/lambdakit').external_attr >> 16 & 0o111
            archive.extractall(root)
        script = r'''
import sys
sys.path.insert(0, sys.argv[1])
from importlib.metadata import distribution
from sclambda import __version__
from sclambda.plugin import plugin_directory
from sclambda.cli.main import main
from sclambda.core.accessor import Accessor
assert str(plugin_directory()).startswith(sys.argv[1])
main(['--version'])  # argparse exits successfully
'''
        # --version raises SystemExit(0); all preceding checks must succeed.
        proc = subprocess.run([sys.executable, '-I', '-c', script, str(root)],
                              cwd=root, env=dict(os.environ, MPLCONFIGDIR=str(root / "mpl")), capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        print(proc.stdout.strip())
        # Check prompt lookup and graph construction without making an API call.
        script = r'''
import sys
sys.path.insert(0, sys.argv[1])
from sclambda.pipelines.celltype.nodes.common import _load_prompt
assert len(_load_prompt('ct_critic.md')) > 100
from sclambda.pipelines.registry import list_pipelines, get_pipeline_class
from sclambda.core.task_runner import TaskRunner
from unittest.mock import Mock
for name in list_pipelines():
    cls = get_pipeline_class(name)
    with TaskRunner(max_workers=1) as runner:
        cls(accessor=Mock(), llm_client=Mock()).build_graph(runner)
print(f'{len(list_pipelines())} installed pipelines compile; prompts and plugin assets are present.')
'''
        proc = subprocess.run([sys.executable, '-I', '-c', script, str(root)],
                              cwd=root, env=dict(os.environ, MPLCONFIGDIR=str(root / "mpl")), capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        print(proc.stdout.strip())
        env = dict(os.environ, PYTHONPATH=str(root), LAMBDAKIT_PYTHON=sys.executable)
        proc = subprocess.run(['bash', str(root / 'sclambda/plugin/bin/lambdakit'), '--version'],
                              cwd=root, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        print('Bundled launcher:', proc.stdout.strip())


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: python scripts/check_release.py path/to/sclambda.whl')
    check(sys.argv[1])
