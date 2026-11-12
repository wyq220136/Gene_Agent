"""公共运行接口不能因包初始化而加载任何物理引擎。"""
import subprocess
import sys
from pathlib import Path


def test_contracts_import_without_engines():
    code = '''
import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco','sapien'}:
            raise RuntimeError('implicit engine: '+fullname)
sys.meta_path.insert(0, Block())
from gene_agent.simulation.contracts import RuntimePort, RuntimeCapabilities
import gene_agent.simulation
assert 'mujoco' not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
