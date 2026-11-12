"""从当前场景独立查看实际 Knowledge 几何候选，不导入物理引擎。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
if __name__=='__main__':
    from gene_agent.visualization.pure_planning.cli import main
    main()
