请使用仓库内的Gene Agent任务失败诊断Skill处理这次实验。

先完整读取：
1. AGENTS.md
2. skills/gene-agent-task-failure-diagnosis/SKILL.md

需要时按SKILL.md继续读取references；如果进入cuRobo的IK、碰撞或路径内部原因，再读取：
skills/curobo-failure-diagnosis/SKILL.md

要分析的实验目录是本次的实验目录。

要求：
- 使用规范阶段图定位失败阶段；微波炉Task1固定使用docs/01-task-skills.svg。
- 从summary.json沿report_ref和traceback_ref进入阶段级证据。
- 从实际任务视频的失败位置抽帧，选择遮挡最少、最能看清问题的视角；不要用规划PNG代替实际视频。
- 按可观察表现、判负条件、参数证据、上游设计根因、准确代码位置的顺序分析。
- 不要停在IK未收敛、path_not_found、endpoint_tracking_error或后验失败这些中间结论。
- 检查目标生成、实际状态传递、TCP/frame、attachment、过渡策略、碰撞语义和判定条件是否合理。
- 找到实际运行调用链中的源码文件和函数，修改产生根因的位置。
- 不要通过放宽合理阈值、关闭碰撞、删除障碍物或盲目延长求解时间掩盖问题。
- 修改后恢复原失败初始状态复跑，再从fresh reset运行完整任务，报告整体成功或新的失败阶段。