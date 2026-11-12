# 查看与反馈入口

区分在线重新规划、只读回放与视频导出；界面不能把规划预览称为实际执行。

| 要查什么 | 从这里开始 |
|---|---|
| 无 MuJoCo 独立 Knowledge 候选 | [view_knowledge.py](view_knowledge.py) |
| 在线检查单个 Skill | [view_skill.py](view_skill.py) |
| 在线检查完整 Task；`--run <目录> --follow` 跟随落盘记录 | [view_task.py](view_task.py) |
| 查看已保存的 Skill 记录 | [view_skill_run.py](view_skill_run.py) |
| 只读 MuJoCo 回放 | [view_skill_mujoco.py](view_skill_mujoco.py) |
| 检查阶段状态链，不启动 GUI | [inspect_skill_run.py](inspect_skill_run.py) |
| 检查重建桌面场景 | [view_reconstructed_tabletop_scene.py](view_reconstructed_tabletop_scene.py) |
| 独立查看 benchmark 初态，不运行 policy | [view_benchmark.py](view_benchmark.py) |
| 检查 mesh 与 Concept | [inspect_concepts.py](inspect_concepts.py) |
| 查看保存的cuRobo现场、导出对齐visual/collision四图并调整相机 | [view_motion_diagnostics.py](view_motion_diagnostics.py) |
| 检查 Motion 调试包 | [view_motion_debug.py](view_motion_debug.py) |
| 新在线实际记录导出 MP4（杯子/勺子，支持失败前缀） | [record_live_video.py](recording/record_live_video.py) |
| 导出操作视频（首选统一入口） | [record_skill_run_video.py](recording/record_skill_run_video.py) |
| 自动截取指定 Skill 帧 | [capture_skill_run.py](capture/capture_skill_run.py) |

视频默认使用实际采样；没有采样不能生成虚构 actual。详见 [视频说明](../../docs/user/visualization/video_feedback.md)。
更专门的任务查看/截图脚本按需在本目录搜索，不要求先读每个辅助脚本。
