# 勺子抓取 Skill（橙色勺）

在仓库根目录运行：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/run_spoon_pick.py \
  --scene data/realworld/my_scene_0/scenes/scoop_orange \
  --output runs/spoon_pick/my_orange_run

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/recording/record_spoon_pick_video.py runs/spoon_pick/my_orange_run
```

视频在 `runs/spoon_pick/my_orange_run/spoon_pick_actual.mp4`，三张验收帧在 `video_frames/`。输入场景可用 `python tools/viewers/view_reconstructed_tabletop_scene.py --scene-dir data/realworld/my_scene_0/scenes/scoop_orange --mode mujoco` 查看。

检验不使用仿真辅助夹持的版本：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/run_spoon_pick.py \
  --scene data/realworld/my_scene_0/scenes/scoop_orange \
  --no-simulation-weld --grasp-handle-x-offset-m 0.006 \
  --output runs/spoon_pick/my_orange_physical

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/recording/record_spoon_pick_video.py runs/spoon_pick/my_orange_physical
```

抓取顺序为张爪、cuRobo 预抓取、cuRobo TCP 直线接近、MuJoCo 闭爪、cuRobo TCP 直线抬升、MuJoCo 保持。抓持目标从 `spoon_orange` 的 `grasp_handle` Concept 和本次落稳物体 pose 生成。每段规划使用执行后的实际状态；`motion_curobo/plans/` 保存真实 cuRobo 返回，`pick/result.json` 保存接触、抬升和保持检查。阶段接近桌面时，规划中仅排除保守的桌面球，执行前仍用完整 MuJoCo 几何检查机器人与桌面的整条轨迹。

原 Concept 抓取点的无 weld 版本有明显滑移，在本次名义场景略超保持后置条件。沿勺柄向勺头移动 6 mm 后，无 weld 版本在同一名义场景通过物理抓取检查。此时勺子完全由 MuJoCo 接触与摩擦保持，模型中没有 weld；规划仍使用抓取后保守的持物碰撞代理。辅助模式也保留：它只在实际双指接触核验通过后启用本次 run 的仿真 weld，并在结果和视频中标为**仿真辅助夹持**。两种模式均未验证随机初态泛化或真机抓取。闭爪时的 Pika 控制增益与力限只在本次 run 的 MuJoCo 模型里提高；源场景和真机配置没有改变。

2026-10-19 验收记录：`runs/spoon_pick/orange_curobo_20261019_v4` 从新 reset 完成，双指接触 8 个，直线抬升后勺子上升约 0.0877 m，保持后约 0.0877 m。录像由实际 MuJoCo qpos 生成，不使用规划轨迹作为物理回放。

无 weld 对照：`runs/spoon_pick/orange_no_weld_20261019` 使用 Concept 原始中心，保持时下滑 10.09 mm、相对漂移 25.78 mm，略超 10/25 mm 门限，判失败。`runs/spoon_pick/orange_no_weld_xplus6_20261019` 抓取点沿勺柄向勺头移动 6 mm，双指持续接触；抬升结束勺子上升 76.84 mm，保持结束仍上升 68.95 mm，下滑 7.89 mm、相对漂移 19.28 mm，满足当前 Skill 条件，判**纯 MuJoCo 物理抓取成功**。两次均由 cuRobo 规划同样的三个运动阶段，且场景中 weld 数量为 0。

同配置从 fresh reset 重跑 `runs/spoon_pick/orange_no_weld_xplus6_repeat_20261019` 得到相同数值和成功判定。当前实验只覆盖名义摆位；19.28 mm 的相对滑移仍可能影响后续舀球，需要在舀球 Skill 中单独检查勺头与球的接触和球的转移结果。

粉色勺使用同一抓取实现，通过 `--scene data/realworld/my_scene_0/scenes/scoop_pink` 选择场景。`runs/spoon_scoop_task/pink_full_verified_20261019` 从 fresh reset 验证双侧指垫接触后的仿真辅助抓取、盛球和倾倒连续任务；完整验收见[盛球倾倒说明](spoon_scoop_task.md)。粉色勺无 weld 全链仍未验证。
