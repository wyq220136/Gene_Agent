# MuJoCo 同步 Runtime 与失败反馈

受众：Pipeline、Skill 和自动化开发入口的实现者。当前边界是短关节轨迹闭环与开发诊断，
尚未实现抓碗或旋钮 Skill。

## 运行边界

`MujocoRuntime` 持有一份在线 `MjModel/MjData`。Pipeline 通过
`query(object_names=..., robot_groups=...)` 主动读取当前状态；查询在 data 副本上执行
`mj_forward`，不改变在线积分状态。返回值只包含真实系统中可由相应感知或机器人状态接口提供的
具名信息：物体 pose、关节位置/速度、TCP pose 和夹爪开口。当前仿真阶段这些值直接取 MuJoCo GT，
不经过相机，也不向 Pipeline 暴露精确 contact、geom ID 或 MuJoCo 内部数组。

每次 observation 都带 `state_id`。规划结果执行时必须传回生成规划所依据的 ID；仿真状态变化后，
旧 ID 会在写 `ctrl` 前被拒绝。`execute_joint_trajectory` 当前只支持具名 hinge 组和唯一的 position
actuator，只写 actuator `ctrl` 并调用 `mj_step`，不写 `qpos`、不瞬移物体。

```python
observation = runtime.query(object_names=("bowl",), robot_groups=("right",))
execution = runtime.execute_joint_trajectory(
    group="right",
    joint_names=plan["joint_names"],
    times_s=times_s,
    positions_rad=plan["positions_rad"],
    planned_state_id=observation["state_id"],
)
```

平行夹爪使用同样的状态版本约束，输入和反馈都以两内表面总间隙（米）表达：

```python
observation = runtime.query(robot_groups=("right",))
gripper_execution = runtime.execute_gripper_command(
    group="right",
    width_m=0.041,
    duration_s=0.3,
    planned_state_id=observation["state_id"],
)
```

该接口逐 timestep 只写两个 position actuator 的 `ctrl`，保存参考/实测开口、对称误差、起止
`state_id`，并与机械臂执行统一报告 timeout、非有限状态和 MuJoCo instability。

## Codex 诊断边界

`query_diagnostics` 单独读取 MuJoCo contact、具名 geom/body、接触距离和力。这些信息用于开发、
异常定位和自动迭代，不进入 Pipeline observation。

`query_frame_diagnostics(runtime, object_name="bowl", group="right", target_pose=...)` 在同一冻结状态中
输出物体中心与规划目标分别相对实际 TCP 的 world/TCP 坐标差。省略 `target_pose` 时仍可只查询物体
中心。`render_views(debug_frames=...)` 可选择性画出 WORLD/right SDK base、TCP 和 TARGET 坐标系：
X红、Y绿、Z蓝，TCP 原点为粉色，TARGET 原点为黄色。叠图的文字和毫米数值放在右侧栏，
不盖住场景中的碗和夹爪；原图仍单独保存。`frame_diagnostics=...` 只接受与当前渲染
`state_id` 一致的数据。坐标箭头只追加到 renderer scene，不改变在线模型、碰撞或积分状态。

## 抓碗操作接口探针

`tools/run/check_pick_bowl_interface.py` 是临时调用方，用现有 Knowledge 候选和 cuRobo 检查接口
是否支持“开爪—查询—规划预抓取—执行—再查询—规划接触—闭爪—接触分类”。它不是正式抓取
Skill，不修改候选生成或 Viser 规划界面。
每个运动阶段通过 `gene_agent.motion.runtime_bridge.CuroboRuntimeBridge` 主动查询最新状态、
冻结新碰撞场景、规划并执行；Pipeline 可独立复用这个单阶段边界。

```bash
MUJOCO_GL=egl PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/check_pick_bowl_interface.py \
  --output runs/my-pick-interface \
  --candidate-id bowl_ring_00_00 --depth-m 0.012 --retreat-m 0.03
```

以上候选、深度和退后距离现为默认值。加 `--seed 17 --perturb` 会按配置在标称位姿附近
扰动场景，并在 run 的 `model/objects/scene_metadata.json` 中记录采样和稳定前位姿。
当前默认场景 `runs/pick-bowl-interface-v9-default-side90/` 和三个扰动
`runs/pick-bowl-perturb-seed-{17,23,41}/` 均完成闭爪时双侧 pad—碗接触，未记录非 pad
夹爪—碗接触；这是四个确定性开发样本，不代表成功率统计或抬碗能力。
将单阶段规划执行逻辑移入 `gene_agent/motion/runtime_bridge.py` 后，默认场景
`runs/pick-bowl-interface-v10-bridge/` 再次通过真实 GPU 规划与 MuJoCo 闭爪。
这个边界接收上一轮查询的 `planned_state_id`，内部再次查询状态、冻结碰撞场景、规划并执行；
旧 ID 在规划前拒绝。失败回调由调用方提供，实验入口用它自动生成 Codex 证据包。

2026-10-06 的三次 RTX 4090 有界迭代均完成开爪和预抓取执行，并保存每阶段 observation、可恢复
state、接触、WORLD/TCP/TARGET 坐标轴图和碰撞叠图。最后一轮预抓取后，碗中心相对实际 TCP 的
坐标差为：

- world：`[-0.00328, 0.03927, -0.05328] m`；
- TCP：`[-0.03267, -0.01319, 0.05613] m`。

该轮最终接触目标相对实际 TCP 的坐标差为：

- world：`[0.02285, -0.02835, -0.04026] m`；
- TCP：`[0.03840, -0.00971, 0.03711] m`。

这组 TARGET 误差来自保存的预抓取状态，不依赖重新执行动力学，可在
`runs/pick-bowl-interface-v3/target_frame_review/` 中复查。

最后接触 pose 的 cuRobo IK 从实际预抓取状态仍返回 `goal_ik_no_solution`，所以探针按三轮上限停止，
没有闭爪或宣称抓取成功。该结果证明状态查询、夹爪控制、阶段状态版本、坐标反馈、证据冻结与结构化
失败接口已连通；同时暴露出正式操作层仍需要更可靠的局部接近规划或笛卡尔短程控制。原先 cuRobo 的
合法 `None` 返回已归一化为 `goal_ik_no_solution`，不再误报 `AttributeError`。

后续失败诊断保存在 `runs/pick-bowl-interface-v6-obstruction/plans/final.json`。cuRobo 的单次
`plan_pose` 返回 `None` 表示该次 IK 没有成功种子，不足以单独证明碰撞。接口在失败后额外运行
一次严格 IK，记录每个种子的位姿误差和约束可行性；如果仍无成功种子，再运行一次**只用于诊断**
的无环境障碍 IK，并把它的解放回原场景作严格碰撞检查。诊断解不会进入执行接口。
这次复现中，16 个严格诊断种子均未成功；无环境障碍 IK 可解，但它在原场景与碗壁
`mj_geom_137_bowl_wall_2_0`、`mj_geom_138_bowl_wall_2_1`、`mj_geom_168_bowl_wall_2_31`
相交。这说明碗壁接触是当前目标的明确障碍线索；单个诊断解仍不能证明不存在其他无碰撞 IK 解。
若未来目标确实要求规划阶段接触碗壁，才需要显式声明允许的夹爪 pad—碗接触，
并保持机械臂其他 link 和路径的碰撞检查。

2026-10-06 对照已有 Viser 成功 run 后，发现它的夹爪规划开口约为 `0.08 m`，而本探针之前
只打开到 `0.041 m`。把本探针默认开口改为 `0.08 m` 后，保持候选 `bowl_ring_00_00`、
`depth_m=0.012`、`retreat_m=0.03` 不变，完整命令行运行
`runs/pick-bowl-interface-v8-open80-default/` 成功完成开爪、预抓取、最终接近与闭爪。
最终规划的离散碰撞复核为 `valid=true`，物理闭爪后两侧 pad 与碗壁共有 8 个接触点，
没有记录夹爪非 pad 部位与碗的接触；实际夹爪开口约 `0.02056 m`。本次没有为碗关闭规划碰撞，
也没有实现 pad—碗规划接触例外。上段所述例外只在未来目标确实要求规划阶段贴合碗壁时才需要，
不能作为当前失败的既定根因。闭爪接触是瞬时判定，未检查抬升或保持。

只检查已有状态的图片，无需再运行 GPU 规划：

```bash
MUJOCO_GL=egl PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/capture/render_pick_bowl_stage.py \
  --run runs/pick-bowl-interface-v3 --stage pregrasp \
  --frames runs/pick-bowl-interface-v3/target_frame_review/frames.json \
  --output runs/pick-bowl-interface-v3/target_frame_review/images_v3
```

查看 `images_v3/contact.png`（原图）和 `images_v3/contact_overlay.png`（右侧诊断栏）。
`views.json` 保存相机、状态编号、图像场景边界和数值诊断。当前叠图提供位置差，
但规划器失败原因、接触力和碰撞约束仍需查看对应的 `plans/*.json`、`diagnostics.json`；
仅靠图像无法判断 IK 无解的具体约束。

闭爪状态现在同时保存 `contact` 与 `contact_side` 两个相差 90° 的接触视角。
可用 `tools/viewers/capture/render_pick_bowl_stage.py --run runs/pick-bowl-interface-v8-open80-default
--stage closed --output runs/pick-bowl-interface-v8-open80-default/closed_side_90` 从同一状态重渲染；
两张原图分别为 `contact.png`、`contact_side.png`，接触点只在各自的 `_overlay.png` 中显示。

`capture_failure` 冻结同一状态并保存：

- `failure.json`：阶段、失败类型、消息、建议查询及证据完整性；
- `operational_observation.json`：Pipeline 可见状态；
- `diagnostics.json`：Codex 可见的接触和物理诊断；
- `state.npz`：可恢复的完整 MuJoCo 状态；
- `traceback.txt`：存在 Python 异常时保存；
- `images/`：全局、任务、侧面、局部接触和碰撞叠图。

`runs/pick-bowl-interface-v12-failure-bundle/` 是实际失败样本：候选 `bowl_ring_00_03`
完成规划与闭爪命令，但没有 pad—碗接触。`failures/close_contact/` 自动保存同一
`state_id` 的 operational observation、接触诊断、可恢复状态和 90° 接触侧视图；
`failure.json` 的 `evidence_incomplete=false`。该样本的侧视图可见夹爪停在碗沿上方。

图片失败不会丢掉 JSON 和状态。每项证据记录同一个 `state_id`。输出目录必须不存在，避免新运行
覆盖旧失败证据。

## 两段闭环验收

入口：`experiments/probes/check_closed_loop_motion.py`。

```bash
MUJOCO_GL=egl PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python \
  experiments/probes/check_closed_loop_motion.py --output runs/closed-loop-motion-gpu
```

流程为：

1. fresh scene 查询实际右臂状态；
2. 冻结第一份全场景碰撞快照；
3. 以首关节 `+0.04 rad` 的 FK pose 为见证目标，cuRobo 规划并由 MuJoCo 执行；
4. 查询执行后的实际关节状态；
5. 从该状态重新导出碰撞快照并新建 planner；
6. 使用实测关节作为第二次规划起点，规划回初始 TCP pose并执行；
7. 保存三次 observation、两份 planner 原始结果、两份 execution summary 和最终误差。

2026-10-06 加入微波炉16cm支撑、实际执行录制和分类碰撞叠图后的 RTX 4090 结果保存在
`runs/closed-loop-motion-support-visual-v3/`：两次规划和执行均完成；
第一段后 observation 的 state ID 等于返回段执行使用的 planned state ID；返回 planner 首 waypoint
与传入实测状态最大差约 `3.25e-7 rad`；两份碰撞快照哈希不同。第一段和返回段的最大伺服跟踪误差
分别约 `0.0379 rad` 和 `0.0358 rad`，最终相对初始关节最大误差约 `0.0323 rad`。
因此该结果只证明查询—规划—执行—再查询—重规划的数据闭环与物理伺服确实运行，不证明精确回零，
也不证明抓取或其他 Skill 成功。

该run另保存三组关键帧、每段12帧的正常/碰撞GIF及逐帧`state_id`。这些帧来自
`execute_joint_trajectory`每次稳定`mj_step`后的只读数据副本，不是规划轨迹的离线动画。
碰撞图中机器人为橙色、碗为绿色、微波炉及支撑为蓝色、桌面为灰色。可用同一模型和保存的
实际关节采样做交互诊断回放：

```bash
python experiments/probes/replay_closed_loop_motion.py \
  --run runs/closed-loop-motion-support-visual-v3 --collision-overlay
```

回放是运动学诊断工具，不重新执行动力学或接触。

沙箱环境可能没有 NVIDIA 设备节点；此时 cuRobo 会在初始化时报 `No CUDA GPUs are available`。
CPU 契约测试不初始化 cuRobo，可以独立验证状态流和失败处理：

```bash
MUJOCO_GL=egl PYTHONPATH= PYTHONNOUSERSITE=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest \
  tests/simulation/test_simulation_runtime.py \
  tests/simulation/test_runtime_failure_feedback.py \
  tests/test_runtime_bowl_feedback.py \
  tests/motion/test_closed_loop_motion_contract.py -q
```

## 当前限制

- Runtime 只支持 hinge arm group；线性或复合规划关节需要带单位的版本化 schema。
- 轨迹完成表示时间序列执行完且仿真状态有限，不包含任务自定义的跟踪误差阈值。
- cuRobo 场景是规划时刻的静态快照；任何执行后状态都必须重新导出，当前不会自动更新 planner。
- planner 使用活动臂原有碰撞球，其他臂、机器人本体、桌面和物体来自 MuJoCo 碰撞几何快照；
  mesh 按编译顶点凸包导出。
- 没有 attachment、允许任务接触、抓取保持、阶段恢复或真机执行语义；这些属于后续 Skill。
- Pipeline 当前不使用 RGB、depth 或 mask；相机和精确 contact 仅服务 Codex 诊断。
