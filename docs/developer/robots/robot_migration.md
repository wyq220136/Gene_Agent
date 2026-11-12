# Realman MuJoCo 迁移记录

受众：开发 Agent。状态：2026-10-04 已实现双臂模型导入与 FK 验证；夹爪保持 fixed，尚未验证物理抓取。

## 实施与接口

实施顺序：冻结原 URDF/mesh/安装和夹爪配置 → 原生 MuJoCo URDF 导入 → 组合双臂、本体、TCP → 独立 FK 和旧 SDK 记录对照。没有修改旧项目，也没有调用真机 SDK。

```python
from pathlib import Path
from gene_agent.robots import build_robot
info = build_robot(Path("runs/robot"))
```

生成 `robot.xml`、两份中间 `*_imported.xml`、`robot_metadata.json`。
`robot.xml` 可独立编译，也可通过 MuJoCo `<include file="..."/>` 引用。
XML 使用绝对 mesh 路径，依赖本仓库资产；移动仓库后重新 build。
返回 `xml_path` 为字符串，`joint_names` 为具名关节列表，`joint_groups` 为组到关节列表的映射，
`tcp_sites` 为组到 TCP site 名的映射，`home_joint_positions` 为关节名到弧度的映射。
调用者按名称查询 qpos/actuator，不依赖七轴或者 right 组。
当前具体 Realman 资产有 left/right 两组共十四个运动关节；四个夹爪关节均为 fixed。

## 坐标依据

默认 world = right SDK base。可显式指定 `world_frame="installation_body"` 作诊断，场景组装统一使用默认。
定义 `B` 为安装 body、`S` 为各臂 SDK base、`C` 为 URDF/cuRobo base、`E` 为 URDF link_7、`F` 为 SDK flange：

- `T_W_B = inv(T_B_Sright)`。
- `T_B_C = T_B_S @ inv(T_C_S)`，来自旧 `RealmanInstallation.body_from_curobo`。
- `T_E_TCP = inv(T_F_E) @ T_F_TCP`。
- `T_W_TCP(q) = T_W_B @ T_B_C @ FK_URDF(q) @ T_E_TCP`。
- `T_F_TCP` 来自 standard_gripper：Z 平移 0.15 m、Z 旋转 180°。
- `T_F_E` 来自 installation 的 sdk_flange_to_curobo_frame，Z 旋转 90°。

旧 `tamp/frames.py:command_flange_to_planning_pose` 实际计算 `T_C_S @ T_S_F @ T_F_E`，
`controller._prepare_tcp_pose` 先右乘 `inv(T_F_TCP)`，故上述 TCP 链与调用一致，不另加经验旋转。
metadata 保留每组 world/base 和 link/TCP 变换以及 world/body 变换。

本体 mesh 与 collision JSON 均沿资产本地 Z 平移 -0.5 m，来自旧
`visualization/scene/curobo_demo.py:305,310` 和 `tamp/curobo/robot_body_collision.py:107` 的实际调用，
不是根据截图猜测的偏移。它在新 `config/robots/realman.json` 中显式保存。

## 保真与近似边界

使用 `mujoco.MjModel.from_xml_string` 原生读取冻结 URDF，打开 visual，关闭 fixed link 合并。
MuJoCo XML 导出默认六位有效数字，因此直接从原 URDF 回填各 joint 的 origin 和 axis，避免链式截断误差。
保留原 URDF 的 1.5708/3.14 等近似值，未替换成数学精确值。惯性由原生导入处理，XML 序列化有限精度。
原安装矩阵有限精度导致轻微非正交，四元数表示会投影到合法旋转；原矩阵保持不变且进入报告。

visual geom group=1、collision geom group=3；隐藏 group=3 仅影响显示，不禁用接触。
每个 geom 都具有 link/用途名称，方便接触诊断。没有复制 cuRobo 大量 self_collision_ignore 到物理场景。
机器人 mesh 物理接触使用 MuJoCo 默认凸包，本体使用原 cuboid；不能把它称为已验证精确接触几何。
position actuators 的 kp=150、kv=20、阻尼=1、armature=.02 是显式仿真参数，非实机辨识。

夹爪原 URDF 是固定开口；0.08m 是配置的最大开口能力，不代表导入网格真实开口。
本轮不编造 slider、传动比或指尖尺寸，不能用当前固定夹爪验收抓碗。
需下一阶段核对开合运动学/指尖测量，再增加明确的驱动近似。

## 验证证据

环境：`PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python`，MuJoCo 3.14.0，无 GPU 计算。

`python -m pytest tests/robots/kinematics/test_robot_model.py -q` 覆盖：

- 冻结资产 SHA256 校验（`assets/robots/realman/provenance.json`）。
- zero、记录 home、六组固定 seed 范围内采样，双臂全部 link 和 TCP 对比。
- 对照算法独立解析 URDF，用 Rodrigues 公式逐关节组合，不调用构建器的旋转 helper 或 MuJoCo FK。
- 所有 link/TCP 位置和旋转矩阵元素误差在 2e-9 容差内。
- include 编译、具名关节/actuator 映射和 fixed gripper 契约。
- 旧默认姿态记录的 SDK TCP 外部锚点：左位置差 4.29 µm、旋转差 0.000223 rad；
  右位置差 19.28 µm、旋转差 0.000187 rad；回归阈值为 0.1 mm / 1 mrad。

home 中已观察到本体与手臂约 0.06 mm 轻微穿透，另有接近零距离的 mesh 面接触；
未通过删碰撞或移动安装消除。需场景后续记录这些接触与执行稳定性，不能宣称 home 完全无碰撞。
当前 FK 验证不等同于动力学、规划碰撞或真机行为验证。

冻结的 `assets/robots/realman/config` 为原 cuRobo 球配置的证据，当前 MuJoCo 不加载它。
`config/robots/standard_gripper.json` 保留旧路径字段以保持源哈希；当前构建只使用其 TCP/几何数值字段，
资产路径由本模块定位到新仓库，运行不读取旧项目。
