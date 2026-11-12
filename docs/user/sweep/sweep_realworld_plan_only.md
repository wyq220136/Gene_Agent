# 刷子清扫：真机接入与只规划预览（旧刷优先）

## 当前能做什么

默认使用 **原 `brush_0` 重建的蓝色旧刷 + Pika 单臂、人工已夹稳刷子、一个方块、固定簸箕、水平桌面**。
它是 `data/realworld/brush_cube_dustpan_0/assets/brush.obj` 对应的旧刷，不是 `brush_small`。
白柄长刷仍支持，通过另一份资产配置选择；两种刷子调用同一个 Sweep Skill。
使用同帧 FoundationPose、只读 SDK 状态与显式坐标配置构建现场场景，调用已有
Sweep Knowledge、Sweep Skill 和 cuRobo，保存接近、直线下降、直线清扫、直线撤离四阶段的预测轨迹。

**没有真机运动开关，没有自动抓刷，也没有实机清扫成功记录。**
轨迹预览中刷子按本次确认的固定夹持关系移动，方块保持观测位置；不会人为演示入盘成功。
当前相机输入支持已标定的**固定相机**。手腕相机需要后续接入随机器人 pose 更新的外参，不能填一个固定矩阵替代。

## 立即查看已验证的离线样例

项目根目录，`curobo` 环境：

```bash
python tools/viewers/view_skill.py --skill sweep_cube \
  --run runs/realworld_sweep/old_brush_20261023

# 同一记录也可用 Task 入口打开
python tools/viewers/view_task.py --task sweep_cube_into_dustpan \
  --run runs/realworld_sweep/old_brush_20261023
```

分别打开 `http://127.0.0.1:8092` / `http://127.0.0.1:8093`，拖动 `Plan sample` 检查规划。
勾选 `Show target TCP frames` 可以对照各阶段期望目标坐标轴。
这是 **synthetic_fixture**，取自历史 MuJoCo 抬刷后的已保存状态，既不是当前拍摄，也不证明实机夹持。
可先看[旧刷初态全景](../../../runs/realworld_sweep/old_brush_20261023/previews/initial_overview.png)、
[初态俯视](../../../runs/realworld_sweep/old_brush_20261023/previews/initial_top.png)、
[下降终点](../../../runs/realworld_sweep/old_brush_20261023/previews/contact_plan.png)。

旧刷样例的第一个候选已通过真实 GPU cuRobo 四阶段规划与完整 world 几何后检查；
165 帧是规划 FK，**不计入任何仿真/真机成功率**。模型使用旧刷原有的 VHACD 碰撞体。
viewer 根据运行记录类型区分只规划预览和原有 MuJoCo 回放，沿用原 Skill/Task 名称。

新输出目录重新复现（需要 GPU）：

```bash
python tools/run/run_sweep_realworld.py --demo --output runs/realworld_sweep/my_demo
```

只构建初态、不调用 GPU：

```bash
python tools/run/run_sweep_realworld.py --demo --scene-only \
  --output runs/realworld_sweep/my_scene
```

截图命令：

```bash
python tools/viewers/view_task.py --task sweep_cube_into_dustpan \
  --run runs/realworld_sweep/old_brush_20261023 --render-only
```

### 可选：切换回白柄长刷

```bash
python tools/run/run_sweep_realworld.py --demo \
  --config config/realworld/sweep_long.example.json \
  --output runs/realworld_sweep/my_long_demo

python tools/viewers/view_skill.py --skill sweep_long_brush \
  --run runs/realworld_sweep/my_long_demo
```

长刷已验证的历史只规划记录仍在 `runs/realworld_sweep/demo_checked_20261023`。
`sweep_realworld` 仅兼容先前的预览命令，不是新建的 Skill。

## 现场需要用户配合什么

1. **设备和安装确认**：提供/确认机械臂 IP、实际型号、Pika 安装方式；协助用多组姿态检查法兰与模型的对应关系。名义 17.2 mm 偏移尚不能当成实测安装标定。
2. **相机与标定**：固定相机，调整到能看见刷子、方块、簸箕的位置；提供序列号。已有有效相机→机器人外参可复用，否则配合摆放标定板、采集标定数据。
3. **实际物体和尺寸**：优先准备对应的蓝色旧刷、簸箕、一个方块。样例方块边长约 22.4 mm；现场不同尺寸应更新资产和 Concept，不能只改显示比例。
4. **现场准备**：先人工夹稳刷柄，量一下当前内侧开口；把方块和簸箕放在机械臂前方可达区域，簸箕口朝方块。保持桌面及相机静止，记录其他障碍物。
5. **确认预览**：检查 mesh 是否对齐实物、夹持关系是否合理、下降高度是否正确。完成这轮只规划检查后，另行授权低速真机运动。

配置整理、图像采集、分割 prompt 尝试、三个 Pose session 注册、mesh frame 对齐与规划复核可由 Agent 处理。
用户不需要手写矩阵；但不能用修改 `site_verified` 字段代替实际现场核验。

## 现场配置和采集命令

默认旧刷模板：[sweep_old.example.json](../../../config/realworld/sweep_old.example.json)；
可选长刷模板：[sweep_long.example.json](../../../config/realworld/sweep_long.example.json)。
具名坐标：[pika_frames.nominal.json](../../../config/robots/pika_frames.nominal.json)。
旧刷现场版本分别保存为 `sweep_old.site.json`、`pika_frames.site.json`，已加入忽略规则。

坐标约定：`T_A_B` 把 B 映射到 A，长度米、关节弧度。

- `T_sdk_urdf_base`：SDK base 中的当前单臂 URDF base。
- `T_sdk_camera`：固定相机在 SDK base 中的外参。
- `T_sdk_table`：桌面 **box 中心**，不是桌面顶面；尺寸另给 `table_size_xyz_m`。
- 每个物体 `T_perception_asset`：FoundationPose mesh frame 到当前场景资产 frame 的组合变换，满足 `T_world_asset = inv(T_sdk_urdf_base) @ T_sdk_perception @ T_perception_asset`。同一 mesh、同一原点也须明确注明来源。
- frame profile 分别保存 `link_7→SDK flange`、`link_7→Pika model`、`Pika model→operation TCP`、`SDK flange→reference control`。控制参考点不自动充当操作 TCP。

静态检查不连接设备：

```bash
python tools/run/run_sweep_realworld.py \
  --config config/realworld/sweep_old.site.json --check-config
```

完成现场核验、已有三个 Pose session、相机服务开启后：

```bash
python tools/run/run_sweep_realworld.py \
  --config config/realworld/sweep_old.site.json \
  --capture --held-brush-confirmed --scene-only \
  --output runs/realworld_sweep/site_scene_01
```

此命令只读机器人并采集图像，不使能或驱动 Pika。`devices.sdk_python` 可指定装有厂家 SDK 的独立环境，
例如 `/home/ubuntu/miniconda3/envs/pika/bin/python`；省略则当前环境必须能 import SDK。
开口使用 `devices.measured_gripper_width_m` 的本次测量值，不假装已经接通 Pika 开口反馈。

同一命令移除 `--scene-only` 即采集新输入并只规划；也可用 `--snapshot <observation.json>`
读取仍有效的现场快照。默认帧龄上限 120 s，快照与当前配置/标定哈希必须匹配，过期就重新采集。
采集前后读取两次关节，变化大于 0.005 rad 即拒绝该帧。第一次仍只查看场景，之后再规划。

## 验证边界与记录

- 真实输入必须有已核验的 site/profile、同帧 pose 和本次人工夹持确认。当前物理观察与标定尚未现场验收。
- 保留由 **SDK 法兰 pose + profile 操作 TCP + 感知刷子 pose** 得到的原始夹持关系；另外检查模型 FK 与 SDK 法兰误差，默认限值 3 mm / 1°。
- 当 SDK 与模型 FK 存在上述预算内的残差时，将同一个观测刷子显式注册到模型 TCP，供 Knowledge、附件、预测及碰撞复核统一使用，原始关系与 `T_model_fk_tcp_observed_sdk_tcp` 都保存。这样不会在相同关节下跳动刷子；这项本帧注册**不是**机器人 DH 标定或开环执行许可。已加入 SDK 高 2 mm 的回归样例。
- 后续阶段只使用上段规划 FK 做前瞻，记录 `previous_plan_predicted_fk`；未来执行接入必须重新读取实际状态。
- cuRobo 粗规划按阶段移出允许刷子接触的目标代理，完整世界几何仍进行密集 FK 复核；其他机器人 link 碰这些物体仍拒绝。刷子穿簸箕侧/后壁也拒绝。
- world 几何穿透检查阈值 1 mm；允许刷子接触桌面时最大模型穿透 2 mm。它是几何容差，不是已经验证的真实刷毛压缩量或力控。
- 开始规划前检查原始初态，拒绝已存在的非白名单 world 接触；碰撞检查记录实际检查的样本数和原计划样本数。
- 当前严格 IK，无 CPU IK fallback。`--execute` 不存在；程序不会发送机器人或夹爪运动命令。
- 输出含 `site_input.json`、`observation.json`、代码哈希、`scene/provenance.json`、初态 NPZ、每阶段完整/过滤碰撞清单、原始 GPU 返回、预测 qpos 和失败 traceback。

## 代码边界

`realworld/sweep/` 负责输入、现场资产适配与只规划接线；`knowledge/sweep.py` 和 `skills/library/sweep/`
继续负责共享几何和阶段顺序。刚体边界计算集中在 `scene/rigid_geometry.py`，仿真旧入口复用同一实现。
新 profile 只有显式传给 `build_pika_robot` 才生效，原仿真默认配置不变。

## 桌面首帧自动入口

如果刷子仍在桌上，希望一次命令自动拍照、分割、首帧估计并预览抓刷与清扫，使用[首帧预览说明](sweep_first_frame_preview.md)。本页的 `--capture` 是已人工持刷、已有 Pose session 的旧接口；新入口不要求手动注册这些 session。
