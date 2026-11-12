# Knowledge 模块最终设计

日期：2026-10-05
受众：后续负责 Knowledge、Visualization、Motion、Skill 的开发 Agent。
状态：已确认的实现契约；取代 `docs/tmp/history/handoff/KNOWLEDGE_DESIGN_V2.md` 中的讨论方案。

## 1. 目标

Knowledge 把结构化场景表示转换为可解释、可复现的操作候选。第一版只迁移：

- 基础几何 Concept，以及 `trifold`、`container` 扩展 Concept；
- `ring`、`cylinder_axial`、`cylinder`、`cuboidal_block`、`trifold_handle` 五类抓取；
- 旋转旋钮和推门两类 operation knowledge；
- 使用 Viser 检查和调节抓取参数的工具；
- 可选的 TCP pose 可达性筛选接口。

本阶段不接感知，不要求 MuJoCo，不执行 Skill，也不规划到达路径。输入以 GT Concept、显式位姿和任务配置为准。

## 2. 职责边界

### 2.1 Knowledge 自己负责

1. 校验 Concept、坐标系、单位和参数范围。
2. 在物体局部坐标系中生成参数化候选，再且只再应用一次世界位姿变换。
3. 给候选稳定编号，保留生成参数、来源 part、TCP pose、建议闭合宽度和诊断信息。
4. 按配置中显式声明的规则排序和做 pose 后处理。
5. 在调用方传入可达性检查器时，按 IK 结果筛选候选。
6. 对无候选、非法参数和候选被拒绝给出结构化原因。

### 2.2 交给其他模块

| 能力 | 所属模块 | Knowledge 与它的关系 |
|---|---|---|
| mesh/pose/part 的感知结果 | perception/scene | Knowledge 只消费结构化输入 |
| TCP pose 的 IK | motion | 通过可选抽象接口调用 |
| 自身碰撞、场景碰撞、路径规划 | motion | 不属于抓取 Knowledge 筛选 |
| pregrasp、approach、lift、速度、力、重试 | skills | 使用 Knowledge 候选后执行 |
| 接触、闭合、夹住和抬升物理 | simulation/skills | 在 MuJoCo 阶段验证 |
| 任务顺序和 Skill 编排 | tasks | Task 绑定 profile 和 Skill |
| UI、截图和人工调参 | visualization | 只调用公开 Knowledge API |

“可达”在本模块中有严格含义：目标 TCP pose 存在满足关节限位的 IK 解。它不代表 pose 无碰撞，也不代表机械臂能沿无碰撞路径到达。

### 2.3 明确不做

- 不直接 import cuRobo、MuJoCo、Viser 或机器人 SDK。
- 不根据场景障碍物筛选抓取，不生成路径。
- 不实现“更容易够到”的通用评分。
- 不保留旧的最小抓取深度、手指长度或对向表面深度门槛。
- 不从 Concept 猜门轴；门轴由用户或任务配置显式提供。
- 不迁移其他旧抓取类型，不保留通用 callable filter/scorer 框架。
- 不为旧接口建立长期兼容层；旧工程保留作参考。

## 3. 输入与输出

### 3.1 抓取输入

```python
generate_grasps(
    concept: Concept,
    object_pose: Pose,
    profile: GraspProfile,
) -> GraspGenerationResult
```

- `concept`：物体或 part 的结构化几何；尺寸单位为米。
- `object_pose`：`T_world_object`，明确四元数顺序和单位。
- `profile`：任务目录加载的已解析配置；参数角在文件中使用度，进入生成器前转为弧度。

统一位姿链为：

```text
T_world_tcp = T_world_object @ T_object_part @ T_part_tcp
```

生成器优先计算 `T_part_tcp`，不得在中途重复应用 object pose。

### 3.2 抓取输出

```python
@dataclass(frozen=True)
class GraspProposal:
    candidate_id: str
    generator: str
    part_name: str
    tcp_pose_world: Pose
    closing_width_m: float
    parameters: dict[str, float | str]

@dataclass(frozen=True)
class CandidateDiagnostic:
    candidate_id: str | None
    code: str
    message: str
    context: dict[str, object]

@dataclass(frozen=True)
class GraspGenerationResult:
    proposals: tuple[GraspProposal, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]
    resolved_profile: GraspProfile
```

要求：

- 同一输入产生相同顺序和 `candidate_id`。
- 不用裸 `except` 静默丢候选。
- 非法 profile 直接报带字段路径的配置错误；某个离散样本无效则记录诊断并继续其他样本。
- `depth_m` 仍可作为 TCP 沿接近轴的位置参数，但不作为物体厚度或手指长度的硬约束。
- `closing_width_m` 是 Knowledge 根据几何推导或由 profile 显式覆盖的目标闭合宽度。

### 3.3 可选可达性接口

```python
class PoseReachabilityChecker(Protocol):
    def check(
        self,
        tcp_poses_world: Sequence[Pose],
        *,
        robot_group: str,
        seed_joint_positions: JointState | None = None,
    ) -> Sequence[PoseReachabilityResult]: ...
```

`PoseReachabilityResult` 至少包含 `reachable`、可选 IK 解、原因码和耗时。Knowledge 只定义协议与筛选函数，不提供 cuRobo 实现。没有 checker 时，候选生成和 Viser 调参必须完整可用。

### 3.4 Operation 输入输出

- 旋钮：输入显式旋转轴、当前父 frame 位姿、抓持/接触参数和角度序列；输出带顺序的 TCP pose 或操作 waypoint。
- 推门：输入 `DoorAxis(point, direction, frame)`、接触点、推行方向/角度和离散参数；输出候选或 waypoint。
- 缺失门轴必须报错，不允许从矩形 Concept 的边或名称猜测。

## 4. 文件结构

```text
gene_agent/knowledge/
├── __init__.py
├── geometry/
│   ├── __init__.py
│   ├── basic.py              # Cuboid/Cylinder/Ring 等基础 Concept
│   ├── transforms.py         # Pose、frame 与变换校验
│   ├── loader.py             # 从结构化数据加载 Concept
│   ├── trifold.py            # 三段式把手等扩展 Concept
│   └── container.py          # 容器扩展 Concept
├── grasp/
│   ├── __init__.py           # 小而稳定的公开入口
│   ├── model.py              # profile、proposal、diagnostic 数据类型
│   ├── profile.py            # JSON 解析、校验和任务内绑定
│   ├── sampling.py           # 确定性范围展开和 center-out 顺序
│   ├── reachability.py       # 可选 Protocol 和筛选编排
│   └── generators/
│       ├── __init__.py       # 显式注册五类生成器
│       ├── common.py         # 小量共享坐标和宽度计算
│       ├── ring.py
│       ├── cylinder.py
│       ├── cylinder_axial.py
│       ├── cuboidal_block.py
│       └── trifold_handle.py
└── operation/
    ├── __init__.py
    ├── knob.py
    └── door.py
```

`ring.py` 等具体策略只位于 `generators/`，不和 `model.py`、`profile.py` 并列。只有两个生成器真正共享且语义一致的逻辑才进入 `common.py`。

## 5. Concept 设计

### 5.1 基础 Concept

`geometry/basic.py` 集中放置简单几何：

- `CuboidConcept(size_xyz_m, pose_in_object)`
- `CylinderConcept(radius_xy_m, height_m, pose_in_object)`；`radius_xy_m` 保存 part 局部
  X/Y 两个有限正半径，允许椭圆截面，两个半径相等时为圆截面特例；高度轴严格固定为
  part 局部 `+Z`，整体方向只由 `pose_in_object` 表达
- `RingConcept(inner_radius_m, outer_radius_m, height_m, axis, pose_in_object, angle_span_rad=2π)`；
  `angle_span_rad` 是从 part 局部方位角 0 开始存在的弧长，范围为 `[0, 2π]`。
  0 表示空弧，`2π` 表示完整圆环，不另设起始角。legacy JSON 的
  `exist_angle[0]` 使用度，loader 校验为 0–360° 后转换为弧度。
- 若现有场景确实使用，再加入 `SphereConcept`；不预建未使用类型。

每个 Concept 只描述几何事实和 frame，不包含抓取偏好、机器人限制或可视化对象。

### 5.2 扩展 Concept

- `TrifoldConcept` 表达把手中段和两侧连接段的几何关系。微波炉把手使用它，但生成器命名为通用 `trifold_handle`。
- `ContainerConcept` 表达开口、内外轮廓和容器轴；第一阶段只提供结构和加载，不因此引入双臂容器抓取。

## 6. Grasp profile

### 6.1 所有权

profile 属于具体任务数据，放在：

```text
data/task_1/
├── grasp_profiles.json       # Knowledge 配置
├── skill_profiles.json       # Skill 迁移时创建和消费
└── task.json                 # Task 迁移时绑定二者
```

Knowledge 和 Skill 文件可同属一个 task 目录，但职责必须分开。第一阶段只创建实际被 Knowledge 工具读取的 `grasp_profiles.json`；不提前制造无人消费的 Skill 配置。

### 6.2 Knowledge 字段

```json
{
  "profiles": {
    "task_1_bowl_0": {
      "generator": "ring",
      "part": "bowl_rim",
      "parameters": {
        "surface": ["bottom"],
        "azimuth_deg": {"start": 0, "stop": 315, "step": 45},
        "tip_y_deg": {"values": [0, -10, 10, -20, 20]},
        "depth_m": {"values": [0.015]}
      },
      "closing_width": {"mode": "geometry", "clearance_m": 0.0},
      "postprocess": [
        {"type": "translate", "frame": "object", "xyz_m": [0.0, 0.0, 0.0]}
      ]
    }
  }
}
```

允许：

- 生成器名和 Concept part；
- 离散值或有界范围；
- 明确顺序，包括 center-out；
- 闭合宽度的几何计算、clearance 或显式覆盖；
- 少量可序列化 pose 后处理，且必须声明 `object` 或 `tcp` frame。

禁止放入：

- `pregrasp_opening_width_m`、approach/lift 距离；
- 速度、加速度、力、等待、重试；
- 碰撞策略、allowed collision；
- planner、机器人型号或 SDK 字段；
- 任意 Python callable 或表达式字符串。

### 6.3 旧数值行为

- cylinder axial、cylinder side、cuboidal block、trifold handle 尽量保持旧实现的候选数值、坐标方向和枚举顺序。
- 明确删除的深度门槛和移到 Skill 的字段不保留。
- Bowl 的旧 `world -X/-Y` 偏置和旧 profile 不作为正确基线；使用 Viser 重新调节并建立新基线。
- 旧 `convert_gripper_axes` 反映过往 gripper opening axis 差异。新实现统一声明 TCP `+X` 为 opening axis、`+Z` 为 approach axis；通过数值 fixture 保持需要的行为，不复制不透明补偿函数。

## 7. 五类抓取生成器

### 7.1 Ring

围绕环形 part 的方位角、表面、tip 倾角和深度生成局部候选。支持确定性 center-out 倾角顺序。
部分圆弧只为 `[0, angle_span_rad]` 内的方位角生成候选；完整圆环按周期解释方位角。
越界离散组合生成带稳定 candidate ID 的 diagnostic 并继续其他组合，空弧返回零 proposal
和清楚 diagnostic。Bowl 使用此生成器，但 profile 重新调节。

### 7.2 Cylinder axial

沿圆柱轴端面抓取，主要参数为端面、方位角、tip 倾角和深度。旋钮抓持可复用，不把 knob 语义写入生成器。

### 7.3 Cylinder side

从圆柱侧面生成抓取，保留旧局部轴方向和宽度推导。旋转轴、接近方向和 opening axis 必须能由坐标测试验证。

### 7.4 Cuboidal block

基于显式面和面内参数生成候选。只支持当前迁移需要的参数，不建立任意多面体抓取框架。

### 7.5 Trifold handle

使用 `TrifoldConcept` 的中段作为抓取段，连接段用于确定结构方向和可用范围。微波炉 handle 通过任务 profile 调节 vertical offset、tip 角、深度和局部后处理；生成器本身不包含 microwave 名称或固定尺寸。

## 8. Viser 与自动反馈

Visualization 位于 Knowledge 外：

```text
gene_agent/visualization/grasp_tuner/
├── app.py
├── controls.py
├── scene_view.py
└── session.py
experiments/inspect_grasp_profile.py
```

### 8.1 第一版交互

- 加载 object mesh、Concept 和 task profile；
- 显示 object/part/TCP 坐标轴、gripper visual 和所有候选；
- 按生成器暴露参数范围、步长、surface、排序和后处理控件；
- 单步浏览候选，显示 `candidate_id` 和生成参数；
- 在适用时允许在 mesh 上点选 part/面或接触区域；
- 原子写回 task 的 profile，并显示修改前后 diff；
- 不支持拖动 gripper 后反求参数。

### 8.2 Headless 检查

`inspect_grasp_profile.py` 与 Viser 必须调用同一个生成入口。每次输出：

```text
runs/knowledge/<run_id>/
├── resolved_profile.json
├── proposals.json
├── diagnostics.json
├── summary.json
├── views/                  # 固定相机截图
└── reproduce.sh            # 实际使用的命令
```

终端只显示候选数量、关键拒绝原因和产物目录。完整数据进 run 目录。若传入 reachability checker，再追加 IK 结果；默认运行不得 import cuRobo。

## 9. 自动验收标准

Knowledge 完成必须同时满足：

1. 单元测试无需 GPU、MuJoCo、Viser server 和 cuRobo。
2. 五类生成器都能由任务 profile 生成稳定候选。
3. 除重新调节的 Bowl 外，迁移 fixture 在容差内保持旧数值行为。
4. 对旋转和平移后的物体，位姿链测试证明 object pose 只应用一次。
5. 输入和顺序相同则 candidate ID、候选顺序与结果相同。
6. `depth_m` 不触发旧抓取深度硬门槛；非法数值、退化几何和非法 pose 仍明确报错。
7. grasp profile 出现 Skill 字段时给出字段路径清楚的错误。
8. 不提供 checker 时生成器和 Viser 正常工作；fake checker 能验证可选筛选协议。
9. 旋钮旧数值行为通过回归；推门缺少显式门轴时必定失败并给出原因。
10. Headless 与 Viser 对同一 profile 产生相同候选摘要。
11. 人工可在 Viser 中检查每类候选的 opening/approach axis，并保存 Bowl 新 profile。

## 10. 阶段完成后仍不代表什么

Knowledge 验收通过只证明几何候选和参数工具正确。它不证明：

- 候选无碰撞或存在无碰撞到达路径；
- 夹爪能在 MuJoCo 中夹紧、抬升或保持物体；
- 微波炉摆放对当前机械臂可达；
- 感知噪声下候选稳定；
- 真机坐标、TCP、力和接触行为正确。

这些结论分别在 motion、Skill + MuJoCo、Task 和真机阶段建立，不能用 Viser 画面代替。
