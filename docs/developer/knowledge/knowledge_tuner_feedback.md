# Knowledge 调参界面反馈契约

受众：后续开发 Codex。状态：2026-10-07 已实现调参状态记录、真实 visual mesh、夹爪近景、物体预览位姿与纯 IK 检查。

`GraspParameterControls` 负责固定值、范围、列表的交互转换；范围的 `order` 保留在读取结果中。范围改为列表时调用正式 `expand_sample_spec`，列表改为范围时仅允许升序等差列表，避免改变候选集合。无效输入不更新 `EditableGraspSession`，界面保留上一次有效候选并显式提示。候选选择优先匹配生成参数，否则按 TCP 几何距离选近邻，不依赖会随采样索引变化的候选 ID。

运行 `view_grasp_knowledge.py` 或 `tune_scene_grasp.py` 时传 `--state-out`，每次成功重生成、候选切换、视角切换及错误都会原子写入 JSON。场景状态至少包含 `scene_revision`、`scene_snapshot_sha256`、object pose、task profile 及其哈希、候选 ID/pose/参数、相机预设和有效性。Lab 状态包含 Concept 几何、采样参数与候选。`capture_knowledge_viser.py --url ... --state-file ...` 连接现有 Viser 服务；脚本在截图前后比较状态文件内容，拒绝参数与图片不一致的产物。输出的截图清单嵌入该状态并保存图片、状态哈希。

场景预览中的黄色夹爪 mesh 从当前 MuJoCo 模型编译的 visual 几何提取，姿态相对于当前组的 TCP site 计算。实验台也复用同一 visual 提取器，默认隐藏大掌部以露出 Concept；缺少 run 时才使用简化代理。场景默认显示完整夹爪；掌部由“显示夹爪掌部”开关按需创建或删除；初始开口取冻结快照，选中候选后按其 closing width 更新 MuJoCo 指尖关节和 visual 相对 TCP 位姿。预览根 frame 必须设置 `show_axes=False, origin_radius=0.0`，否则 Viser 会在 TCP 画一个巨大的黄色原点球，遮挡碗并误导判断。单个夹爪画面只能说明几何关系；纯 IK 检查另由显式按钮触发，仍不检查碰撞、路径或物理夹持。简化方块代理可由“显示夹爪 visual mesh”关闭后查看。Gripper 近景隐藏当前机器人 visual，只显示目标物体与黄色候选夹爪；用户可以再打开实际机器人。

两个界面均不再叠加蓝色启动基线。实验台在 `GEOMETRY_DEFAULTS` 中按 Concept 类型定义尺寸初值：轴向圆柱复用现有旋钮几何，侧面圆柱与长方体采用原可视化工具的示例尺寸；`_geometry_by_kind` 与 `_parameters_by_kind` 保留用户切换类型前的有效编辑。全部候选用批量点和短线显示 TCP 位置与接近方向；当前候选仍显示真实夹爪。相机在修改参数时固定，只有切换视角或按“对焦当前候选”才重新定位。拖动碗时只变更 visual 和调参 session 的 `object_pose_world`，不变更冻结的 MuJoCo run 或碰撞快照；状态记录 `object_pose_source` 区分原始 GT 和编辑预览。纯 IK 在 `pose_ik.py` 中用 MuJoCo site 雅可比、关节限位及确定性多初值求解，没有查询碰撞；每次参数或碗位姿变化立即清除旧结果，用户需重新点击检查。截图清单记录逐候选 IK 布尔值和误差，可用 `capture_knowledge_viser.py --tool scene --check-ik` 自动验收。它只能说明单个 TCP 位姿的运动学收敛，不能证明无碰撞、可到达路径或夹持成功。

截图进程除了等待 WebGL ready，还检查青色碗与黄色真实夹爪的像素是否出现；Bowl 视角只在避开机器人本体的固定区域检查金色，gripper 侧视则检查整个三维画面，避免固定视区造成误报。材质上传未完成时继续截图，超过 15 秒报错。场景相机按当前候选的双指内侧 site 计算开合轴，在水平面取垂直该轴的侧向视线，且相机高度必须高于桌面。浏览器 `camera.on_update` 同步实际 position、look_at、up、fov 到状态文件；截图前后状态哈希相同才接受，因此手动改动视角也能被复现。

场景默认显示一个完整选中夹爪和一个浅紫色代表夹爪的真实手指，其余候选图层隐藏。代表候选由 `representative_candidate_indices` 按 TCP 位置与旋转的最远点规则确定；默认数量上限为 12，用户可把上限调到候选总数减一。它们只影响可视化，不参与 Knowledge 筛选。ghost `GripperMeshPreview` 按数量懒创建，减少默认页面的渲染负担；数量降低后隐藏多余实例，再次增大时复用。点击手指会切换当前候选。全量点/接近方向线缩小后按需显示，IK 检查后自动显示红绿点。截图状态包含 `representative_count`、`representative_limit`、`representative_indices`、`show_all_candidates`、`bowl_drag_visible`，恢复时校验代表序号。用户可以把代表数量调为零，仅观察一个真实夹爪。

`read_tuner_state` 可从实时 JSON 或截图清单读取有效状态。两个入口都支持 `--restore-state`；场景 tuner 在构建 Viser 之前验证目标、scene snapshot hash 和 profile，重建后比对候选 ID 与 TCP 矩阵。Lab 恢复 Concept 几何、采样规格和选中候选。旧版 Ring 同时包含深度与轴向偏移的实验台状态不再可恢复；它没有唯一的参数含义，需重新调参生成新清单。输入无效的快照拒绝恢复，因为它没有保存无法通过 Knowledge loader 的草稿。

回归命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest \
  gene_agent/visualization/grasp_tuner tests/knowledge -p no:cacheprovider -q
```

实际截图产物位于 `runs/knowledge_tuner/`；请用图像查看工具打开 PNG，不能只检查文件是否存在。
