# 微波炉 Knowledge 的 Viser 调参

在 `curobo` 环境和项目根目录运行：

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
python -u tools/editors/view_microwave_knowledge.py --mode handle_grasp --port 8082
```

侧栏“IK 求解器”可在 `mujoco`（默认的无碰撞位姿 IK）和 `curobo`（位姿、关节限位与机器人自身碰撞 IK）之间切换。两者在 Knowledge 页面都不检查场景物体碰撞。也可从命令行直接打开 cuRobo 模式：

```bash
python -u tools/editors/view_microwave_knowledge.py --mode handle_pull --waypoint 0 --ik-backend curobo --port 8082
```

cuRobo 模式按当前夹爪开口同步机器人球，但传给求解器的场景障碍为空；显示严格 IK、关节/自身约束可行性、位姿误差及失败类型。启用“近似 IK 候选”后，只有这些约束可行且误差在预算内的 cuRobo 候选才会被标记。这里仍只检查单个关节姿态；通过 IK 不表示与微波炉等场景物体无碰撞，也不表示存在可执行轨迹或能实际开门。拖动参数时 GPU 检查可能需要数秒；“检查全部”会逐个检查路径点。

cuRobo 严格 IK 失败时，页面另外检查一组种子，优先把通过关节限位和自身碰撞检查、且实际 world TCP 误差较小的姿态画在机械臂上；这是**诊断预览**，不改变严格 IK 失败结论，也不表示 Skill 能执行。文字会分别说明画面候选与原求解返回的失败种子，避免把原求解的第一个失败姿态误称为“最接近的解”。“跨求解器复核”还会在同一目标上求 MuJoCo IK；无论它是否达到严格阈值，只要返回了候选关节姿态，就用 cuRobo FK、关节限位与自身碰撞再次检查。因此 MuJoCo 候选看起来更接近目标时，也能看到它是否自碰。`--state-out` 保存的状态包含 `selected_tcp_pose_world`、全部滑块值和 `ik_diagnostic`，可用于复现具体分歧；仅提供四舍五入后的 TCP 位置不能确定完整姿态和场景摆位。

浏览器打开 `http://127.0.0.1:8082`。侧栏“检查内容”可切换 `handle_grasp`、`handle_pull`、`door_push`、`knob_grasp`、`knob_turn`。界面只显示当前 Knowledge 自己的参数；“操作目标”标题保留，并在推门时显示门的起止角、在旋钮旋转时显示旋钮目标角。切换 Knowledge 不会清除其他参数已经调整的数值。把手抓取可调沿轴偏移、两轴倾角和深度；门板推压沿用旧项目的逐角候选策略，零偏移即旧参数，右臂默认开门 `30°→85°`；旋钮抓取可调全部候选的方位角偏移、深度和倾角。滑块变化会重建 Knowledge 并只检查当前 TCP 的纯 IK，以保持拖动响应；点击“检查全部候选与路径 IK”才计算整段。绿色路径表示全部 waypoint 可解，橙色表示至少一个未求得，灰蓝色表示整段尚未检查。旧 profile 与冻结 run 不会被写入。旋钮模式可切换 6 个候选；操作模式可拖动 waypoint。全景开关可切回部件近景。

查看从预抓取到拉门的多个 pose：

```bash
python -u tools/editors/view_microwave_knowledge.py --mode handle_pull --port 8082
```

`handle_pull` 是独立的把手拉门 Knowledge。它保留 Task1 的局部 Y `0,+5,-5,+10,-10,+15,-15°` 与局部 X `0,+10,-10,+20,-20,+30,-30°` 有序候选，并剪除让指尖更朝门板的 X 旋转；当前校准场景生成 28 个候选。“把手路线候选序号”可手动切换；页面不自动声称已按整路 IK 选中 fallback。默认路线显示 8 个 waypoint：预抓取（沿 TCP 局部 −Z 后退 30 mm、夹爪开口 80 mm）、到达把手、闭爪（15 mm），然后按旧 Task1 的三个等分点拉到约 `8.3°、16.7°、25°`；每个拉门点还保留旧实现的 TCP 局部 X 负门增量角姿态补偿，最后松爪并撤离。撤离默认沿末端局部 −Z 15 mm、不抬高；局部后退和世界 +Z 抬高距离都可调。通过“路径 waypoint”滑块逐点查看机械臂、夹爪和门；勾选“全景”可同时查看整臂。路径随把手参数、候选序号和微波炉摆位重算。这是目标位姿和纯 IK 的运动学预览，未验证碰撞、连续轨迹、夹持接触或实际开门。

“微波炉摆位”区域可用 X/Y/Z 与相对名义姿态的 roll/pitch/yaw 滑块调整，也可拖拽场景中的微波炉坐标轴。“恢复冻结场景摆位”可回到起点。机身、门、旋钮及垫台一起移动，抓取与操作目标、当前纯 IK 随之重算；碗和机器人不动。这只修改当前 Viser 进程内的预览模型，不写入冻结 run，也不更新原有 cuRobo 碰撞快照，因此摆位后的绿色 IK 不代表无碰撞或可执行。

Knowledge 页面默认预览摆位为 `x=0.52 m、y=0.29 m`，只保存在 `data/task_1/knowledge/microwave_knowledge_view.json`；冻结 run 仍是 `x=0.65 m、y=0.52 m`。该摆位没有检测到初始穿透，把手纯 IK 可解。Task1 同源名义首选下，右臂开门完整 Knowledge 为 **2/5**、关门为 **4/7** 个 waypoint 纯 IK 可解；分母包含 pre-contact，不含 Task 撤离。旋钮抓取仍未求得。更近的摆位能提高纯 IK 数量，但可能让微波炉垫台穿透碗，不能把它当作任务成功。

推门区域可选右臂 `right_open` 和 `right_close`。`right_open` 从门内侧接触，Task1 逻辑按当前门角改变门宽位置、距顶部距离、standoff、局部 X/Y 倾角和后半段的门板倾角；`right_close` 从门外侧接触，默认 `85°→0°`。每个门角保留完整的确定性有序候选集合：当前默认预览摆位中，开门四组分别有 `495、495、450、405` 个候选，关门六组分别有 `297、297、363、363、297、297` 个。画面最多绘制当前门角前 50 个候选的 TCP 位置与方向：紫色为候选，红色球为所选 contact，绿色球与线段为 pre-contact 和门面朝外法向。侧栏可切换起始门角候选序号；状态记录所选候选 ID、参数和门面法向。

pre-contact 在候选选择后生成：默认沿所选接触面的朝外单位法向退 **50 mm**，可调范围 **10–120 mm**，TCP 姿态不变。默认开门路径为 `pre-contact、30°、50°、70°、85°`，关门为 `pre-contact、85°、65°、45°、25°、5°、0°`。此前的“世界 +Z 20 mm 再沿局部 +Z 30 mm”短撤离不属于用户指定的 Task1 行为，已从 Knowledge viewer 默认路径移除。Task1 推完后返回默认关节 FK 得到的 TCP，仍由 Skill/Task 编排。本页只检查纯位姿 IK，不检查候选间连续轨迹或场景碰撞。

“近似 IK 候选”区域默认关闭。显式打开后，可输入当前调参所允许的平移/姿态误差上限；严格 IK 未求得但误差在预算内的 pose 会标成“近似候选”，状态清单同时保存原误差、阈值和严格 IK 标志。这仍是 MuJoCo 纯位姿检查，不检查碰撞和连续轨迹，也不会自动开启 cuRobo Skill 的近似规划。后者必须由 Skill 阶段单独配置误差预算。

直接打开操作终点：

```bash
python -u tools/editors/view_microwave_knowledge.py --mode door_push --waypoint -1 --port 8082
python -u tools/editors/view_microwave_knowledge.py --mode knob_turn --waypoint -1 --port 8082
```

自动截图同时保存同名 JSON 清单，其中包含选中 TCP、场景版本、校准 part pose、路径 waypoint、逐点纯 IK 结果和相机：

```bash
python -u tools/viewers/capture/capture_knowledge_viser.py --tool microwave \
  --mode handle_grasp --output runs/knowledge_tuner/microwave_handle_grasp.png
python -u tools/viewers/capture/capture_knowledge_viser.py --tool microwave \
  --mode knob_grasp --output runs/knowledge_tuner/microwave_knob_grasp.png
python -u tools/viewers/capture/capture_knowledge_viser.py --tool microwave \
  --mode door_push --waypoint -1 --output runs/knowledge_tuner/microwave_door_push_end.png
python -u tools/viewers/capture/capture_knowledge_viser.py --tool microwave \
  --mode knob_turn --waypoint -1 --output runs/knowledge_tuner/microwave_knob_turn_end.png
```

Task1 候选与 pre-contact 的固定验收图可用下面的命令重拍；图片旁的同名 `.json` 是截图与状态清单：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. MUJOCO_GL=egl \
  python -u tools/viewers/capture/capture_knowledge_viser.py \
  --tool microwave --mode door_push --waypoint 0 \
  --output runs/knowledge_tuner/task1_door_candidate_precontact.png
```

调参后可连接正在运行的页面截取当前状态：

```bash
python -u tools/editors/view_microwave_knowledge.py --mode door_push --port 8082 \
  --state-out runs/knowledge_tuner/current_microwave.state.json
python -u tools/viewers/capture/capture_knowledge_viser.py --tool microwave \
  --url http://127.0.0.1:8082 --state-file runs/knowledge_tuner/current_microwave.state.json \
  --output runs/knowledge_tuner/current_microwave.png
```

截图清单的 `state` 保存微波炉 world 位姿、实际滑块数值、候选序号、waypoint、场景快照哈希和相机。之后可以复现：

```bash
python -u tools/editors/view_microwave_knowledge.py \
  --restore-state runs/knowledge_tuner/current_microwave.json --port 8083
```

如果场景快照变了，恢复会报错，避免把旧参数套到另一版资产上。

这个页面使用当前校准 URDF 的把手 visual 和旋钮元数据对齐 Concept。把手抓取沿用 `data/task_1/knowledge/grasp_profiles.json` 中从旧项目迁移的 `microwave_handle` 参数：沿把手上移 5 cm、局部 X 轴倾斜 −60°、局部 Y 轴 0°、抓取深度 13 mm，以及局部 TCP 偏移 −12 mm。截图 JSON 的 `selected_grasp_parameters` 记录实际使用参数；`data/task_1/knowledge/microwave_knowledge_view.json` 不再覆写把手姿态。

**冻结摆位下旧把手抓取的纯 IK 未求得；上述预览摆位下可解。** `door_push` 是抓把手并拉开门之后的单独推门 Knowledge，不表示始终抓住把手。机械臂上的橙色原生夹爪始终跟随实际预览关节；不可达时本体显示求解器的最近尝试姿态，红线连接实际 TCP 与目标 TCP。默认不画悬空的目标夹爪，勾选“显示目标夹爪预览”后才以半透明金色显示期望抓取姿态。参数无效时界面恢复上次有效的滑块与场景，并显示被拒绝的原因；无需重启。此前为取得绿色 IK 将把手改为 `tip_y=-45°` 的做法已撤销。**纯 IK 不检查碰撞、关节之间的连续路径、接触力或真实抓紧。** 门／旋钮的关节只做运动学 visual 更新，不步进动力学。

录制当前默认摆位的逐角推门 Knowledge 视频：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/recording/record_microwave_knowledge_video.py \
  --output runs/knowledge_tuner/microwave_door_push_preview.mp4
```

当前录制会包含 pre-contact 与 Task1 默认 20° 步长的 `30、50、70、85°` 接触点，不包含 Task 撤离，并固定同一 Viser 相机；同时保存同名 `.json` 清单与 `.frames/` 截图。清单包含每帧角度、目标 TCP、所选候选参数、纯 IK 结果、场景快照及相机。录制时会打开目标夹爪预览，以区别目标与机械臂的最近尝试姿态。也可加 `--closeup` 输出门板近景，或用 `--restore-state <此前保存的状态 JSON>` 录制当前调好的参数。视频只是离散 Knowledge 的运动学展示，不包含 MuJoCo 动力学、碰撞检查或实际成功声明。此前所有采用 10° 步长、`legacy_viser` 单 pose 或两段短撤离的视频都不能代表当前默认值。

关门视频使用 `right_close` 的基础逐角参数、门外侧接触和 `85°→0°` 起止角，不能把开门视频倒放：

```bash
conda run --no-capture-output -n curobo env PYTHONPATH=. \
  python -u tools/viewers/recording/record_microwave_knowledge_video.py \
  --push-mode right_close \
  --output runs/knowledge_tuner/microwave_door_close_legacy_preview.mp4
```

视频左上角逐帧标出 `IK PASS/FAIL`、TCP 平移和旋转误差；原始未标注的 Viser 截图仍在 `.frames/frame_*.png`。纯 IK 判定阈值是平移误差 **小于 3 mm** 且旋转误差 **小于 4°**。失败点仍显示求解器找到的最近机械臂姿态，因此单凭画面上夹爪靠近门板不能判断该点可解。当前摆位关门 7 个 waypoint 中 `85°、65°、5°、0°` 四个接触点可解，pre-contact、45° 和 25° 未求得；整段尚不可执行。

**关门与旧任务的关系。** 当前生成器对齐旧 Task1 的实际长程调用链，而非独立 Viser 的单 pose 选择。85° 的名义首选门宽比例是 `0.75`；随后保留 X/Y 倾角搜索和三个门宽位置构成的完整集合。Knowledge viewer 默认显示有序集合的第一个候选，也允许人工切换起始候选；它没有替代 Skill/Motion 执行逐角 IK、碰撞和路径筛选。资产坐标不同时不能直接比较 world TCP 绝对坐标。

真实浏览器鼠标回归：

```bash
python -u tools/diagnostics/check_microwave_viser_interaction.py
```

脚本会用 Chromium 依次点击“微波炉 X”和“沿把手上移”滑块，要求状态 JSON 与目标 TCP 同步变化，并保存摆位前、摆位后及抓法调整后三张截图。截图前会等待场景材质加载完成。

底层 Knowledge 支持旧 Push 的四种逐角模式与完整候选集合；当前 Viser 的活动臂是右臂，因此只提供 `right_open/right_close` 两种模式切换。页面不自动用 IK 选择每个门角的候选，也不做连续轨迹筛选；这些属于外部 Skill/Motion。
