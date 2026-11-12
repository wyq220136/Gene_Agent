# 两种 Grasp Knowledge 可视化工具

先激活环境：

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
```

**1. Knowledge 实验台**：与原 `viser_grasp_visualizer.py` 同一用途。用虚拟 Concept 单独检查 ring、cylinder axial、cylinder side、cuboidal block、trifold handle 的抓取公式。可调几何尺寸、Ring 0–360° 弧角、抓取面及参数采样方式（固定值／范围／列表）。默认从已保存的 run 读取真实夹爪 visual；可用 `--gripper-run` 指定其他已验证场景。只有预览层读取 MuJoCo，Knowledge 公式本身不依赖它。

实验台按 Concept 类型设置初值：`cylinder_axial` 用现有旋钮的半径 27.5 mm、高 10 mm；`cylinder` 用半径 30 mm、高 200 mm；`cuboidal_block` 用 30 mm 立方体；Trifold 保留现有把手尺寸。切换类型时会恢复该类型上次调过的几何和采样参数。蓝色启动基线已从两个界面移除。

```bash
python -u tools/editors/view_grasp_knowledge.py --kind ring --port 8080
```

**2. 完整场景单物体 tuner**：读取已保存的 MuJoCo GT run；显示机械臂、桌面、碗和微波炉，但只调整目标碗的 Knowledge grasp。滑块可改方位角、倾角、沿轴偏移及 TCP 后处理偏移；全部候选的位置和接近方向随参数同步更新。碗可用坐标手柄拖动，也可输入世界坐标和相对 yaw。点击“检查全部候选纯 IK”后，绿色表示求得姿态解，红色表示未求得；不检查碰撞或路径，也不修改原始 run。

默认画面显示当前候选的完整金色真实夹爪，以及一个分散位置的浅紫色真实手指。点击浅紫色手指可切换当前候选；“代表夹爪数量”默认可选 0–12 个。需要更多时，先修改“代表夹爪上限”（最多为候选总数减一），再调数量滑块。其余候选默认隐藏以保持碗沿清晰；勾选“显示全部候选位置／IK”才展开小点和接近方向，运行纯 IK 后会自动展开红绿结果。碗拖拽手柄也默认隐藏，编辑时可在“移动目标碗”中打开。当前选择、代表数量及上限、代表候选序号和图层开关都会写入截图 JSON。

```bash
python -u tools/editors/tune_scene_grasp.py \
  --run runs/pick-bowl-planning-final --target bowl --port 8081
```

界面“视角”可在 `bowl`、`gripper` 侧面近景与 `overview` 全景之间切换；启动时也可传 `--view gripper`。侧面近景按真实双指内侧 site 的开合轴定方向，并把相机放在桌面上方，便于查看两指与碗沿的位置关系。切入此视角时默认隐藏夹爪掌部，可手动重新打开。黄色候选夹爪由当前 run 的 MuJoCo visual mesh 构成，手指开口随当前候选的 closing width 变化。关闭“显示夹爪 visual mesh”可对照简化方块代理。`gripper` 视角默认隐藏实际机器人 visual，避免和黄色候选夹爪混淆。当前场景目标先只支持 Bowl，其他物体的 Concept 与 MuJoCo body 坐标映射需要逐个核对后再开放。

修改参数时近景相机保持不动，便于比较；选到较远候选时可点击“对焦当前候选”。Ring 只有一个沿物体轴线的偏移参数；旧 `depth_m` 仅在读取旧 profile 时兼容，不能和 `vertical_offset_m` 同时出现。

浏览器分别打开 `http://127.0.0.1:8080` 或 `http://127.0.0.1:8081`。**命令持续运行是在提供 Viser 服务**；关闭时按 Ctrl-C。如果不用 `conda activate`，改用 `conda run --no-capture-output -n curobo python -u ...`。普通 `conda run` 会缓存输出，导致 URL 不显示；它被 Ctrl-C 中断时打印 `CondaError: KeyboardInterrupt`，并不代表 Viser 启动失败。

场景 tuner 默认不覆盖正在使用的 profile。修改滑块后点击“保存调参结果”，结果写到不跟踪的 `archive/task_1/configs/grasp_profiles.tuned.json`；可用 `--output` 更改路径。验证后要正式采用这些参数，需要显式更新运行配置，并在任务目录说明中登记。Bowl 的 object +Y 对应 world +Z，因此 object Y 调为 `-0.015 m` 与旧 run 的世界高度校正一致。该匹配只说明两套位姿数值相等，不构成真实抓取成功证明。

Codex 或人类可以自动截图：

```bash
python tools/viewers/capture/capture_knowledge_viser.py --tool lab --kind ring \
  --output runs/knowledge_tuner/knowledge_lab.png
python tools/viewers/capture/capture_knowledge_viser.py --tool scene \
  --run runs/pick-bowl-planning-final \
  --output runs/knowledge_tuner/scene_tuner.png
python tools/viewers/capture/capture_knowledge_viser.py --tool scene --view overview \
  --output runs/knowledge_tuner/scene_overview.png
python tools/viewers/capture/capture_knowledge_viser.py --tool scene --check-ik \
  --output runs/knowledge_tuner/scene_ik.png
```

截图脚本等待 WebGL 和参数侧栏就绪，旁边保存同名 `.log` 与 `.json` 清单。已有的 `tools/editors/tune_grasp_profile.py` 是早期只看单个 Concept 的 JSON 编辑器；日常检查请使用上面的两个入口。

若要截取**正在手动调节、尚未保存**的界面，启动服务时加实时状态输出：

```bash
python -u tools/editors/tune_scene_grasp.py --port 8081 \
  --state-out runs/knowledge_tuner/live_scene.json
python tools/viewers/capture/capture_knowledge_viser.py --tool scene \
  --url http://127.0.0.1:8081/ \
  --state-file runs/knowledge_tuner/live_scene.json \
  --output runs/knowledge_tuner/current_scene.png
```

实验台把启动命令换为 `view_grasp_knowledge.py --state-out runs/knowledge_tuner/live_lab.json`，截图命令换成 `--tool lab --url http://127.0.0.1:8080/ --state-file runs/knowledge_tuner/live_lab.json`。场景截图 `.json` 内含当前参数、选中候选、浏览器实际相机（包括手动旋转后的相机）、场景 revision、快照哈希及输入是否有效；如果此时滑块输入无效，界面和清单会明确标注画面仍为上一次有效结果。截图过程若发生参数或相机修改，脚本会拒绝生成不一致的清单。

要在新进程中恢复截图时的参数、候选、物体预览位姿和视角，可以直接把截图清单传给启动入口；场景工具会自动读取清单中的原 run，并核对快照哈希：

```bash
python -u tools/editors/tune_scene_grasp.py \
  --restore-state runs/knowledge_tuner/current_scene.json --port 8081
python -u tools/editors/view_grasp_knowledge.py \
  --restore-state runs/knowledge_tuner/current_lab.json --port 8080
```

Codex 可以一步重建并截图：

```bash
python tools/viewers/capture/capture_knowledge_viser.py --tool scene \
  --restore-state runs/knowledge_tuner/current_scene.json \
  --output runs/knowledge_tuner/scene_replayed.png
```

输入无效的清单只包含上一次有效画面，恢复入口会拒绝把它当成当前参数。恢复有不同场景快照或不同候选结果的清单也会报错，不会悄悄套用旧画面。
