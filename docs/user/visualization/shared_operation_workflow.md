# 共享操作流程：当前入口与验收范围

旧刷抓取的 MuJoCo、首帧预览和真机适配现在共同调用
`skills/library/pick/skill.py::run_pick_flow`。同一个循环组织打开、pregrasp、
直线接近、闭爪、夹持检查、建立持物关系、抬升和稳定检查。
后端负责设备动作、物理接触、SDK 和现场确认。

## 参数与证据

默认值及环境覆盖位于 `config/tasks/sweep/operation.json`。
场景/site 可通过 `pick_overrides` 覆盖支持字段，命令行倾角仍是显式覆盖。
各 run 保存最终值、来源和 hash。仿真默认世界 Z 抬升，现场/预览默认沿
pregrasp 方向抬升，这是保留原有行为的明确环境差异。仿真暂不支持关闭
hold 检查，不支持的覆盖报错。直立长刷保持专用几何，实际值记录在
`pick_geometry`，不能把物体相对倾角当成长刷的生效参数。

实际执行拒绝预测初态/预测检查，未知夹持不能直接抬升。预览可以使用
显式夹持假设，仍标为预测。仿真辅助 weld 继续单独记录，不代表真机夹持。

## 运行和查看

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_sweep_cube.py --randomize --seed 36 \
  --output-dir runs/shared_operation_refactor/my_test/seed36

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_skill.py --skill sweep_cube \
  --run runs/shared_operation_refactor/my_test/seed36
```

完整任务使用 `tools/viewers/view_task.py --task sweep_cube_into_dustpan --run <同一记录>`。

以下命令读取已有现场数据，只做 cuRobo 规划，无相机、SDK 或电机操作：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/shared_operation/replay_brush_pick.py \
  --run runs/realworld_sweep/execution_20261025_134304/first_frame \
  --output runs/shared_operation_refactor/my_test/capture_approach \
  --stop-after approach

PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_skill.py --skill sweep_cube \
  --run runs/shared_operation_refactor/my_test/capture_approach
```

拖动到 approach 末端查看夹爪与刷柄。点云、原始 mesh、目标 TCP 继续可选；
新记录增加参数来源和检查证据，旧记录不补造当时没有的证据。
真机执行入口仍为 `tools/run/execute_sweep_realworld.py`，默认只到 pregrasp。

## 设备与微波炉边界

型号驱动已迁至 `realworld/devices/arms/realman/` 和
`realworld/devices/grippers/{pika,realman}/`，模型/TCP 仍属 robots 与 config/robots。

微波炉现共用实际门角到推门起点、门子任务顺序、门角/新帧检查和默认 TCP
过渡构造。过期/缺失门角、MuJoCo/预测证据不能冒充现场成功。
本轮只做离线和 MuJoCo 验收，未进行微波炉真机运行。

完整 Task 已使用共享 Task1 循环；放入、取出、放置与旋钮也已接入共享 Skill 阶段执行器。
第二轮开门前的默认 TCP 过渡保留显式环境策略差异：现场启用，当前 MuJoCo 基线不启用。

## 微波炉抓碗共享化（2026-10-29）

通用阶段循环已从刷子模块移入 `skills/library/pick/skill.py`。仿真抓碗与现场
`run_observed_bowl_grasp` / `run_observed_bowl_pick` 现在使用同一循环；
抓取前缀、已确认夹持后的接续都要经过 grasp_check，不能直接跳到附着/抬升。
已有现场函数和子任务入口保持兼容；仅预检仍停在首段，不读取预测终点继续。
现场回调分别提供新帧、SDK、夹持和抬升检查；MuJoCo 提供物理接触和保持检查。
实际执行异常先标记终态未知，SDK仍提供的夹爪宽度单独保留。

本轮固定配置 fresh-reset 完整 Task1 在修改前后均完成11个记录子阶段。
抓碗候选、profile 以及 pregrasp/approach/lift 三段关节轨迹数值一致；
该记录仍为辅助抓取/取出和运动学旋钮的混合仿真，不表示真机验收。
可重跑（输出目录需不存在）：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_microwave_task1_physical.py \
  --output runs/shared_operation_refactor/my_bowl_regression
```

本轮证据集中于 `runs/shared_operation_refactor/20261029/`；实际闭爪和
抬升截图在 `evidence/`，完整任务在 `after_full_task/`。
上述记录是抓碗迁移阶段的历史验收；剩余操作与完整 Task 的迁移结果见下节。


## 剩余操作与完整 Task（2026-10-29）

放入、取出、桌面放置、旋钮已经使用共享 Skill 循环；完整 Task 由
`tasks/policies/microwave_task/policy.py` 统一决定顺序、检查和停止。独立测试入口继续调用相同函数。
仿真与现场仍分别提供接触/感知、SDK、执行及环境参数；未知证据不会自动当作成功。

本轮完整 MuJoCo 任务 13 个共享阶段通过，保留旧11个物理子记录。
最终门关闭、旋钮45°、碗回桌面、附着已释放。抓碗候选和三段关节轨迹与原基准一致。
该结果仍为辅助夹持与运动学旋钮的混合仿真，未进行微波炉真机操作。

查看本轮完整实际记录（默认 Viser，只读）：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_task.py --task microwave_task1 \
  --run runs/shared_operation_refactor/20261029/remaining/full_task
```

单独查看完整记录里的取碗 Skill（含炉内抓取与携物撤离）：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_skill.py --skill extract_bowl \
  --run runs/shared_operation_refactor/20261029/remaining/full_task
```

增加 `--play` 可在本地 MuJoCo 窗口回放。`--stage`、`--time` 选择保存的阶段和时刻。
viewer 读取实际采样，不重新规划或驱动设备。完整 Task 与单 Skill 共用同一保存模型和时间线；
松爪/落稳只有边界快照时就只呈现边界，不补造中间的物理执行。

接口和已知环境差异详见 [开发说明](../../developer/architecture/shared_operation_runtime.md)。

同样的 `--run` 也接受现场微波炉 Skill/Task 报告，进入“静态观测 + 实测端点/unknown 证据”面板：
保留各阶段 SDK 测量、原检查、目标和失败原因。保存了同源场景 XML、完整观测关节快照及来源时，
可同时查看该静态观测；只有报告时仍能打开证据面板。没有连续完整采样的现场记录不支持
`--play`，也不把离散端点插值成执行轨迹。缺场景的 `--output` 会明确报告缺项。

本轮核心回归421项通过；独立的记录/回放范围56项通过（与核心范围有重叠），
仿真基础范围68项通过、1项跳过。已实际检查本轮 `remaining/evidence/final.png` 和 `held.png`。
完整 Task 的 GPU 验收之后还修复了报告路径、异常证据及只读查看器的问题；这些修复由离线回归覆盖，
没有修改历史 run 或动作参数。固定样本成功不等于所有摆位分布都已验收。

## 蓝色旧刷抓取修正（2026-10-30）

操作 TCP（Pika 模型 X=180mm）不是夹持中心。当前旧刷抓取由实际安装
URDF 推导选定内侧夹持区，使该区域对齐 `grip_handle` Concept 中心。
该夹持区在 TCP 内为 `[-7.5, -0.0095, 6.4182] mm`；安装变换或 TCP
变化时重新推导，不能把这组数复制到其他夹爪。接触检测 site 保持原值。
显式夹持区模式不再应用旧 `tcp_above_handle_m=9mm`，以免再次变成浅抓。
旧几何探针未传夹持区时仍保留历史数值；正式旧刷调用已传入。

纯视觉闭爪现在按完整手指 mesh 的首次接触预测开口，并使用 approach
实际 FK 终态计算。当前样例预测约25.02mm，刷柄中心截面仅16.35mm；
两者不同，因为较粗柄尾先接触。局部截面不能当整指闭合宽度，更不能
直接展示零开口。预测不包含物体受力运动，**不证明已经夹紧或能稳定持物**。

查看本轮同一工作台场景的新规划（选择 close 阶段和“夹爪侧面”）：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/viewers/view_skill.py --skill pick_brush \
  --run runs/tmp/brush_grasp/20261030/pure_pick_verified
```

本轮验收：cuRobo 纯规划通过；MuJoCo 实际双侧接触通过，随后原有显式
辅助 weld 抬升通过，记录为仿真辅助夹持。关闭 weld 后，刷子仍绕夹点
转动、刷头留在桌面，稳定夹持判失败；不能称为自然夹持成功。
10N 是仿真执行器的力上限，不代表实际已施加10N。本轮未修改力控制、
接触门限、材料参数或安装/TCP，也未执行真机或重新验收完整清扫任务。
本轮记录集中在 `runs/tmp/brush_grasp/20261030/`，包括规划夹持近景
`grasp_verified_close.png`、无辅助失败及辅助夹持对照。
