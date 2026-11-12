# Pika 模型、控制点与指尖核对

## 打开已保存的预览

在项目根目录使用 `curobo` 环境：

```bash
python tools/viewers/view_pika_frames.py \
  --report runs/pika_frames/gpu_20261023/report.json
```

浏览器打开 `http://127.0.0.1:8094`。`Sample` 切换四组关节姿态和 80/40 mm
开口，分别开关坐标轴；关闭 `Show mesh` 可以检查被实体挡住的参考点。
`Focus on Pika` 切换夹爪局部与机械臂全景。只做 FK 预览，不模拟夹持，不连接设备。

点颜色：模型原点橙色、仿真 TCP 粉色、指垫中心绿色、指尖几何极值中心青色、
SDK 名义法兰紫色、参考控制点红橙色。轴颜色始终为 X 红、Y 绿、Z 蓝。

截图已保存到 [预览目录](../../../runs/pika_frames/offline_20261023/previews)。
局部截图使用半透明 mesh，以便看见内部参考点；不会改变任何碰撞或生产模型。

## 本次结果（2026-10-23）

**模型点与控制点不能混为一个 TCP。** 模型 +X 为手指延伸方向：

| 点／关系 | 数值与含义 |
|---|---|
| 当前仿真 TCP | 模型原点前方 180 mm；历史暂定操作点 |
| 当前指垫中心 | 模型原点前方 166.5 mm；由两个 inner site 中点得到，非实测接触中心 |
| 指部 mesh 最远 +X | 约 190.342 mm；两指整体包围范围的远端中心是几何辅助点，不一定在网格表面 |
| 参考代码的 150 mm | `TCP_OFFSET=[0,0,.15]` 被实际代码用于 SDK pose → reference control；不能凭此认定是模型原点或最远指尖 |
| URDF link_7 → SDK 名义法兰 | 所选 RM75 标准末端离线模型下，约局部 Z −17.2 mm；不能直接当相同 frame |

参考源是 `~/Documents/yixuan/umi_data_train/src/umi_data_train/realman/robot.py`。
读取转换实际使用 `rm_mat @ T_RM_PIKA`，下发使用其逆；过时注释不参与数值。
诊断通过 AST 读取数值字面量，不导入或运行外部控制模块。

在“SDK base 与单臂模型 base 重合、关节顺序和零位相同”的**诊断假设**下：

- SDK 名义法兰和 URDF link_7 的上述固定偏移，在另外三组姿态下最大位置残差约 **0.00143 mm**。
- 参考控制点投影到当前 Pika mesh frame 后，零位约在 +X **132.8 mm**，方向相对模型接近绕 X 180°；这表明当前两套参考点/方向定义并不相同，不能直接替换。
- 真实 GPU cuRobo 与 MuJoCo 对当前仿真 TCP 的四组 FK 最大位置差约 **0.000359 mm**，最大转角差约 **0.0000343°**。这只证明软件内部一致，没有规划或执行运动。

用户说明中的“15 cm 与 Pika 原点/指尖的关系”与代码变量的实际参考端点需要结合现场安装确认。
本次保留两种信息，不将变量名解释成已标定物理点，不把 15 cm 改成最远指尖偏移。
没有改动 `config/robots/realman_single_pika.json`、既有任务或标准夹爪默认配置。

## 重新生成核对报告

默认命令仅做模型几何核对：

```bash
python tools/diagnostics/check_pika_frames.py --output runs/pika_frames/my_audit
python tools/viewers/view_pika_frames.py --report runs/pika_frames/my_audit/report.json
```

增加厂家离线 FK 和真实 cuRobo GPU FK：

```bash
python tools/diagnostics/check_pika_frames.py \
  --output runs/pika_frames/my_gpu_audit \
  --sdk-python /home/ubuntu/miniconda3/envs/pika/bin/python \
  --sdk-model RM_MODEL_RM_75_E --curobo
```

`--sdk-model` 是显式选择的名义型号；不是读取真机型号、DH 或工具配置。厂家子进程只实例化
`Algo`，不实例化 `RoboticArm`。不提供 `--sdk-python` 或 `--curobo` 时，报告会明确记录未请求对应检查。
已有报告目录拒绝覆盖，请换一个 `--output`。报告保存参考代码和资产哈希、矩阵、关节、开口与误差。

无需浏览器的截图：

```bash
python tools/viewers/view_pika_frames.py \
  --report runs/pika_frames/my_audit/report.json --render-only
```

## 后续现场核验

下一步应核对实际机器人型号、安装件与 mesh 原点，用多组实测关节和 SDK 法兰 pose
检查名义模型是否适用，再把选定的操作点在三维图中与实际指垫位置对照。
若不一致，再测安装变换或标定操作 TCP。仅有 SDK 法兰 pose 无法反推出实际指尖位置。
现场核验通过前，报告始终标记 `accepted_for_realworld_execution=false`。

## 代码结构

- `gene_agent/robots/pika_frames.py`：坐标与几何，不持有设备或执行接口。
- `tools/diagnostics/check_pika_frames.py`：组织离线对照、厂家子进程和报告。
- `gene_agent/visualization/pika_frames.py`：共享报告的 Viser 显示与 MuJoCo 截图。
- `tools/viewers/view_pika_frames.py`：用户入口，不实现另一套机器人运动。

## 2026-10-25：修正前的遥操作与 UMI 交叉核对

用户确认机器人、Pika及安装没有变化，因此沿用成熟控制代码已有的固定变换，无需再让用户手填TCP验证布尔值。沿实际入口追踪：`pika_usage/scripts/teleop_single.sh` → `run_teleop.sh` → `teleop_backend.py::_start_controllers` → vendor 中的 `pika_realman_interpolation_controller.RealManInterface`。未采用被注释的旧旋转矩阵。

令 F 为SDK法兰、C为当前遥操作控制frame、U为UMI推理控制frame、O为本项目现有180mm操作frame。T_A_B将B坐标映射到A，米制。

```text
T_F_C = [ 0  0 -1  0    ]    T_F_U = [ 0  0  1  0    ]
        [ 0  1  0  0    ]            [ 0 -1  0  0    ]
        [ 1  0  0  0.15 ]            [ 1  0  0  0.15 ]
        [ 0  0  0  1    ]            [ 0  0  0  1    ]
```

读取 `T_base_C = T_base_F @ T_F_C`，写目标 `T_base_F = T_base_C @ inv(T_F_C)`。两个项目都有对应正/逆转换，15cm均只应用一次。某处注释写0.01m已过时，实际生效字面量是0.15m。遥操作配置没有额外覆盖此固定矩阵。

两者控制点位置相同，但U相对C的姿态为Rx(180°)：两者X都沿SDK+Z，Y/Z方向相反。不能拿UMI轴角目标直接冒充当前遥操作目标；先做frame转换。这是坐标约定区别，不要求改变硬件安装。

当前本项目O的朝向近似C，原点比C沿局部+X前移约30mm。精确 `T_C_O = inv(T_F_C) @ T_F_O` 中平移为 `[0.029999943, 0, 0.000143339] m`，朝向差约0.102°，小量来自原有URDF安装RPY使用3.14/1.57的舍入。`T_base_O = T_base_C @ T_C_O`。因此不能把15cm直接改成18cm或把控制点当作最远指尖；本次不改变抓取Knowledge、模型挂载或180mm操作点数值。

报告：[source_crosscheck_20261025/report.json](../../../runs/pika_frames/source_crosscheck_20261025/report.json)，包含两个源文件SHA256、实际常量行号、变换及4组任意法兰姿态的读写/换算验证。所有变换链一致至浮点精度，未导入外部控制模块，未访问硬件。此前RM75标准末端的17.2mm结论仅属于旧离线对照；当前192.168.1.18实机的RM75/6FB profile使用SDK法兰与link7重合的已核对关系。

## 2026-10-25：按遥操作基准完成修正（当前生效）

用户明确以当前pika_usage遥操作为准。umi_data_train/src/umi_data_train/realman/robot.py已修正RM_TO_PIKA_ROT，读/写边界使用的T_RM_PIKA现与遥操作完全一致，150mm偏移不变。上节UMI多转180°的对比仅描述修正前版本；原始审计报告不改写。

Gene_Agent更新 config/robots/pika_frames.nominal.json 与 pika_frames.rm75_6fb.json 中的T_sdk_flange_reference_control，使参考坐标也跟随正确遥操作，并记录源文件SHA256。模型挂载、SDK法兰关系和180mm操作TCP逐元素保持不变。真实执行链实际使用T_sdk_flange_operation_tcp，不是reference_control；因此这次参考坐标纠正不改变已有抓刷/清扫几何。

物体mesh、Concept、相机内外参、首帧物体pose和历史run均未改写。那些数据并非以旧UMI控制轴定义，不能因为两个控制frame曾不同就给全场景旋转180°。后续按原入口重新采集/规划会记录新profile哈希。

新报告：runs/pika_frames/aligned_teleop_20261025/report.json，保存遥操作/修正后UMI来源、配置前后哈希及数值不变量检查。UMI离线4项边界测试通过；本项目相关55项测试通过。全程没有连接或运动硬件。
