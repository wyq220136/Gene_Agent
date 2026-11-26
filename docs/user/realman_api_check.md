# RealMan API 基础自检

脚本：`tools/diagnostics/check_realman_api.py`。独立于 Task、感知和 cuRobo，只依赖 Python 标准库及现场已安装的 `Robotic_Arm` SDK。`--help` 不导入 SDK。

## 运行位置

代码已放在 58.220.16.98 的 `/data1/yuquan/Gene_agent`。实际运行必须在能访问机械臂控制器的现场主机上。现场若使用日志中的路径，先把脚本同步到该主机的 `/home/ubuntu/yuquanwang/Gene_Agent/tools/diagnostics/`，激活原来运行真机任务的 Python 环境。

```bash
cd /home/ubuntu/yuquanwang/Gene_Agent
python tools/diagnostics/check_realman_api.py --help
```

以下 `ROBOT_IP` 必须替换成对应侧机械臂控制器 IP，不能填开发服务器的公网 IP。每次只连接一侧；两侧分别执行。避免其他程序同时控制机器人。

## 1. 默认只读（先执行）

```bash
python tools/diagnostics/check_realman_api.py --ip ROBOT_IP --group left
python tools/diagnostics/check_realman_api.py --ip ROBOT_IP --group right
```

记录 `rm_get_joint_degree`、`rm_get_gripper_state` 的完整原始返回。某项读取失败仍尝试其他只读项。不会设置夹爪行程、电压、使能，也不会开合或移动机械臂。检查 status=1、enable_state=1（兼容旧字段 enable）、error=0、actpos 原始反馈在 -10～1500；SDK 成功码不等于夹爪自身在线。

## 2. 单关节微动并返回

现场确认机械臂静止、工具未持物、运动空间净空，操作员可立即急停。程序直接调用 API，不做场景碰撞规划；小角度不代表任意初态都安全。

```bash
python tools/diagnostics/check_realman_api.py --ip ROBOT_IP --group right --action joint --execute
```

默认第 7 轴 +0.5 度、1% 速度，其他关节保持实测起点；到位后回到实测起点。可用 `--joint 7 --delta-deg -0.5` 改方向，允许幅度 0.2～1 度。监测全部七轴、连续三次到位才返程，超时或异常尝试停止和清空轨迹，不盲目返程。通信中断时软件停止可能无法送达，需现场急停。

## 3. 夹爪微动并返回

确保夹爪空载、手指及物体远离夹持区域。

```bash
python tools/diagnostics/check_realman_api.py --ip ROBOT_IP --group right --action gripper --gripper-step -5 --execute
```

此例从当前行程略微闭合 5/1000，然后返回。正数张开、负数闭合，允许 1～10/1000。默认 +5；如果当前为 1000，应选 -5，如果为 0，应选 +5；越界会拒绝，不会截断。该值是 SDK 归一化行程，不代表已标定的毫米值。夹爪非空闲、离线、未使能、报错或位置异常时拒绝命令；执行失败不自动反向。本工具未实现夹爪专用急停，需使用现场急停。

## 输出与本次故障

终端打印 API 原始返回；JSON 默认写到仓库 `runs/tmp/realman_api/<时间>/<侧别>-<唯一编号>.json`。可用 `--output /指定路径/report.json` 指定文件。退出码 0 表示所选自检通过，1 表示失败。报告中 calls 保留请求参数、原始返回和异常，checks 给出解释。请保留两侧报告以进一步定位。

上传日志 `task1_plan_20261126_151014` 定位到 `docs/01-task-skills.svg` 的 Stage 2 load_bowl → pick_bowl，初始场景观察阶段（尚未进入 Pick 的运动阶段）。`scene.py` 同时读取两侧夹爪；`RealmanGripperAdapter.read_width_m` 对 actpos 的有限性/0～1000 范围校验失败。原日志未保存左右侧及原始状态，因此不能确定离线、未使能、硬件错误或异常数值的来源，也不能归因为 cuRobo 或视觉估计失败。

本次新增诊断入口，未放宽运行时位置校验，未改动原有任务代码和未提交修改。尚未在真机执行，不声称原故障已修复。下一步根据只读报告修复实际设备配置或 SDK 数据契约，随后再验收真机运行。

API 语义参考：https://develop.realman-robotics.com/robot/apipython/struct/gripperState/ 和 https://develop.realman-robotics.com/robot/apipython/classes/movePlan/


## 当前读取方式与现场报告

- 夹爪开口：rm_get_gripper_state 的 actpos，原始反馈允许-10～1500，位置指令仍严格为0～1000；几何宽度 = max(0, actpos) / 1000 × max_width_m。enable_state为Python SDK使能字段，兼容旧enable字段。
- 机械臂关节：rm_get_joint_degree；原任务的夹爪空间位姿由 rm_algo_forward_kinematics 加已标定 T_flange_tcp 变换计算。
- 自检已移除 rm_get_current_arm_state，不再将该综合接口的-3作为自检失败条件。原任务夹爪读取本来就不依赖它。
- 位置、抓取、释放命令仅返回整数状态码，到位事件也不包含actpos/实测TCP。因此不能从命令返回值获得动作后位置，仍保留必要的实测终态采样。已有执行层返回 measured_end_width_m，后续阶段可复用经确认的终态；该字段来源于实测，不能用命令目标伪造。

现场 right-1795683836435607586.json 的actpos=1003直接来自rm_get_gripper_state。当前范围允许1003反馈；动作终点检查也区分反馈上限与指令上限，目标到位误差检查保持不变。

```bash
python tools/diagnostics/check_realman_api.py --ip 169.254.128.19 --group right --samples 5
```

多轮读取不会发运动；原始值在calls，各轮检查在samples。请同步新版程序后运行。


当前反馈容许范围为用户指定的软件策略，不是新的硬件标定。-10～0反馈不能作为负几何开口，映射为0；原始actpos及修正标记保存在适配器last_reading，动作终点报告readings.sdk_reading也保留它们。正反馈不截断，1003按8cm配置换算为0.08024m，1500为0.12m。自检calls始终保存原始返回。

夹爪往返自检若起点反馈在0～1000之外，会拒绝运动以免返程发出非法目标。普通任务仍可下发合法目标；不扩大SDK位置、力、速度或行程配置范围。不使用rm_get_current_arm_state。


## 抓碗执行前先将右夹爪设为800

入口 tools/run/run_microwave_task1_realworld.py 在 --execute 且 --subtask pick_bowl 或 load_bowl 时自动启用。沿用原执行命令，无需新增参数。完成必要的本地参数校验后，先连接右臂，直接下发800（8cm配置对应0.064m）；发令前不读取夹爪开口、不检测场景。随后读取反馈，连续3次在800±10内且波动不超过5才继续连接左臂、构建观察器和后续流程。SDK失败或3秒内未确认稳定到位则停止，不继续抓取。

记录位于本轮run/right_gripper_preparation.json。只规划/预检不发令；其他子任务及all不自动启用此动作，避免改变已持物状态。该初始化是直接夹爪控制，不经过场景碰撞规划；执行前需保证夹爪为空载且附近无手指或障碍物。本次代码仅离线验证，未远程触发真机。
