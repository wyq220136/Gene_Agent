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

记录 `rm_get_joint_degree`、`rm_get_gripper_state`、`rm_get_current_arm_state` 的完整原始返回。某项读取失败仍尝试其他只读项。不会设置夹爪行程、电压、使能，也不会开合或移动机械臂。检查 status=1、enable=1、error=0、actpos 在 0～1000；SDK 成功码不等于夹爪自身在线。

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
