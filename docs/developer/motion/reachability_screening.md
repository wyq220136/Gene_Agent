# 几何候选的单臂 IK screening

受众：开发 Agent。2026-10-04 实际 GPU 结果；不是完整任务可达性、轨迹安全或物理抓取验证。

## 输入与复现

```bash
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/check_reachability.py --output runs/ik-screening
```

分别 `build_scene` 构造 nominal 与 `seed=17, perturb=True` 场景；MuJoCo 步进3秒后冻结GT。
读取落稳碗、固定微波炉的父pose，以及右臂真实qpos；之后每个候选都使用同一状态，不修改MuJoCo。
两组各40个bowl、6个knob候选来自 `gene_agent.knowledge`，未进行预筛选或删除。
两组MuJoCo warning计数均为零；这一事实不代表碰撞穿透已经消除。

world=right SDK base。调用单个复用的 SynchronousPlanner，16种子、seed123、CUDA graph关闭；
保留原Realman自碰撞球、self_collision_ignore和限制。
碰撞场景只有URDF base [3,3,3]的0.1m诊断cube，未加入另一臂、本体、桌面、碗或微波炉。
这是明确独立的IK screening，不是从完整任务碰撞场景删除障碍获得的结果。

每个候选保存ID、参数、world pose、完整solver结果/异常、IK解、FK复核误差和失败类别。
后端通过后额外要求TCP位置误差<5mm、姿态误差<0.05rad；与当前后端默认目标容差一致。
所有拒绝结果都是 `ik_solver_rejected`，不是Python异常。
有限种子优化失败不能证明几何上绝对不可达，也不能据此归因为碰撞。

## 结果

| 场景 | Bowl IK通过 | Knob IK通过 |
|---|---:|---:|
| nominal | 35/40 | 0/6 |
| seed17扰动 | 36/40 | 0/6 |

nominal bowl被拒绝的候选：`bowl_ring_04_02`、`bowl_ring_04_04`、`bowl_ring_05_02`、
`bowl_ring_05_04`、`bowl_ring_06_04`。
seed17被拒绝：`bowl_ring_04_04`、`bowl_ring_05_02`、`bowl_ring_05_04`、`bowl_ring_06_04`。
两组knob的六个候选全部保留。没有把失败样本静默改成另一组采样。

## 固定预算摆放搜索

两组原始knob均被拒绝后，按脚本预声明预算评估12个父pose，每个六候选，完整执行72次：

| world平移增量(m) | world yaw增量0° | +30° | -30° |
|---|---:|---:|---:|
| (-0.10, 0, 0) | 6/6 | 0/6 | 6/6 |
| (-0.20, 0, 0) | 6/6 | 0/6 | 6/6 |
| (0, +0.10, 0) | 6/6 | 0/6 | 6/6 |
| (-0.10, +0.10, 0) | 6/6 | 0/6 | 6/6 |

平移在world直接相加，旋转左乘world Z yaw且保持父原点；Concept局部安装不变。
建议优先检查 `parent_0_0`：只将原微波炉world X减0.10m，位置变为
`[0.380516753814, 0.212951524518, 0.0883250779414]`，保留原姿态。
这是待验证的固定摆放建议，未修改场景配置，也没有验证该位置与桌面/机器人/其他物体是否重叠。

92次原始候选+72次搜索=164次完整记录；119次IK通过。
通过样本最大FK位置误差2.264mm、姿态误差0.001224rad。
这些数字依赖当前seed、候选顺序与优化容差，不能当作抓取误差或真实机器人精度。

结构化证据：[screening.json](../../../runs/ik-screening/screening.json)，
[运行日志](../../../runs/ik-screening/run.log)。报告包含源码hash、版本、完整解析规划配置、
场景元数据、GT pose、qpos/qvel/ctrl快照、逐候选结果和全部12组搜索pose。
下一步应先接完整碰撞快照，再验证候选接近/旋转/退出整条轨迹；不能把单目标IK成功等同旋钮任务可行。
