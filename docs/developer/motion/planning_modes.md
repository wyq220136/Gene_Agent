# 统一规划碰撞模式

2026-10-30 实现，用户目标是缩短规划等待。后端仍为 cuRobo；本轮没有接入 PyRoki，也未运行真机。

## 配置和边界

`gene_agent/motion/configuration.py` 定义 `CollisionMode`、`MotionConfig`、`motion_configuration()` 和入口装饰器 `configured_motion(config_argument)`。配置来自顶层 `motion` 段；CLI 覆盖配置。显式的 ContextVar 作用域用于现有同步 Task，退出/异常后恢复；不依靠环境变量，也不修改全局默认。顶层 `robot_only` 优先于内部旧阶段默认值。

`full` 沿用各 Skill 的世界/接触/attachment 策略；`robot_only` 不生成普通 world 几何及规划 attachment。运动请求类型、直线/旋转约束、精度预算和状态交接沿用已有实现。配置尚不自动覆盖未接入装饰器的其他独立 CLI；库调用可以显式使用作用域。

```python
from gene_agent.motion import motion_configuration

with motion_configuration({"collision_mode": "robot_only"}):
    # 导出机器人快照、创建 planner_pool、调用 Skill/Task 均放在此同步范围内。
    # 不把该作用域跨线程、子进程或异步作业的配置传递视为已实现。
    ...
```

## 准备、求解和缓存

1. `curobo/scene.py::export_collision_scene()` 在遍历场景前分流到 `robot_scene.py`。仅读取机器人/夹爪状态，保存机器人保护原语和小型 JSON；不调用完整场景 mesh 导出。
2. 非活动臂和模型元数据 `collision_protection_body_names` 声明的本体仍受保护。已接入 Realman 双臂显式声明 `robot_body`。原 box 保持几何；机器人非 box 几何转换为保守局部包围盒；这些代理在 cuRobo API 中仍属于 world，但不含桌面、任务物体。另一臂冻结，非双臂协调规划。
3. `planner_scene_options(snapshot)` 是直接构造 solver 时的统一快照适配。快速模式应传入经过标记的机器人快照；`CuroboPlannerPool` 拒绝把普通快照当成 robot_only，并在 key 中区分模式。旧阶段筛选不能移除机器人保护。夹爪显式预期开口更新仍需改变快照身份。
4. `SynchronousPlanner` 在快速模式丢弃普通 `scene_model` 与 `attachment_spheres_tool`，只接受单独传入的 `robot_protection_scene`。自碰撞保持开启，原 ignore 表和误差标准不变。最终离散姿态检查保持有效；不宣称连续碰撞证明。
5. `runtime_bridge.py` 快速模式跳过 attachment 回调、普通场景筛选/接触策略及场景几何预检查。真实执行、夹持证据、后置条件照常运行。抓持关系、显示、物理 weld 与规划 attachment 是不同事实。
6. 现场 `prepare_plan_only_context()` 继续验证实测关节、坐标、来源和另一臂冻结；快速模式忽略 attachment 球。清扫 FK 世界几何预检查显式记录 skipped。原始 MuJoCo 接触与真机运行边界检查不由该开关伪造成功。
7. 返回记录保留 `collision_mode`、`environment_collision_enabled`、`planning_attachment_enabled`、`self_collision_check`，以及原有 solver 原始返回/缓存记录；IK 无解时 robot_only 跳过重复“关闭环境再解一次”的诊断 solver。

Pika 的固定夹爪几何曾每阶段重复读取 STL、重建球两次。`profiles/pika.py` 现在以配置/STL **实际字节内容**和 URDF 路径为 key，用容量 4 的缓存保存固定几何；返回深拷贝。实际指关节/开口、规划起点和锁关节不缓存；源文件改变立即失效。这是基于本轮计时确认的额外优化。

## 验收和速度预期

产物在 `runs/tmp/robot_only/20261030/`，临时脚本同目录，允许后续删除。长期规则以本文件和回归测试为准。

RTX 4090；CUDA Graph 开启；固定 seed=123，单目标 16 IK seeds、4 trajopt seeds、1 attempt。固定输入首次调用、额外预热 2 次、测量 8 次分开记录。以下耗时含 Python wrapper/FK/结果验证，不只 CUDA kernel：

| 样本 | 热 IK 中位数 | 热 joint 轨迹中位数 | 首次 IK/轨迹 |
|---|---:|---:|---:|
| 原诊断目标 full（远处 box） | 7.67 ms | 53.19 ms | 见 full_gpu/motion_result.json |
| 相同目标 robot_only | 7.06 ms | 53.26 ms | 1.34 / 1.66 s |
| Pika 内部工作姿态 robot_only | 7.35 ms | 57.75 ms | 1.78 / 1.91 s |

Pika 同姿态沿 world Z 上移 1 cm，TCP 直线热中位数 66.56 ms，最大横向偏差 0.012 mm，终点平移误差约 0.000057 mm。它与已捕获的 joint 规划共用 planner，直线首次调用 65.81 ms。原全零伸直姿态向上再伸 1 cm 无解，诊断单独保留在 `pika_gpu/extended_arm_unreachable.json`，不计入成功样本；内部工作姿态输入来自已有首帧关节记录，已保存全部请求/结果。

固定几何缓存前，Pika 首轮上下文和配置准备合计约 0.22 s；优化后首次固定几何构建仍约 0.11 s，配置读取 4.5 ms，后续**上下文+配置**准备约 8.7–9.6 ms。不是把真实开口状态缓存掉。

相同微波炉静态场景，full 导出 168 个障碍/12 个 mesh 文件；robot_only 保留 21 个机器人保护 box/0 个 mesh 文件。4 次导出取后 3 次中位数：61.45→18.97 ms，约减少 69%。简单远处 box 的热求解速度基本一致，收益主要来自场景准备和减少场景求解负担，不能泛化为全部任务加速倍数。

GPU 证据还包括：故意包住机械臂的场景盒在快速模式不参与规划；从固定种子样本找出的已知自碰撞起点仍按 `start_collision` 拒绝；Pika attachment 为零、pool 实际命中。`mode_switch_gpu.json` 验证 full→robot_only→full 的真实 solver 切换和缓存命中，以及快速模式中机器人保护包围盒仍被拒绝。Pika 固定几何缓存前后当前夹爪快照 ID 完全一致。

完整微波炉 fresh reset、seed=123、fixed placement：full 13/13 阶段及最终条件通过，总 wall 177.16 s（混合仿真，非纯物理/真机）；robot_only 在推开门 pre-contact 的真实执行中提前碰门，13.98 s 后按原判定停止。不能拿失败前缀耗时与完整成功任务作加速比较。

工程排查线：正常可达的短程目标，热 IK 超过 0.1 s 或热轨迹持续超过 1 s 时应定位。分别查看准备/求解器冷启动、pool miss/重建、IK 候选数、重试、后检查与结果记录；困难不可达目标和失败诊断单独统计。该排查线不强制超时，也不放宽精度或碰撞来获得更短数字。

## 复现

```bash
python -m pytest -q tests/motion tests/skills tests/realworld tests/applications/microwave
python tools/diagnostics/check_motion.py --collision-mode robot_only --output runs/tmp/motion_check/fast
python tools/diagnostics/check_motion.py --collision-mode full --output runs/tmp/motion_check/full
python tools/run/run_microwave_task1_physical.py --collision-mode full --seed 123 --fixed-placement --output runs/tmp/motion_check/task
```

单次诊断不代表位置分布成功率；现场入口本轮仅完成离线接线/回归，没有授权或执行真实硬件。

同轮 CPU 回归 **574 passed**；新增模式/缓存回归在 `tests/motion/test_robot_only.py`。独立只读审查未发现 Critical/Important/Minor；未把代码审查当作硬件验证。

## 完整纯视觉对照（同日后续）

用户要求排除 MuJoCo 物理执行后重新计时。使用现有 `Task1VisualSession` 和
`data/task_1/visualization/task1_visual_explicit90_verified.json`：拉门25°、显式推到90°、放碗、旋钮、取碗、最终关门，同一初始场景/配置，两个独立进程各跑一次。场景资产与 FK 仍由 MuJoCo 查询；把 `mj_step/mj_step1/mj_step2/Renderer` 替换为报错守卫，实际未触发物理步进/渲染。未运行 GUI、录像、相机或真机。

| 项目 | full | robot_only |
|---|---:|---:|
| 整轮纯规划（含初始场景、记录） | 90.05 s | 63.88 s |
| 初始场景构建 | 2.24 s | 2.14 s |
| cuRobo wrapper 构造 | 2.77 s | 2.47 s |
| cuRobo IK/batch IK/轨迹调用合计 | 73.87 s | 53.66 s |
| 其余准备/状态转换/验证/记录 | 13.41 s | 7.75 s |
| 规划分区成功 | 7/7 | 7/7 |

本轮总耗时下降 **29.1%**；单次对照不作统计泛化。初始场景构建是整轮时间的子项，不再重复加到后面的求解器与其余时间合计中。

| 分区 | full | robot_only |
|---|---:|---:|
| 第一次开门 | 16.82 s | 8.66 s |
| 抓碗放入 | 15.15 s | 12.88 s |
| 第一次关门 | 4.88 s | 2.01 s |
| 旋钮 | 27.83 s | 23.96 s |
| 第二次开门 | 7.08 s | 2.83 s |
| 取碗放到桌面 | 10.95 s | 9.27 s |
| 最后关门 | 5.09 s | 2.12 s |

慢点已定位到现有纯视觉装配反复创建 solver：full 创建28个 wrapper、robot_only 创建26个。首次使用 solver 的调用合计58.19/46.05s，包含延迟初始化和首次图捕获，不能把这个数当成纯 CUDA 捕获时间。旋钮与物体操作仍有每阶段直接构造 planner 的路径；统一开关不会自动把这些局部实例变成共享长生命周期 pool。这是后续继续提速的明确方向，本轮未改任务候选、约束或大规模重构。

真实完整运行补出了前一轮低层测试未覆盖的三个兼容性问题，均已修复：显式快照 scope 被 constructor 加前缀导致 typed adapter 拒绝；纯视觉放碗仍读取已禁用的 attachment 球；快速快照缺少旧报告直接读取的阶段字段。新增回归验证身份不变、无球返回 None 和快照字段齐全；没有放宽 adapter 校验或伪造 attachment。

输入、逐次调用计时、所有真实规划返回及首次失败证据位于
`runs/tmp/robot_only/20261030/visual_comparison/`，汇总为 `comparison.json`。
这组纯视觉任务配置与上文物理 Task 的装配有所不同，不能用两组总时间相减推算动力学耗时。

兼容修复后 `tests/motion tests/skills` **196 passed**（2.09s），新增模式回归9项通过。
额外添加的 ObjectVisualSession UI 回归在 threading.wait 阻塞后主动中断，日志保留；
当时200项已通过，不能称整个附加UI集合通过。本轮完成的两种GPU纯规划不启动UI。

## 2026-10-30 后续保行为优化

旋钮分段已改为同一 Skill 的 pool，直线路径后检改为全采样批量 FK；本轮
完整前后对照与接口功能表见 [同步规划接口](motion_baseline.md#现行接口与保行为优化2026-10-30)。
上文 26/28 次构造是修改前的历史状态，不能继续当作当前构造次数。
