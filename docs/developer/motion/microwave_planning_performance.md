# 微波炉完整任务规划提速（2026-10-29）

## 结果

同一台机器、`curobo` 环境、seed=123、fresh reset，完整执行开门→抓碗→放碗→关门→旋钮→开门→取碗→放回桌面→关门。墙钟包含 Python 启动、场景初始化、Knowledge 筛选、cuRobo 规划、MuJoCo 执行及原有记录写入；不包含视频导出、额外截图或本轮开发时间。

| 版本 | 完整墙钟 | Task成功 | 已完成共享阶段 |
|---|---:|---|---:|
| 原录像对应运行 | 340.73 s | 是 | 13/13 |
| 本轮现有代码复测 | 343.91 s | 是 | 13/13 |
| 精确夹爪几何更新＋solver复用 | **139.06 s** | **是** | **13/13** |

相对本轮基线减少 **204.85 s / 59.57%**，速度为 **2.47倍**。这是固定场景的一次完整对照，不能推断泛化成功率或耗时分布。前后均已启用 `cuda_graph_benchmark`，因此不能把收益归因为“首次启用 CUDA Graph”。

| 门操作规划记录 | 基线 | 优化后 |
|---|---:|---:|
| 44个运动阶段的 `plan_next` 墙钟合计 | 218.97 s | 61.08 s |
| 44个运动阶段的不同planner结构key | 44 | 3 |
| 运动阶段 acquire 命中数 | 18/44 | 36/44 |
| 推门规划轨迹最大侧向偏差 | 1.802 mm | 1.834 mm |
| 推门实际TCP最大侧向偏差 | 16.180 mm | 16.514 mm |

`plan_next` 合计包含导出碰撞、配置、获取planner、局部候选和最终规划；不包含在它之前进行的逐角Knowledge筛选。基线命中可发生于同一阶段的“Knowledge筛选→实际规划”，不代表跨阶段solver复用。不同key统计跨四个门会话，各会话结束仍会释放GPU资源，不把3个key解读为全程只创建3个实例。实际TCP偏差来自原有MuJoCo接触/控制，不等于规划直线度；没有为了提速放宽任何阈值。

## 改动及边界

- `motion/curobo/session.py`：显式开关 `reuse_gripper_geometry`。对 `parallel_slide_proxy`，把经过内容哈希及追加位置校验的动态pad球值从结构key移出；保留原arm球、球数、源版本、锁关节、机器人变换、场景拓扑和碰撞策略。
- `motion/curobo/core.py`：从本阶段精确配置加载新的 `KinematicsParams`，核对shape、自碰撞对和padding后原位copy到已存在的planner、独立IK、batch IK及碰撞复核器；清空起点检查缓存。球和reference球都更新，不舍入夹爪开口、不把旧球标成新快照。球数或结构不兼容时重建。
- 实际夹爪快照变化并复用时重置求解采样；保留CUDA Graph和tensor地址。更新/重置若异常，池丢弃整个实例，绝不使用部分更新状态。重置不保证热/冷求解逐点轨迹相同。
- `applications/microwave/door_debug.py` 读取开关；`data/task_1/simulation/door_physical_open.json` 已开启。该次对照时通用pool默认尚关闭；后续已按用户要求推广为默认开启，当前契约见 [Motion默认说明](motion_baseline.md)。本轮未接入真机、未验证Pika真机行为。
- Knowledge、Skill、Task操作几何、候选顺序、误差预算、物体摆位和物理成功判断保持原配置。运行时每阶段同步的实际夹爪几何仍可不同，实际执行终态继续传递。

尝试过“先完整规划名义目标，失败再筛局部候选”的组合方案：75.74s时在第二次开门pregrasp被MuJoCo禁止接触检查停止（arm碰门把手），**不是完整任务成功**。该组合同时改变候选策略与solver采样历史，不能单独归因某一项；已撤回名义目标优先的实现和开关，只交付保留原候选顺序的复用方案。该失败运行的CLI还有既存 `evidence_mode` 缺失引发的二次KeyError，原始失败报告已保存，不作为本轮性能方案处理。

## 验证证据

- 完整成功运行：`runs/tmp/microwave_performance/20261029/reuse_only/task/summary.json`。
- 44个门运动规划均成功、均有实际cuRobo返回、碰撞复核全部通过；18段推门保持 `linear_tcp`，没有非直线fallback。
- 最终门关闭、旋钮45°、碗在桌面、无剩余attachment。仍为**混合仿真**：有效双侧指垫接触后的辅助weld与旋钮运动学更新保持原声明，不是真机或全物理夹持成功。
- 冻结3阶段GPU检查：快照ID与结果一致，planner与wrapper球一致，graph球buffer地址不变，IK/trajopt共享参数；碰撞与直线检查通过。没有逐个独立solver对fresh expected球做额外数值验收，不将此称为那一级别的证据。
- 缓存生命周期测试先RED后GREEN；相关 `tests/motion tests/skills`：**183 passed in 2.07s**。独立只读审查无Critical/Important；Minor为上述GPU数值验证覆盖边界，已明确记录。

本轮原始配置、日志、`/usr/bin/time`、碰撞快照、失败方案patch和 `comparison.json` 集中在 `runs/tmp/microwave_performance/20261029/`，属于可清理实验产物；本文件保留结论。之前原始340.73s记录位于 `runs/video_feedback/20261029_205520/`。

## 复现及之后的经验

```bash
PYTHONUTF8=1 PYTHONPATH=. PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  /usr/bin/time -p /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/run/run_microwave_task1_physical.py --output runs/<本轮新目录>
```

需要对照时复制Task及door配置到本轮run输入目录，将Task的 `door_config_template` 指向副本；只把door中的 `reuse_gripper_geometry` 置false。保持场景、seed、模式、候选、碰撞、预算不变。不要修改历史run或将失败前缀与成功全程比较为加速结果。

先检查缓存是否真正跨阶段命中，再优化solver算法；不要仅看pool命中数，也要看实际几何身份、key及惰性solver初始化。复用动态模型时同步真实几何并清缓存，不能只删除哈希。候选策略和缓存分别试验；任何提速必须重新通过完整任务成功条件。视频导出单独计时，不为性能测试批量生成截图。
