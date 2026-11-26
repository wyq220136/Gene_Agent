# 微波炉 cuRobo 提速

真机微波炉入口可在不改任务阶段、目标候选、碰撞策略或执行检查的前提下减少规划器重建、场景世界写入及静态几何计算。原有命令默认对兼容的 Realman 右臂、parallel-slide 夹爪、full 碰撞配置启用 `single_reuse`；其他配置沿用 baseline。可用 `--curobo-optimization baseline` 建立对照，`--no-freeze-import-gc` 关闭独立命令的启动 GC 优化。优化仅覆盖同步执行路径。

`single_reuse` 要求 CUDA Graph、单环境、full 碰撞、cuboid/mesh 世界及受验证的机器人模型；不兼容输入明确报错，不把几何或约束静默降级。固定缓冲区容量不足、mesh 文件身份不确定或更新失败时也拒绝沿用旧 solver。独立规划句柄由任务 scope 管理，完整清理 GPU 资源。每轮运行写入 `planning_optimization.json` 与 `planning_bootstrap.json`。

结果保持真实目标数和顺序，IK 分桶只在 GPU 图中用重复的最后一个目标填补输入，返回时严格裁掉填充项；原候选筛选和碰撞/约束后检不变。baseline 保留原精确批次大小。缓存只用于静态 YAML、完整输入匹配的纯几何门候选、编译 MuJoCo shape 以及同一加载内容的 mesh 位姿更新；不缓存场景 pose、可达结果或操作状态。网格源文件和适配器在失败请求回放时仍记录实际身份。

可核验的范围：现有 `tests/motion/test_realworld_scene_collision_policy.py` 正反例验证忽略场景碰撞的近基座 link 之外的场景碰撞及所有自碰撞仍会检查；复用模式附加碰撞球、mesh 与 cuboid pose 更新、25项批 IK 及 solver buffer 地址在 CUDA 上对照 baseline。运行：

```bash
CUDA_VISIBLE_DEVICES=6 python -m pytest -q tests/motion/test_realworld_scene_collision_policy.py
```

这是离线规划与碰撞回归，不是完整任务成功率、吞吐基准或真机验收。固定批次填补可能改变 cuRobo 求解随机采样，实际几何与成功返回仍应通过上述冻结输入验证。操作流程见[真机 Task1 操作指南](../../user/microwave/realworld_task1_operation_guide.md)，本加速方案不更改指南或真机动作流程。

## 2026-11-25 回归记录

在 `gene_agent` 环境中设 `PYTHONUTF8=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=6`，全部 `tests/motion`，现场 CLI、单次感知 plan-only、规划资源模式、轨迹时序、纯几何门候选缓存及 Python 3.9 兼容测试：383 项通过。包括真实 CUDA 规划、碰撞正反例、持物/夹爪切换和 25/119 两档批 IK；另补诊断配置接线的定向回归通过。

更大范围的真机、Knowledge 和微波炉应用回归仍有 20 项既有失败，已通过只读加载改动前 HEAD 的对应 Python 模块、使用原资产和原测试逐项重现同样失败。这些涉及感知坐标保护、历史候选摘要和仿真测试替身，与此次加速分开处理，没有修改它们的逻辑或断言。未连接机器人，未测完整现场任务耗时；不引用另一仓库的仿真加速比例作为本次真机收益。
