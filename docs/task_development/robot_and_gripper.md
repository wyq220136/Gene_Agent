# 迁移机器人与夹爪

## 何时读取与输入

更换机器人、安装方向、夹爪或 TCP 时读取。输入包括 URDF / mesh、安装变换、关节限位、控制接口与实际标定。

## 原生 benchmark 中的新 embodiment

已有仿真题目使用其原生机器人模型和控制器，不要求先生成 MuJoCo 模型。机器人描述（关节组、base/TCP、夹爪）、规划 profile（cuRobo 几何与约束）和仿真适配（观测/控制/反馈）分别接入；详细接口见[设计规格](../developer/architecture/native_benchmark_and_embodiment.md)。

RoboTwin 的 `aloha-agilex` 每臂六关节，已接入原生模型/控制、cuRobo profile与共享Pick/Hit；具体FK、碰撞和Hammer任务证据见[原生流程](native_benchmark.md)。该范围不代表任意物体、机器人或真机已验收；支持不同embodiment也不允许在严格原benchmark中随意替换机器人。

按以下证据链验收新模型：

1. 冻结实际生效的模型/控制配置与资产版本，明确名称映射、单位、TCP、mimic及夹爪命令含义。
2. 多组关节配置对照原生引擎与 cuRobo FK/TCP，检查实际张合；不能靠同一转换函数自证坐标正确。
3. 检查已知碰撞/净空、非活动臂、本体、动态夹爪及 attachment 几何。
4. 执行真实短轨迹与 TCP 直线，比较目标和实际终态、控制周期及预算终止。
5. 用共享 Skill 在指定场景验收；保存成功、失败和画面，范围不能外推到完整Task或任意机器人。

下文是已存在的 Realman/Pika MuJoCo 路线及其证据要求，外部引擎按上述同等语义验证，不直接调用其 `model/data` 接口。

## 执行与完成条件


新任务不一定沿用双臂 Realman 和原夹爪。仓库已有显式的
`realman_single_pika` 配置：正装单臂、没有双臂身体底座，保留 Realman 七轴，
将 Pika 原始 URDF 的双指关节装到 `link_7`。它目前是**仿真与规划基线**，
不能仅凭能显示模型或规划短轨迹就认定 Skill、完整 Task 或真机可迁移。

换机器人/夹爪时，按同一条链逐项核对：

1. **资产与安装。** 提供机械臂和夹爪 URDF、visual/collision mesh、关节限位，
   在独立 profile 中声明 `T_world_base`、`T_link7_gripper_base`、
   `T_gripper_base_tcp`、活动臂组和关节顺序。Pika 示例的配置在
   `config/robots/realman_single_pika.json`，装配和 MuJoCo 元数据由
   `gene_agent/robots/realman_pika.py` 生成。配置中的 Pika 法兰平移与 TCP
   是暂定值；已确认的是零位装配方向，不能把视觉对齐当成实物标定。
2. **控制与状态。** 验证 MuJoCo 中各指关节的开合方向、实际指内面宽度、
   指垫接触、TCP 和关节零位；Skill 应通过机器人元数据取得当前活动臂与
   夹爪命令，而不是写死双臂或旧夹爪宽度。物体 pose、接触和执行终态
   仍须来自同一次仿真状态。
3. **cuRobo 几何与坐标。** 用同一装配 URDF、base、`link_7` 到 TCP 变换和
   关节顺序建立规划配置。区分外观 mesh、MuJoCo 接触几何与 cuRobo
   `collision_spheres`：一个 STL 文件不会自动成为规划时实际使用的碰撞体。
   Pika 当前由 `gene_agent/motion/curobo/profiles/pika.py` 读取三个原始 STL，
   为底座生成凸包保守网格球、为两指生成 AABB 网格球，再把当前 MuJoCo 指关节位置锁入规划快照；
   开口改变后要重新导出快照。检查球是否覆盖手指和掌部、是否误占抓取
   净空，以及相邻 link 的 self-collision 排除和物体 attachment。
4. **从小到大验收。** 先看零位模型、底座 XYZ 与张合；用多组关节角对比
   MuJoCo/cuRobo 的 FK 和 TCP；再测已知碰撞与已知净空、短距离连续轨迹，
   最后在指定物体位姿上跑单个 Skill、MuJoCo 接触和完整 Task。每层分别
   记录成功或失败，不能用短路径成功代替抓取成功。

本地可复现的两个起点（`curobo` conda 环境，不控制真机）：

```bash
PYTHONPATH=. MUJOCO_GL=egl conda run --no-capture-output -n curobo python tools/viewers/capture/render_realman_pika_preview.py
PYTHONPATH=. MUJOCO_GL=egl conda run --no-capture-output -n curobo python tools/diagnostics/check_realman_pika_motion.py
```

上述命令用于零位模型与 FK / 碰撞诊断；运行前核对当前 CLI 和输出目录。它们不证明新机器人的完整抓取任务或真机部署已通过。旧指南引用的 `experiments/probes/check_pika_pick_bowl.py` 已不在当前入口中，不再作为验收命令。

**旧项目参考与当前边界。** 旧项目的 `robots/realman/gripper_collision.py`、`scripts/tools/gripper/generate_long_gripper_collision.py` 与 `robots/realman/curobo_assets.py` 是旧延长夹爪的参考，不是 Pika 的尺寸或 frame 真值，不能直接复制。当前 Pika 配置已有上述规划球覆盖及局部净空检查；不要沿用旧指南“尚无审计”的笼统结论。是否还需生成可调 collision mesh，应由新夹爪的接触面、净空和误差预算决定，不能只生成 STL 就宣称规划碰撞完成。

Pika 配置仍标明法兰平移、TCP 和接触参数未完成实物标定；完整 Skill / Task 和新分布须按当前任务另行验收。旧仓库仅作参考，不直接修改。

## 交付与失败处理

交付零位与坐标轴图、多组 FK 对照、已知碰撞/净空样例、开合和操作结果，说明哪些参数仍为暂定。失败先查安装、单位、TCP 与碰撞覆盖，再查操作候选；不把单臂成功自动外推到双臂，反之亦然。
