> 2026-10-05：模块边界继续沿用；当前缺口、自动诊断、人机交互与分阶段验收见 自动开发总计划讨论稿（历史材料已本地归档）。拟议接口尚未实现。

# 迁移设计与验收契约

受众：开发 Agent。状态：设计基线，非已实现 API。依据 2026-10-04 用户确认。

## 依赖方向

`scene + robots → knowledge → skills ← tasks`；Skill 请求 motion 规划，再通过 simulation 执行。
MuJoCo 是当前运行物理状态的权威；planner 使用带版本的只读快照。feedback 记录各层事件，
visualization 展示相同状态，不能反向成为 Task 装配入口。

初期模块为 scene、robots、knowledge、motion、skills、tasks、simulation、feedback、visualization。
按职责逐步创建，不预建插件系统。实验入口同时提供 Python 和 CLI，复用同一实现。

## 状态、计划与执行

- 机器人模型描述任意 arm/group、joint names/order、base、flange、TCP 和 gripper。
  world 不隐式等同右臂 SDK base。单臂规划时另一臂仍在场景中，不能忽略其碰撞。
- 双臂接口需要表达两个 group 的状态和同步目标；协同轨迹需额外验证两臂相互碰撞与时间关系。
  不把“两个单臂顺序运行”标为协同验证。
- 计划携带场景版本、起点、碰撞策略、候选 ID 和目标；执行前检查状态匹配，失配则重新规划。
- grasp、闭爪、attachment、抬升等阶段有显式前后条件。候选 IK 成功不等于连接轨迹成功。
- 碗的物理接触状态由 MuJoCo 决定，planner 的刚性 attachment 只用于预测；滑移后需要检测和更新。
- 旋钮本轮使用运动学模式，保持 Cylinder 轴语义和 TCP 与旋钮的几何关系。结果单独标记
  `kinematic`，不进入物理抓取成功统计。固定可行摆放先验证接近/转动/退出整条轨迹。

## 首批资产迁移检查

旧项目待核对入口：`robots/realman/curobo_assets.py`、`tamp/curobo/installation.py`、
`tamp/curobo/controller.py`、`robots/realman/realman_arm.py`、gripper 配置、URDF 和 collision YAML。
旧坐标文档只作为核对清单。用户确认当前整套 Realman 配置可运行，应先冻结副本/哈希和典型输入输出。

当前规划 URDF 的 gripper joints 为 fixed。MuJoCo 开合需要独立的驱动模型；任何近似都要标记，
并验证开口、指尖、TCP 和物理接触几何，不能声称直接导入即可物理抓取。
碗的凹腔不能被单个凸碰撞 mesh 封口。显示、Concept、规划和接触几何分别保存，统一 frame。

## 位姿扰动

保存名义 pose T、扰动所属 frame、delta 和实现版本。world 平移与局部旋转不应混用未注明的
矩阵乘法；每种采样模式写清楚公式并测试。默认桌面 XY/yaw；范围在实验配置中声明，不能暗藏在代码。
旋钮从父物体与局部 pose 组合，先搜索一个固定可行父 pose，再在其附近做声明范围的实验。

MuJoCo reset 后可进行有限 settling；记录采样前、采样后、settling 后 pose。
初态有效性检查与机器人可达性结果分开；前者可以做有上限的拒绝采样，后者不能被无声筛掉。
开发、回归、最终验收清单及采样预算固定保存。感知噪声将来作用于观测，不偷偷改变物理 GT。

## cuRobo 整理方法

先建立 API 版本清单和最小普通 pose/直线/IK 基线。原模块的策略逐项进入决策表：

| 原策略 | 初步处理 | 决策所需证据 |
|---|---|---|
| 坐标转换、关节映射、当前关节 attach | 保留语义，集中适配 | FK 和 attachment 数值/可视对齐 |
| collision cache、enabled 状态同步 | 检查版本后保留必要部分 | 添加、移动、禁用、attach/detach 回归 |
| batch IK | 待基准证明收益再启用 | 相同候选顺序的成功率、冷/热时间、显存 |
| batch 完整规划 | 后置 | 必须用真实预测起点与正确附着几何 |
| nearest IK、局部姿态 fallback | 显式策略，默认不叠加 | 是否改善困难样本、是否改变任务语义 |
| 多层 retry、CUDA graph recovery | 合并重试预算，先最小复现 | 区分环境错误、求解失败与已知版本缺陷 |
| 碰撞放宽 workaround | 不原样照搬 | 说明对象/link 和影响范围，并检查真实净空 |
| 旧异步 runner/queue | 不迁移 | 用户已明确取消 |

比较时固定场景、起点和碰撞规则；不能把不同约束的成功率当成速度优化收益。
一张表记录版本、策略、样本、结果、必要性、替代办法；尚无结果时不凭猜测删修复。

## 反馈与独立实验

每次运行包含 manifest、resolved config、summary、events、states、原始日志、图像与复现参数。
事件包含 task/skill/stage/candidate、时间、scene revision、起点、目标、碰撞策略、结果与源码定位。
terminal 摘要限制为主要结果/首个失败/证据路径；底层原始错误完整保留。
相机包含全景、顶视、侧视、接触近景与失败对象视角；原图和 overlay 分开保存。
截图必须与状态原子对应，暂停仿真或使用状态副本，避免多个视角来自不同时间。

三类独立入口：Knowledge 只生成/解释候选；Skill 在明确初态运行原子操作；Task 组合阶段。
允许开发恢复失败快照，但最终验收从 fresh reset 开始。代码更新默认新进程载入，初期不做热重载框架。
环境缺失、渲染失败、无候选、IK/轨迹失败、碰撞、跟踪误差和物理失败分开编码。

## 配置和文档

配置只保留机器人/夹爪、场景/资产、Knowledge/任务、实验四类。
同一尺寸/变换仅一个权威来源，覆盖项注明来源；所有运行保存 resolved config。
对旧配置中的经验偏移记录来源与效果，不能静默移植后当作通用常数。
每阶段实施前补充该阶段的具体文件、接口、验收实验和命令；本文件不替代所有阶段的实现细案。

## 本轮完成边界

环境与场景基础现已实现，见 phase1_plan.md 和 ../user/phase1_acceptance.md。
Realman 已导入 MuJoCo，原子 Knowledge 和同步规划基线已有独立检查；抓碗 Skill 与长程 Task 尚未实现。
本体proxy与基座球固定误报已恢复旧运行期语义并增加局部复核，详见 body_collision_audit.md。
新微波炉场景已接入，旋钮候选仍未通过完整场景检查，不能宣称可安全执行。
环境检测证据另记于 environment.md。
