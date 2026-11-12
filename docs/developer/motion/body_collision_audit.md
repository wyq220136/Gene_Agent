# 本体碰撞迁移审计与修复

受众：开发 Agent，也供检查迁移依据的人类查阅。日期：2026-10-04。

## 结论和此前遗漏

此前“全场景下基座球与本体proxy恒相交”的数值观察正确，但没有追踪旧后端的运行期link屏蔽，
不能据此推断旧可用机器人配置失效。迁移确实漏了已有语义，已修复；没有调整TCP、安装frame、
本体位置或删碰撞资产来提高通过率。

旧项目并不是始终以完整基座球检查外部本体proxy。它默认在优化期间关闭`base_link`、`link_1`的球，
解决作为“场景障碍”加载的自身安装区域重叠；退出再恢复。本轮实现恢复这一优化机制，
同时增加更窄的最终复核：只允许两link与明确三块本体proxy的安装接触，外物和原自碰撞对仍保护。

## 实际旧调用链与文件

以下路径相对旧只读项目 `/home/ubuntu/Documents/yixuan/AffordToolAct`，行号来自本次读取：

| 文件/行 | 实际行为 |
|---|---|
| `scripts/run_microwave_task1.py:363–364` | CLI默认motion_backend=curobo、installation=mounted |
| `scripts/run_microwave_task1.py:1043,1055,1087` | create_realman_env；按配置加静态左臂；再幂等登记本体 |
| `scripts/common/realman_task.py:206–257` | 按CLI创建Realman环境；没有传入scene_collision_ignore_links覆盖 |
| `envs/realworld/realworld_env.py:232–241` | cuRobo环境无条件读取body并登记`scene.set_static_collision('robot_body', ...)` |
| `scripts/run_microwave_door_cycle.py:733–767` | 同名本体登记是幂等set，避免双份；同时构造预览 |
| `tamp/curobo/controller.py:218–228` | 创建backend时只给arm/assets/scene/execution_config，没有覆盖近基座ignore |
| `tamp/curobo/planner_backend.py:242–258` | `DEFAULT_SCENE_COLLISION_IGNORE_LINKS=('base_link','link_1')`，构造默认直接使用 |
| `tamp/curobo/planner_backend.py:3038–3076` | 上下文调用disable_link_collision，finally恢复；并非裁切body |
| `tamp/curobo/planner_backend.py:3871–3881` | 真正plan_pose调用位于上述上下文内，不是闲置帮助函数 |
| `tamp/curobo/planner_backend.py:2009–2040,2692–2720` | 失败诊断按场景link mask查场景，自碰撞要求恢复完整球后单独查 |
| `tamp/curobo/assets/realman/config/realman_right.yml` | 保存球、原self_collision_ignore；近基座运行期例外不在YAML里 |

安装链没有发现遗漏：`tamp/curobo/installation.py:66–70`计算
`T_B_C=T_B_S @ inv(T_C_S)`；`robot_body_collision.py:99,145–159`明确
`T_C_proxy=inv(T_B_C) @ Tz(-0.5m) @ T_asset_proxy`。
新模型保留同一body局部Z偏移及乘法顺序；已有独立URDF/MuJoCo/cuRobo FK和SDK记录对照。
没有发现近基座球裁切、body网格挖空或本体总开关作为这条Task1路径的默认处理。

还有两项旧任务语义必须区分：

- `configs/tasks/microwave_object_manipulation.json:609–613`的static_left_arm_collision.enabled=false，
  `scripts/run_microwave_task1.py:521–528`据此不添加；新仿真始终保留另一臂，更严格。
- `tamp/skills/rotate_knob.py:173–183,287–297`在旋钮接近/旋转使用ignore_scene_collision=True。
  backend `3150–3170`保留robot_body但可暂时忽略普通环境。`pick.py:150,199`等允许目标物体接触。
  本轮不照搬这些宽泛阶段规则到独立IK screening，因此任务候选通过率不能直接与旧任务等同。

## MuJoCo是否有必要

复现旧SDK执行或验证cuRobo FK/IK，MuJoCo不是必需依赖。
但当前项目要求通过仿真检验夹爪接触、碗是否真正抬起/保持、滑落及伺服跟踪；cuRobo只输出几何轨迹，
无法代替这些物理证据，因而保留MuJoCo有明确用途。

两套碰撞体系须分别定义：

- MuJoCo：link mesh/代理、物体接触几何、接触pair、力学求解，负责执行后的物理状态。
- cuRobo：运动臂球、本体静态proxy、快照障碍及允许接触规则，负责预测可行性。

“把当前接触几何导出给planner”不等于两套规则自动一致。固定安装区域可有模型重叠，
必须保留已知局部例外；也不能让planner为消除安装误报而忽略整个机器人与环境。
本次修复只处理规划例外，未修改MuJoCo link mesh、接触mask或真实物理接触结果。

## 最小修复及边界

1. Realman metadata声明`mounting_contact_exceptions`：`base_link/link_1`与
   `robot_body_body/head/back`，包含旧代码来源。其他机器人不默认继承。
2. 导出快照把source geom精确映射到障碍名，返回`mounting_contact_policy`，并把策略纳入snapshot hash。
3. `SynchronousPlanner(..., mounting_contact_policy=snapshot['mounting_contact_policy'])`
   在优化调用期间按旧逻辑暂时屏蔽两link，finally原样恢复全部球值。
4. 当前cuRobo公开接口没有link×obstacle过滤。成功结果必须由`MountingCollisionValidator`
   使用独立完整Kinematics复核：外物用全部球、本体只屏蔽允许的两link、原自碰撞pair与padding全部重查。
   复核失败返回`collision_postcheck_rejected`；任何复核异常不放行。
5. 所有IK成功解都复核，轨迹对全部返回采样点复核。轨迹采样复核不声称连续碰撞证明，
   当前没有扩大到连续swept验证；优化期较宽mask可能造成候选事后拒绝，不会为提高通过率重试。

文件：`gene_agent/motion/collision.py`、`curobo.py`、`scene.py`、`gene_agent/robots/realman.py`。
入口：`tools/diagnostics/check_body_collision_audit.py`。现有全场景检查入口也显式传入该policy。

## 新微波炉内部接触兼容

新场景有60对cabinet与door/knob的显式contact pair。两端都是非活动组、接触mask非零，
均已作为静态障碍导出；快照保留逐pair记录，不能全局跳过npair。
含活动臂或双零mask的pair仍明确拒绝。非零gap、负margin、活动组margin也拒绝。
外部geom的有效margin取max(geom_margin, incident_pair_margin)，本场景100µm：
box半尺寸加margin；圆柱半径/半高保守扩张；mesh凸包按最内侧面距离外扩。
这是接触激活壳的保守规划近似，不把margin称为真实实体厚度；原物理pairs完全未删除。

新旋钮候选显式使用`metadata['objects']['knob_concept']`；默认legacy仅用于冻结旧基线。

## 实际GPU证据

旧冻结场景和新场景分开运行，避免混合因果：

| 检查 | 旧冻结场景 | 新微波炉集成场景 |
|---|---:|---:|
| 原严格完整球起点与本体相交 | 8/87球 | 8/87球 |
| 局部例外后起点body/external/self代价 | 全0 | 全0 |
| Bowl候选IK+完整局部复核 | 27/40 | 30/40 |
| Knob候选IK+完整局部复核 | 0/6 | 0/6 |
| 桌内正例/远方净空负例 | 通过 | 通过 |
| link_7与body代理相撞拒绝 | 通过 | 通过 |
| base_link与普通外物相撞拒绝 | 通过 | 通过 |

冻结旧场景修复前同样完整障碍曾bowl0/40；本轮原始严格起点碰撞依然保存，未删body或改frame。
非基座保护负例body代价0.04822，近基座外物保护负例external代价0.07085，均被拒绝。
原始本体碰撞贡献：body与base_link球0–6及link_1球13；back与base_link球1。

新场景额外执行完整MotionPlanner小幅运动：从settled GT起点，首关节目标+0.04rad对应TCP，
单次规划成功，返回整条轨迹通过局部安装策略之外的外物与自碰撞复核。
这不是抓碗/旋钮任务完成，也不是MuJoCo物理执行成功。

- [旧冻结审计](../../../runs/body-collision-audit/audit.json)
- [新版集成审计](../../../runs/body-collision-audit-current/audit.json)
- [新版运行日志](../../../runs/body-collision-audit-current/run.log)

CPU回归27项通过，新增优化mask异常恢复、body/external精确分组、内部pair/margin转换检查。
GPU命令：

```bash
PYTHONUTF8=1 PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/check_body_collision_audit.py --current-scene --output runs/body-collision-audit-current
```

所有失败候选完整保留。knob的solver拒绝不能直接认定为绝对不可达；本轮未关闭微波炉碰撞来求通过。

## 汇总条件修正

交叉审核发现原`audit_pass`只检查安装例外及碰撞探针，遗漏完整规划状态。
现由纯函数`evaluate_audit`统一计算，额外要求small_motion_smoke.success、
collision_verification.valid，以及sample_count>1全部成立。
新增5项汇总测试通过，覆盖正常、规划失败、碰撞复核失败、仅单点与缺少规划证据。

本次只对已保存的新版GPU结果重算汇总，没有重跑候选或GPU规划：
实际轨迹success=True、复核valid=True、sample_count=21，修正后audit_pass仍为True。
原始报告保存在`audit-before-summary-fix.json`；新版报告的summary_revalidation记录原文件SHA256
和汇总脚本SHA256，原GPU运行时code_sha256保持不变，明确区分求解证据与后续汇总修正。
旧冻结基线报告保留其当时有限验收范围，不能按缺少motion字段的旧报告宣称新版整体验收。
