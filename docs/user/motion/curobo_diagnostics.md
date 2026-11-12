# cuRobo 失败诊断

先读证据，再调参数。诊断不会改变正式 success、碰撞策略或阶段近似 IK 策略，也不执行候选关节解。新功能默认关闭；已有诊断与正式规划检查仍保留。

## 最短流程

1. 启动原生评测时，给原命令追加 `--diagnostic-mode on_failure --diagnostic-budget-s 5`。沿用原 benchmark 初始化、筛选、预算和评分。
2. 读取摘要：`python tools/diagnostics/inspect_native_run.py --run <run>`。可追加 `--episode <id> --stage approach`。默认不加载仿真或 GPU；原 JSON 输出用 `--format json`，原始规划返回用 `--details`。
3. 按摘要给出的 `diagnostics/<id>.json` 查看规划现场：`python tools/viewers/view_motion_diagnostics.py --bundle <json> --port 8081`。Viser全局/局部双图追加 `--output <本轮目录>/views`，不重跑动作。
4. 需要重新检查候选时：`python tools/diagnostics/inspect_native_run.py --run <run> --episode <id> --stage <stage> --recheck --output <新目录> --planner-python <cuRobo解释器> --planner-pythonpath <原运行记录中的依赖路径> --budget-s 5`。必须提供完整现场及未变更依赖；缺失则拒绝，不以当前场景补历史证据。

独立诊断 JSON 也可用相同 cuRobo 环境运行 `python -m gene_agent.motion.curobo.diagnostics --bundle <json> --output <新目录> --budget-s 5`。复查只做 FK/约束/球碰撞查询，不启动 benchmark、不 reset/step，不执行运动，也不重新搜索轨迹。

## 计时与独立重规划

判断原失败候选为何不合法用 `--recheck`；验证修复后的真实 IK/路径求解或比较同请求冷/热开销，用独立重规划。后者调用正式 cuRobo adapter，不执行仿真/真机。

```bash
# 旧run也能只读汇总；缺少的时间保持未知。--format json 供Agent读取。
python tools/diagnostics/inspect_native_run.py --run <run> --performance

# 原命令显式追加此选项，记录成功/失败请求和完整batch。
<native-python> tools/run/run_novel_tool.py <原参数> --save-motion-requests

# 使用原记录中的planner解释器与PYTHONPATH；目录必须不存在。
<planner-python> tools/diagnostics/replay_motion_request.py \
  --bundle <episode>/motion_requests/<id>.json --output <新目录> --warm-repeats 1
```

同进程应用在 `session.stage(..., request_output_dir=...)` 开启；隔离进程在 `ProcessPlanningSession(..., request_output_dir=...)` 开启。记录仅保存输入与结果，不额外求解。`metrics.replay_request_ref` 指向对应工件；batch共享一份完整目标集。写盘失败保留原结果及 `request_recording_error`，记录开销包含在请求wall中。

重规划也接受完整旧失败诊断；旧记录若只存batch中的一个目标，会拒绝改成单目标。资产、Python、cuRobo来源及签名严格核对，缺现场不能从当前环境补齐。项目代码变化默认拒绝；验证修复可显式加 `--allow-code-change`，输出保存变化前后代码hash，该选项不允许替换模型、mesh或依赖。新记录校验完整cuRobo包源码/配置与共享Motion代码；旧工件只有模块入口签名时，输出明确标为legacy_module_only，不能视为已验证完整依赖。`replan.json` 保存首调用cold、后续warm的原结果、耗时和异常。每次重置同seed采样，但不能保证复制原进程的全部随机/缓存历史；批次、约束和误差预算保持原值。

| 记录 | 计时语义；包含关系不能相加 |
|---|---|
| `duration_s` | 后端计时，可能含懒初始化、图捕获、检查与已有失败诊断 |
| `metrics.timing` | 新单请求细分起点检查、solver构建、含图捕获的solve、后处理；缺项不填零，批量接口未细分 |
| `planner_cache.setup_duration_s` | 获取/更新/新建pool资源；按acquisition_id只计一次 |
| `planning_timing.call_wall_s` | StageMotion一次调用wall，batch共享call_id，不重复计时 |
| `planning_calls.jsonl` | 隔离worker往返，含传输/序列化和记录；异常也追加，随log_path保存 |
| 原生planning事件的`planning_wall_s` | 已有观测和目标至规划返回，包含scene_snapshot_s；失败尝试照计，不含物理执行 |
| `candidate_timing.wall_s` | 整次筛选父级wall，包含子规划与场景准备，不能再加子项 |

原生episode结束生成 `planning_summary.json` 与 `development_summary.md`，摘要给出事实、未知和下一步命令。默认展示前三个失败，`--details`看全部。候选IK失败数量不是任务失败次数，任务成功仍读取原评分。旧记录没有call/lease身份时，旧batch按工件内元数据识别并提示限制；组件之和不是任务墙钟时间。运行时优先等待进程/工具完成通知，必要时查看状态，避免短间隔反复读文件。

## 如何判断下一步

| 证据 | 下一步 |
|---|---|
| 起点无效 | 检查实际起点、关节限位、初始穿透和刚更新的 attachment；不要归因目标不可达 |
| 目标 IK 无成功 seed | 核对原始目标、TCP、坐标/单位、工作空间、Knowledge候选；再看独立 FK 误差与约束 |
| 候选 self / scene 碰撞 | 查看同一候选的双方身份和规划碰撞几何；先分清真实碰撞、错误球覆盖、错误mesh变换或接触例外 |
| 轨迹失败 | 起点和终点的可行证据必须分别存在；查看后端实际返回的失败轨迹与无效点。未返回轨迹不表示整条路径无碰撞 |
| 误差小且约束可行 | 仅代表候选pose满足已检查约束；不证明路径可达，不自动启用近似执行 |
| 缺数据、超预算或诊断异常 | 保留原失败，标未知；显式离线复查或补采，不能猜测“无碰撞” |

“已检查中误差最小”和“已检查且约束可行中误差最小”分别报告。排名使用保存的求解器位置/转角容差归一化；容差缺失时仅用明确的展示尺度，严格收敛标未知。每项保存原目标、实际 FK TCP、q、位置/转角误差及约束检查。最多检查三个候选，不能把有限搜索称为全局最近解或不可达证明。诊断候选与原求解器所选候选可能不同，不能混合归因。

近似执行仍需显式阶段策略、误差预算、重新规划和实际后置条件检查，见项目近似 IK 规则。本工具不会放宽关键接触终态或修改 success。

## 球、场景与图片

球身份绑定有效模型hash及场景snapshot：机械臂组 / 物体 / link / 球编号。持物球同时保留原物体和源shape（历史记录缺失则未知，不猜Concept部件）。self pair使用实际cuRobo排除表与padding；环境证据记录球与mesh标识、允许接触标志。显示的碰撞位置是球中心或两球中点，不是精确mesh接触点。

Viser/离线图读取本次保存的规划碰撞mesh、动态夹爪球和持物球，显示目标及候选TCP；不能用visual或另一引擎的接触几何代替。图片标 `planned_diagnostic`，不表示机器人到过该姿态。原生现场图为 `actual`，保存阶段、state_id和相机参数；无法可靠取得实际图时明确未知，不用规划图填充。

默认软预算5秒、最多一次补充IK、三个候选、两条后端失败轨迹（每条最多检查64个采样点）、前20对碰撞。CUDA调用不能安全强制中断：单次调用可能超时，记录实际耗时并停止后续检查。预算允许时采一张已初始化原生相机的现场图；碰撞图冷启动渲染留给离线命令，避免延长评测。正式碰撞检查不受诊断预算削减。

## 代码边界与迁移

- `motion/curobo/diagnostics.py`：cuRobo证据采集及单现场复查；正式规划仍在core/adapter/session。
- `feedback/motion_diagnostics.py`、`feedback/native_run.py`：不依赖引擎的摘要、工件与只读索引。
- `visualization/motion_diagnostics.py`：保存现场的显示；不含Skill或物理执行。
- `simulation/robotwin/diagnostics.py`：原生只读现场图片；隐藏状态不进入 RuntimePort 的策略观测。

接新 benchmark 时复用 MotionRequest/Result、场景快照和摘要；在该适配器的失败分支保存工件，提供其原生图片回调即可。无需转换为MuJoCo，也无需复制诊断算法。当前自动工件/现场图接线在 Novel Tool Pick/Hit；其他入口不会因增加通用模块就自动获得相同记录覆盖。

旧run可读取原始IK误差及已有碰撞证据；没有完整现场的旧run不能直接严格复诊。显式重建旧请求的研究实验必须标记重建来源，不能覆盖旧run或冒充精确回放。

## Visual / collision 对照与Agent调视角

默认导出四张**纯场景图**：`global_visual.png`、`global_collision.png`、`local_visual.png`、`local_collision.png`。同一全局/局部对使用相同相机、世界坐标与诊断配置；文字独立保存在`README.md`和`views.json`，不覆盖场景。

- visual保留原生几何、颜色/基础色纹理，默认不透明；Viser与原渲染器的光照/PBR效果不保证逐像素相同。
- collision：碰撞A红、B蓝、机器人灰、attachment橙、其余环境统一浅绿灰。碰撞角色覆盖普通角色。
- 局部保留碰撞点周围默认0.25米的几何，按包围盒与邻域相交选择，不截断mesh。可通过`--context-radius`调节，遮挡优先换视角。
- 图片是保存的规划候选/轨迹配置，不代表实际执行。不重算碰撞，不改变规划几何或碰撞策略。巨大100m plane碰撞代理的显示过滤记录在每图warnings，visual不按尺寸删减。

显示环境安装`viser`、`yourdfpy`；自动导出需要`playwright`与Chromium（`python -m playwright install chromium --only-shell`）。已有独立Chromium可用`GENE_DIAGNOSTIC_BROWSER`指定可执行文件。显示环境与benchmark/cuRobo worker依赖分开。

```bash
python tools/viewers/view_motion_diagnostics.py --bundle <diagnostic.json> --output <新目录>/views
python tools/viewers/view_motion_diagnostics.py --bundle <diagnostic.json> --pair 0 --view local --layer both --azimuth 35 --elevation 20 --context-radius 0.16 --output <新目录>/side
```

省略`--output`启动交互查看器，可切全局/局部、visual/collision及碰撞对。`--pair`是保存列表的下标，不是球编号。鼠标可旋转、平移和缩放；Agent用`--look-at X Y Z`（world米）、`--azimuth`/`--elevation`（度）、`--distance`（米）重复截图。换角度不重跑规划/仿真。每图记录可见对象、源/图片hash与相机参数；原始碰撞双方、数值及单位保存在`views.json`。

### 新记录与旧记录

原生Robotwin场景快照增加独立`visual_snapshot`：保存原生渲染GLB（基础色与纹理）、资产hash、`T_world_body`、`T_body_visual`、状态身份、URDF/hash、planning base、关节状态和attachment关系。GLB按内容寻址，provider缓存不变几何，后续只更新pose；该缓存适用于当前资产/材质不变的episode，运行中更换资产须重建provider。采集失败记录warnings，不改变正式规划结果；当前接线不等于其他后端已接入。

离线加载按诊断选中的关节角覆盖活动臂状态，纯FK以保存的planning base计算，attachment跟随候选link；冻结的其他机器人/物体沿用保存pose。资产hash改变、关节缺失或候选状态无法绑定时，缺失部分跳过并明确标记。**旧记录没有visual就不输出visual PNG**，绝不用collision上色代替。只依据JSON中本次列出的图片查看，避免误读旧目录残留文件；建议每轮使用新目录。

新原生采集已通过SAPIEN渲染对象与受控FK测试；完整benchmark新一轮的visual采集仍需随实际评测验收。旧现场的碰撞图与受控四图显示测试不能代替该验收。

代码：`visualization/motion_diagnostic_viser.py`管显示/相机/导出，`motion_diagnostic_visual.py`管visual读取与纯FK，`simulation/robotwin/visual_snapshot.py`管原生只读保存。旧Matplotlib单PNG兼容入口显式使用`--renderer matplotlib --output <图片.png>`。
