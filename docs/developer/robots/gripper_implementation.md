> 当前状态（2026-10-05）：已恢复原连杆与指端STL，移除代理滑轨及box的visual。
> 下文构建过程与复盘中的旧滑轨描述保留为历史依据，当前实现以本节和源码为准。

# 可动夹爪实现与验收

受众：后续开发 Agent。当前默认 `parallel_slide_proxy`，原始 fixed URDF 保留不改。

## 模型与接口

- 派生文件：`assets/robots/realman/urdf/generated/realman_{left,right}_parallel_slide_proxy.urdf`。
- 生成器：`gene_agent/robots/gripper.py`；参数：`config/robots/gripper_parallel_slide_proxy.json`。
- 保留机械臂、安装、TCP；移除旧固定连杆，以沿 TCP ±X 的两个 prismatic 替代。
  开口 w=q_positive+q_negative，范围0–0.08米。URDF mimic +1；MuJoCo组装显式添加 equality。
- 默认双臂模型14个arm关节+4个指关节。机械臂规划组仍各7维，不能把指位移当弧度。
- `set_gripper_width`只设置ctrl，不覆盖qpos；`measure_gripper_width`在复制状态上测量内面间距。
- 单指力上限10N；这不等于接触冲击峰值上限。质量、摩擦、惯量和真实编码映射尚未标定。

## 几何修复与边界

掌部、原连杆和指端mesh用于显示；pad、shank、carriage仅作box接触代理，正常画面隐藏。已移除代理rail。
原滑块中心TCP z=-0.047m时与原掌部collision_2/3重叠约2mm，卡住闭合。
现将滑块与rail移至z=-0.043m，连接杆相应缩短，未禁用掌部碰撞。
本模型是可运行的平行滑动近似，不是恢复了厂家真实连杆机构。

## 规划契约

`motion/gripper.py`从实际开口导出碰撞快照，对所有活动指碰撞box在旧规划link坐标中
生成保守覆盖球，保留原arm球及self-ignore。加载配置时显式传入gripper_snapshot。
完整场景碰撞与本体审计入口已接入；无快照的历史诊断必须标记未同步。
快照仅适用于该开口；开口变化后需重建。CPU覆盖测试不等于GPU轨迹或抓碗验证。
本轮未重新验收GPU规划、bowl接触抓取、真实硬件。

后续碗接触台架已完成，当前结果为未通过而非缺少实验：见
`bowl_contact_bench.md`（历史材料已本地归档）。固定腕候选在撤支撑后仍由双侧 pad 接触，
但冻结最终候选最大平移 7.073 mm、最大转角 6.109°，分别超过 6 mm和5°阈值；
未改共享夹爪参数。

## 实际证据

`tools/diagnostics/check_gripper_actuation.py`：固定右腕，0/7/41/80mm空载，20g自由方块
接触夹持2.5秒，撤台最大下降2.83mm，松爪下降77.17mm；无weld及物体pose改写。
0.8秒闭合斜坡单接触峰值2.85N，最终间距18.78mm，软接触仍有约1.22mm压入。
结果与图片在 `runs/gripper-actuation/`；完整场景零开口证据在 `runs/gripper-scene-final/`。
当前自动回归71 passed / 1 skipped；桌面反馈测试另跑7 passed（含viewer生命周期）。

## 原始资产审计（2026-10-05）

旧项目根目录为 `/home/ubuntu/Documents/yixuan/AffordToolAct`。本次搜索该项目文件和
`tamp/curobo/assets` 中的 URDF，找到以下可复用来源：

| 来源 | 内容与限制 |
| --- | --- |
| `tamp/curobo/assets/realman/urdf/realman_left.urdf`、`realman_right.urdf` | 整臂与夹爪装配；各侧 gripper_joint1/2/3/4 全部为 fixed，无可直接使用的开合自由度 |
| `tamp/curobo/assets/realman/meshes/gripper/{l,r}_inspire_jaw_*.STL` | 掌部、连杆、指端分件外观，可复用；mesh本身不定义轴、耦合或驱动 |
| `visualization/knowledge/standard_gripper.stl`、`long_gripper.stl` | 旧Knowledge可视化mesh，不是带关节的独立URDF |
| `tamp/curobo/assets/realman/meshes/gripper/long_gripper_collision.stl` | 长夹爪碰撞代理，不是当前标准夹爪的运动学模型 |
| `configs/gripper/standard_gripper.json`、`long_gripper.json` | TCP、开口等软件配置，不提供完整连杆运动学 |

因此不是“没有夹爪资产”，而是本次检查范围内没有发现独立、关节可动、可直接用于
动力学开合的夹爪URDF。未搜索整台机器或厂家其他资料，不将此结论外推。

## 生成过程：后续Codex从哪里改

1. `robots/realman.py::build_robot`读取当前机器人与标准夹爪配置，计算原有
   link_7 到 TCP 的变换，将其传给 `gripper.py::generate_gripper_urdf`。
2. 解析新项目冻结的 `assets/robots/realman/urdf/realman_{side}.urdf`，沿joint链累计
   link_7后代变换。使用源文件的实际xyz/rpy，不将近似角度强制取整。
3. 保留掌部、手腕与机械臂；移除旧link1/link2/link11/link22及其固定joint。
   新增TCP固定frame和两个平移指体。这样实现的是代理运动学，不是识别出的真实连杆。
4. 将旧link11/link22的visual复制到新指体。变换关系为：
   `T_TCP_visual = inverse(T_link7_TCP) @ T_link7_old_finger @ T_old_finger_visual`。
   再沿TCP X分别减去/加上参考内间隙69.971204mm的一半，使零位对应闭合。
   实际开口由两指各移动w/2产生；80mm上限来自旧软件配置，并非新实测。
5. 加入独立pad、shank、carriage box接触代理，以及显式质量、惯量和关节限值。
   当前生成器还加入了这些box的visual和固定rail visual，这是当前外观偏差的来源。
6. 写出派生URDF、修正相对mesh路径，保存源URDF/参数/派生文件哈希。
   生成文件会被重建覆盖，应改生成器或配置，不只手工编辑generated目录。
7. `realman.py`将URDF导入并组装为双臂MJCF，添加指执行器、内面测量site和
   equality耦合。单独URDF的mimic不能替代这里的MuJoCo驱动与耦合设置。
8. cuRobo仍使用原7轴运动学链；`motion/gripper.py`把当前实际接触代理导出到旧指link
   坐标系，增加覆盖球。改变开口后必须重新导出，不能复用旧开口快照。

生成/检查入口（项目根目录、curobo环境）：

```bash
python tools/diagnostics/check_gripper_actuation.py
python tools/diagnostics/check_gripper_actuation.py --viewer
python -m pytest -q tests/robots/gripper/test_gripper_model.py tests/robots/gripper/test_gripper_actuation.py tests/motion/test_motion_gripper.py
```

## Visual与collision的不同验收要求（用户2026-10-05明确）

- **Visual：尽量与原始夹爪一致。** 优先复用掌部、连杆、指端STL及装配关系。
  不为简化接触就把真实外观替换成显眼方块或另造一种机械结构。
- **Collision：允许明显简化与形状差异。** 可用box、凸体和球覆盖；接触pad需保留
  有效内间距和抓取面，其他区域适度保守。允许外观差异不等于允许无理由挡住开口、
  忽略障碍或改变TCP。
- visual mesh默认不参与物理碰撞；collision默认不显示在正常画面，调试叠图可单独显示。
  两者可以引用不同尺寸与文件，不强制共用一套box尺寸。
- **当前未完全满足外观要求：** 保留了掌部和指端，但删了旧连杆并加了可见滑轨/连接块。
  后续应恢复尽可能接近原件的visual；若连杆运动学不足，应明确标注视觉近似，不能凭
  静态装配坐标宣称真实机构已恢复。这是外观恢复前的历史状态，2026-10-05已按下述变更恢复原STL。
- 外观、开口运动学、物理接触、规划覆盖分别验收。改visual后对比开/半开/闭三种近景；
  改collision后重跑全开口扫查、自由物体夹持与松爪负例；保持arm/TCP回归。

## 上轮返工原因与效率复盘

没有完整逐步骤计时，以下是已观察到的问题，不是耗时比例统计。

1. 输入是固定装配模型，新增可动关节后，还要保持原TCP、指端变换和7轴planner接口。
   主要难点是这几套语义一致，不是STL文件格式本身难写。
2. 为避免画面中指端与掌部分离，又新增了可见连接杆/滑块，并同时给它们添加collision。
   这把外观修补与接触几何修改混在一起，引入了原本不必要的一轮动力学返工。
3. 已定位的卡滞是滑块与原掌部collision_2/3重叠约2mm，执行器达到限力而闭合缓慢。
   将滑块移开并缩短连接杆后，同一验收通过；没有通过全局关闭碰撞解决。
4. 新增两侧shank/carriage后，规划快照从2个pad变成6个几何体，旧测试仍假定2个，
   需要同步覆盖检查。完整场景审计还缺少“夹爪快照已同步”的必要条件，随后补齐。
5. 并行Agent中途遇到额度限制，主Agent接手整合与复测。实际完成范围还扩展到
   自由方块接触保持、松爪负例、规划覆盖和viewer，而不只是修改URDF文本。

后续顺序应为：先审计原资产并确定visual保真边界，再验可动关节/TCP；保持visual不参与
接触，单独建立最小collision并扫描开口范围；最后验证动力学与规划同步。
无需每次改外观都重建接触机构，也不应把代理外观当成用户已经认可的最终外观。

## 2026-10-05 外观恢复交付

`gripper.py`将原link1/link2的visual与对应link11/link22一起挂到同一指体，分别使用
原URDF链变换转换至TCP，再扣除参考开口的一半。两件保持原相对装配，不增加物理关节、
碰撞体或质量。删除rail visual、pad visual与shank/carriage visual；保留全部collision。
配置中移除了废弃rail字段，更新了代理假设。

这仍是视觉近似：连杆掌端随指横移，闭合时移到掌部中央，全开时移至边缘。
不是恢复厂家真实铰链转动。检查安装关系必须使用metadata中的link_from_tcp，
不要直接将SDK flange_to_tcp当作URDF link_7到TCP；当前右侧前者含Rz180，后者实际为Rz90。
独立审查曾因漏这一转换误判悬空，按实际metadata重算并检查侧视后已撤回。

验收入口：`python tools/diagnostics/check_gripper_visual.py`。结果位于
`runs/gripper-visual-comparison/`，左右列分别是原fixed装配、旧代理、新外观；原fixed装配
始终保持约70mm，不会响应行标题开口。0/40/80mm分别从105°、195°观察。
`collision_overlay_40mm.png`单独显示接触代理；`visual_sweep.gif`是直接赋值qpos的外观检查，
不能冒充动力学结果。`comparison.json`逐项验证原有碰撞几何/接触参数与刚体惯量不变。
修改前基线保存于`runs/gripper-visual-baseline/no_load_bench.xml`，该对照实验依赖此基线。

物理实验另用 `python tools/diagnostics/check_gripper_actuation.py --output-dir runs/gripper-visual-restored`。
测试新增约束：两个指体均仅使用原mesh作visual，无额外box visual，visual不参与接触。
仍未重新验证cuRobo GPU轨迹或抓碗任务，不能将外观与台架通过外推成任务完成。
