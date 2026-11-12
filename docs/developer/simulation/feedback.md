# 反馈与渲染接口

受众：开发 Agent。已实现：MuJoCo 小场景反馈闭环；不是 Realman 或抓碗验收。

## 使用方式

```python
from gene_agent.feedback import snapshot, save_state, restore_state, RunRecorder
from gene_agent.visualization import render_views

recorder = RunRecorder(run_dir, config=resolved_config, metadata={"seed": 0})
recorder.event("scene.settle", "started", scene_revision=0)
state = snapshot(model, data)
save_state(model, data, run_dir / "states/settled.npz")
views = render_views(model, data, run_dir / "images", target_body="bowl")
recorder.event("scene.settle", "passed", state_id=state["state_id"], views=views)
recorder.finish("passed", sim_time=state["sim_time"])
```

异常路径应调用 `recorder.record_error(stage_id, exc)`，随后 `finish("failed", ...)`。
事件保留短错误，完整 traceback 在 `tracebacks/<stage_id>.txt` 追加，日志在 `logs.txt`。
稳定 stage_id 由业务调用方指定，例如 `scene.load`；不可用动态计数替代阶段名称。
调用方可将 task/skill/candidate、碰撞策略、源码定位等作为 event 关键字参数写入。
`fingerprint_files(paths)` 返回文件 SHA256 映射，建议放进 manifest 的 metadata；不要记录凭据。
已存在 manifest 的目录拒绝覆盖，防止不同运行证据混合。

## 状态契约

`snapshot(model,data)` 是 JSON 可序列化阅读快照：sim_time、完整 qpos/qvel/ctrl/act、
关节名与 qpos/dof 地址、body/site 的 `T_world_body`/`T_world_site`（米、旋转矩阵）、
接触 geom/body 名、距离（米）、world 接触点、接触坐标轴和接触 frame 中六维力/力矩。
接触 frame 三行分别是法向及两个切向；前三项力单位 N，后三项力矩单位 N·m。
派生状态在副本执行 mj_forward 得到，原 data 不变。

`save_state`/`restore_state` 保存 `mjSTATE_INTEGRATION`，包括 time、qpos/qvel、act、
warmstart、ctrl、外力、mocap、等式约束启用状态、userdata、插件积分状态等 MuJoCo 所定义字段。
恢复使用 mj_setState，mj_forward 刷新派生量，再写回原积分量避免 warmstart 被覆盖。
文件检查 MuJoCo 版本、维度、状态长度和编译后完整模型的 MJB SHA256。
同维度但质量、几何或其他模型参数不同会拒绝恢复；外观参数变化也会被严格拒绝。
必须加载相同模型/资产/配置，另在 manifest 保存源码/原始资产指纹以便追溯。外部 Python 控制器、随机数生成器、回调隐含状态不属于 MjData，须调用方另存。
恢复后不得同时更改控制策略再声称相同轨迹复现。JSON 快照不能代替积分状态文件。

## 相机与线程

`render_views(model,data,output_dir,target_body='bowl',width=960,height=720,cameras=None)`
返回并保存 views.json。overview/top/side/contact 四张 PNG 共享同一份冻结 data。
每张记录 state_id、仿真时间、尺寸、相机 lookat/distance/azimuth/elevation。
可用 cameras 按视角名覆盖这些参数；复现实验应将最终相机参数用于后续对比。
默认 lookat 是目标 body 原点，不代表已检测真实抓取接触点；需要时显式覆盖 contact.lookat。

复制 data 期间必须停止其他线程写入，本库不提供锁。渲染会临时扩大模型离屏 framebuffer
尺寸并在 finally 恢复，因此同一个 model 上也不应并行调用渲染或 viewer。
进程启动前设置 `MUJOCO_GL=egl`；不更改用户全局 shell 设置，不启动浏览器。
`open_viewer(model,data)` 是可选显式入口，返回 MuJoCo passive viewer，调用方关闭。

## 验证

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest tests/simulation/test_feedback.py -q
```

小场景测试完整状态恢复后 30 步 qpos 逐元素一致，并检查已施加外力被恢复；
四视图读取 PNG 校验尺寸与非空图像，确认同 state_id/time 且原 data 不变；
异常测试检查阶段事件、summary 与完整 traceback 分开。开发时实际读取 contact.png，
观察到红色球体接触灰色平面，阴影和接触近景正常；此球仅是渲染/反馈夹具。

## 碰撞代理与坐标轴叠图

`render_views(..., collision_overlay=True, site_axes=True)` 保留四张原 RGB，另生成
`overview_overlay.png`、`top_overlay.png`、`side_overlay.png`、`contact_overlay.png`。
两项可独立启用；叠图沿用原视角参数、state_id、仿真时间，并在 views.json 记录
collision_overlay、site_axes、site_names。RGB 不增加坐标轴。

碰撞叠图按接触几何所属body分类着色：机器人橙色、任务物体绿色、微波炉及其支撑蓝色、
桌面灰色；左上角图例使用同一palette。group 1/2 visual透明度降至0.18，分类只作用于
contype/conaffinity已启用的真实接触几何，未将接触禁用几何变成物理接触。
site_axes 显示所有 site 的 MuJoCo 坐标轴，X/Y/Z 为红/绿/蓝，包含左右 TCP 与旋钮 site。

RGBA 和临时 framebuffer 设置均在 finally 中还原；渲染不改变模型指纹或积分状态。
测试还将叠图调用保存的 RGB 与无叠图基线逐像素对比。
实际场景输出位于 `runs/phase1-integrated-02/collision_review/`，已读取 contact_overlay
和 overview_overlay：可见碗内部开口及阶梯碰撞壁、机器人本体碰撞盒与透明外观差异。
叠图揭示代理边界，不表示这些代理已经通过全工作空间碰撞精度验收。

## Linux viewer 安全关闭

`with open_viewer(model, data) as viewer:` 的返回对象代理原 passive Handle，支持 cam/opt/
sync/is_running/lock 等访问。退出上下文或显式 close 会先请求关闭，再等待真正渲染线程结束；
默认 shutdown_timeout=10 秒。is_running=False 仅表示停止请求，不能代替线程退出条件。

本地 MuJoCo 公共 launch_passive 隐藏 daemon 线程，直接 close 后立即进程退出曾稳定触发
exit139。适配在 viewer.py 内集中复用 `_launch_internal` 并持有 thread，用 join 等待其完成
render_loop、内部线程join和simulate.destroy；没有用固定sleep掩盖竞态。
这是明确的 Linux/当前MuJoCo私有API依赖；不承诺其他平台，升级依赖需执行以下桌面回归：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 MUJOCO_GL=glfw DISPLAY=:1 /home/ubuntu/miniconda3/envs/curobo/bin/python -m pytest tests/simulation/test_feedback.py::test_passive_viewer_closes_render_thread -q
```

离屏测试不启动窗口，该测试在无DISPLAY或非GLFW时明确skip。真实完整场景已运行30帧并
以exit0退出，复现和失败/成功证据保留在 `runs/viewer-lifecycle-review/`。
