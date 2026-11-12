# MuJoCo 视觉质量与相机契约

受众：开发 Agent。实现位于 `gene_agent/visualization/style.py`，唯一配置是
`config/visualization/studio.json`。目标是便于看清桌面、物体轮廓、夹爪和碰撞代理，
不将渲染美化解释为接触、可达性或抓取通过。

## 实际接口

```python
from gene_agent.visualization import apply_scene_style, render_views, open_viewer
apply_scene_style(xml_root)  # compose_scene完成include以后、编译XML以前
render_views(model, data, output_dir,
             view_names=('overview', 'top', 'side', 'contact', 'task', 'knob'))
with open_viewer(model, data, camera_preset='overview') as viewer:
    # 调用方负责step/sync，退出时适配会join渲染线程。
    ...
```

`apply_scene_style` 幂等，只写 visual、skybox texture 和三个灯。替换旧 scene_light，
不替换源 material/RGBA/贴图、不修改质量惯量或碰撞参数，不增加地面或任何geom。
因此没有背景geom漏入planner的问题；新增的是无物理作用的skybox。

`render_views` 默认仍返回原四视角，新增 task/knob 通过 view_names 选择；不存在knob_0_site
的模型不提供knob预设，显式请求不存在的视角会失败。overlay仍另存，不混入普通RGB。
共享 `camera_presets(model,data,target_body='bowl')` 和 `configure_camera(camera,params)`
也可由外部入口调用。桌面viewer不应再覆盖旧135度相机，否则和离屏图不一致。

## 视觉选择与取景

阅读了本地 MuJoCo 3.14.0 Renderer.update_scene、MjVisual 和
ASPIRE 的 `aspire/real/robot/models/station/station.xml`，采用其headlight/gradient skybox
配置机制，没有继承其模型、碰撞或控制配置。

- 背景使用浅蓝灰到近白梯度，避免黑背景使深色物体失去轮廓。
- 单一较弱主光产生阴影，另有无阴影补光和背光；降低明暗对比并保留接触阴影。
  MuJoCo仍是实时光栅阴影，不宣称真实面积光或离线软阴影。
- 保留源资产颜色：机器人白/灰与蓝夹爪，新微波炉米色外壳和红色旋钮/把手。
  碗和机器人本体仍呈现其原mesh细节，不能靠统一灰色材质掩盖资产缺陷。
- overview 的300度方向从工作台前侧看到双臂与任务物体；曾实际渲染0/45/180/225/270/315
  六个方位对照，排除被本体遮挡的背视方向。camera_search保留在旧场景实验目录。
- top以scene_table中心取景，覆盖整张桌面；无桌面时退回目标body。
- task以碗和微波炉中点取景，展示完整物体与右夹爪；contact对碗口/支撑近景。
- knob使用calibrated资产约定：knob_0_site的+Z朝机身外。
  相机朝向为其world法向取反，故物体yaw变化后仍从旋钮前方看；配置角度为该方向的微调。
  旧proxy的site方向不同，旧对照实验不把knob视角作为验收项。

实际相机lookat、距离（米）、azimuth/elevation（度）随每次 views.json 保存。
实验 comparison.json 另保存完整resolved_style和SHA256，不依赖以后被修改的配置文件。

## 物理不变与模型身份

光照、纹理、视觉参数本来就是完整 MJB 的一部分，所以apply_scene_style前后完整模型SHA256
通常不同。没有修改反馈模块来绕过恢复身份检查；生产restore仍要求同一个完整编译模型。

视觉对照实验先逐项断言物理字段及源RGBA/material索引一致，再显式复制完整积分状态到
仅改变视觉的模型。两者继续100步，完整mjSTATE_INTEGRATION向量最大差为0。
新小场景测试也比较200步后状态逐元素相等，并检验重复应用XML不变。
这份比较证明本次视觉改动没有改物理；不是跨不同资产模型恢复的许可。

## 对照与复现

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 MUJOCO_GL=egl /home/ubuntu/miniconda3/envs/curobo/bin/python \
  tools/diagnostics/check_visual_quality.py \
  --state runs/phase1-acceptance-nominal/state.npz \
  --views overview top side contact task
```

旧资产同状态证据：`runs/visual-quality-old-scene/`。
`before/` 是旧视觉+旧相机；`after_same_camera/` 仅改变视觉；`after/` 包含新取景。
不要把相机变化当成光照变好的证据，两类对照已分开。

新calibrated资产最终证据：`runs/visual-quality-calibrated-final/after/`。
其源scene已由compose应用新风格，主要用于最终场景验收与幂等/物理不变验证；
不可将这组before/after宣称为旧灯光和新灯光的对照。
六个最终视角均需实际读取检查；图片路径或文件存在本身不算视觉验收。

单元回归：`tests/test_visualization_style.py`，以及原反馈/overlay/viewer生命周期测试。
桌面生命周期测试需另用 DISPLAY=:1、MUJOCO_GL=glfw；EGL套件明确skip这一个窗口测试。


最终实读结果：overview、top、side、contact、task、knob六图全部打开检查。
新版rim visual修复后旋钮红色/米色环无原三角撕裂；微波炉、碗、右夹爪在task图均可辨，
top覆盖整张桌面，overview不会被机器人背部挡住。rim修复来自资产模块，不由灯光补偿。
最终comparison.json仍记录100步完整积分向量最大差0，初始状态一致。

对用户的最终入口统一为 `runs/scene-revision-final/`（避免在多次开发截图间选择）：
CLI `check-scene --collision-overlay` 已生成6张RGB与6张独立overlay，overview/task/knob及
contact_overlay已再次实际打开检查。该目录还保存 viewer-smoke.log / viewer-smoke.json：
完整最终模型恢复状态后，GLFW实际打开窗口、运行30帧、默认azimuth=300、关闭后仅MainThread，
子进程returncode=0。旧视觉灯光对照仍保留 `runs/visual-quality-old-scene/`。

最终自动回归：51 passed，1 skipped（EGL下桌面专用测试跳过）；桌面功能另由上述真实最终模型smoke验证。
