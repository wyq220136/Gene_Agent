# 给人类提供 Skill / Task 视频反馈

在项目根目录、`curobo` 环境运行。统一实现为
`gene_agent/visualization/video_export.py`，CLI：

```bash
python tools/viewers/recording/record_skill_run_video.py \
  --run <运行目录> --output <本轮输出目录>/operation.mp4
```

默认 **actual、EGL 离屏、12 fps、每视角640×480**。完整微波炉 Task1 自动选择
全景＋动作近景，横向拼成1280×480。输出 MP4 及同名 JSON，保存实际采样来源、
逐帧原时间、阶段覆盖、相机、证据等级及各部分导出耗时。

## 常用方式

- 单视角：加 `--camera task` 或 `--camera gripper_side`。
- 多视角：加 `--cameras overview detail --profile microwave_task1`。
- 单阶段：加 `--stage <保存记录中的阶段ID>`；支持的 Skill 范围加 `--skill pick|insert|extract|place`。
- 显式规划预览：加 `--mode planned`；不能称为实际执行或任务成功。
- 同采样效率/几何诊断：`--full-forward`；正常导出不需要。
- 软件渲染诊断：命令前加 `MUJOCO_GL=osmesa`，耗时不能与GPU EGL混比。

记录格式须已接入共享 ReplayTimeline。未支持的任务先接只读记录适配，不能把
轨迹/目标冒充 actual。静态真机报告没有完整采样时用现有证据面板，不伪造执行视频。

## 重新运行并自动出视频

```bash
python tools/run/run_microwave_task1_physical.py \
  --output runs/video_feedback/my_run --video
```

该开关保存实际状态后统一导出 `video/task.mp4`，默认不启用昂贵的在线逐帧PNG/GIF。
换镜头只需重新导出。用户明确要求重新运行时，仍必须从 fresh reset 开始。
视频导出失败独立保存 `video_error.txt`，不会把已经通过的任务事实改成操作失败。

## 效率与验收规则

1. 执行时保存完整实际状态；导出时复用一个 Renderer、多机位共用状态，通过FFmpeg管道编码。
   不默认生成成千上万的PNG/GIF，不用屏幕录制或为了换机位重跑cuRobo。
2. 完整实际qpos只做画面所需的位置运动学，不重新计算接触/动力学，不把渲染副本用于继续物理执行。
   缓存阶段字幕和重复采样画面；真实接触证据仍查原始运行报告。
3. 保留段间等待、最后状态、短状态事件。极短事件补帧时在JSON记录原时间与视频时间，
   不将播放长度当仿真/规划walltime，不插值虚构物体状态。
4. “成功视频”需先核对完整Task后置条件；辅助夹持、运动学更新必须保留标签。
   失败前缀可以录像，但不称为成功。
5. 默认只导出MP4和JSON，快速检查可解码、时长及阶段覆盖后即交付。
   不额外批量截图、制作拼图、重复计时或反复导出。仅首次核对新机位、发现异常或用户要求时，
   按需抽取1–3张相关帧；只做解码检查时不声称已完成画面验收。
   录像反馈不扩展为工具重构或额外评测。

旧 `record_physical_task1_video.py <run>` 仍可用，内部调用同一模块，再编码裁切出
全景、近景兼容文件。`split_physical_task1_video.py <run>` 根据视频manifest的实际
帧时间裁出七段，不再用仿真时间误切新增短事件。

本轮实测、原始视频与计时位于 `runs/video_feedback/20261029_205520/`。
计时仅代表该本机与场景，默认的12fps/640×480适合反馈，精细操作可提高分辨率和帧率。

杯、勺和普通/长刷的原生 `task_result.json`/`result.json` 现也支持统一 actual 导出，包括失败前缀；按原 XML 哈希检查，只有完整 qpos 采样可进入画面。当前原生桌面适配只支持整个已执行前缀，不支持 `--skill` 筛选。任务成败优先读取 `task_result.json`，辅助夹持说明保留在视频中。
