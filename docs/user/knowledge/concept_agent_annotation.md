# 功能 Concept 自动标注

输入一张 RGB、一个已经在 **object frame 且单位为米**的参考 mesh，以及一句操作要求。Agent 最多标注 6 个功能 part，每个只用一个 Cuboid、Cylinder 或 Ring。默认调用 Codex CLI 两次：首次给出参数，再看正交叠图复核一次；不运行数值拟合，也不添加装饰细节。

## 锤子命令

在项目根目录、`curobo` 环境运行。Codex CLI 须已登录；此命令不需要另外启动 SAM3、SAM3D 或 Viser 服务。

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/run/annotate_concept.py \
  --rgb runs/robotwin_foundation_pose/clean_50_seed0/seed_000/initial/rgb.png \
  --mesh runs/robotwin_foundation_pose/clean_50_seed0/meshes/hammer_scaled.obj \
  --prompt '锤子只标注用于抓取的锤柄和用于敲击的锤头；各用一个基础几何体，忽略羊角和装饰。敲击面在锤头远离羊角的一端。' \
  --output-dir runs/concept_annotation/my_hammer_run
```

`my_hammer_run` 是**新运行的示例目录**，上面的标注命令成功结束后才会创建。`--output-dir` 必须是新目录，避免覆盖已有标注。需要更快时加 `--review-rounds 0`，只调用一次模型；允许 0–2 次复核，默认 1 次。`--model` 可覆盖本机 Codex CLI 默认模型。

运行目录会保存 `concept.json`、`overlay_final.png`、每轮的 Agent 参数/日志/叠图、输入哈希和分轮耗时。失败时保留已产生的材料，并在 `annotation_run.json` 写明错误。`concept.json` 的每个 `pose_matrix` 是米制 `T_object_part`，可被现有 Knowledge loader 和参数编辑器读取。

直接查看本项目**已经生成**的锤子结果，并继续手动调整：

```bash
PYTHONPATH=. /home/ubuntu/miniconda3/envs/curobo/bin/python tools/editors/edit_concept.py \
  --mesh runs/robotwin_foundation_pose/clean_50_seed0/meshes/hammer_scaled.obj \
  --concept runs/concept_annotation/robotwin_hammer_agent_20261017_v2/concept.json \
  --output runs/concept_annotation/robotwin_hammer_agent_20261017_v2/concept.tuned.json
```

如果你先运行了上面的自动标注命令，再查看自己的新结果，把这条命令中两处 `robotwin_hammer_agent_20261017_v2` 都换成 `my_hammer_run`。

## 本次真实验收

`runs/concept_annotation/robotwin_hammer_agent_20261017_v2/` 是从空目录开始的自动运行。结果为一个锤柄 Cylinder、一个敲击端 Cuboid；已人工打开三视角叠图并确认功能区域大致对齐。两轮耗时分别为 **22.16 秒**、**15.53 秒**，总耗时 **37.87 秒**。第一次试验中 Agent 误读了锤头坐标轴，记录保留在 `robotwin_hammer_agent_20261017/`；随后加入轴说明和 part AABB 的反馈，再从空目录运行得到上述结果。

实现边界、坐标系检查与失败经验见[开发记录](../../developer/perception/concept_agent_annotation_experience.md)。

这属于视觉近似标注，不提供部件真值分割或毫米级拟合保证。RGB 只帮助辨认功能；所有三维参数以 mesh 的 object frame 为准。当前命令用于静态物体，尚未把 articulated object 自动路由到 Articraft 重建。发现偏差时可直接在 Viser 编辑器中保存修订版。
