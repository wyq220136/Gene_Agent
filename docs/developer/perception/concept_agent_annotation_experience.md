# 功能 Concept 自动标注：方案与实践记录（2026-10-17）

## 目的与现有入口

输入一张 RGB、一个已规范到 **object frame、米制**的参考 mesh，以及用户对操作功能的要求；输出可由现有 Knowledge loader 和 Viser 编辑器读取的 Concept JSON。只标注用于抓取、接触或施力的 part；每个 part 用一个 Cuboid、Cylinder 或 Ring，最多 6 个。视觉相似只是人工检查手段，Concept 不替代 visual mesh 或 MuJoCo collision。

现有命令是 `tools/run/annotate_concept.py`，实现位于 `gene_agent/perception_modules/concept_agent.py`，查看/微调入口是 `tools/editors/edit_concept.py`。具体命令和结果文件见[用户说明](../../user/knowledge/concept_agent_annotation.md)。这是**刚体的功能标注**；尚未接入 Articraft 的 articulated object 重建或其自动迭代运行器，也没有复制 Articraft 的完整实现。

## 已实现的有限轮流程

1. 校验 RGB、mesh 路径和新输出目录；读取 mesh bounds，要求有限、正尺寸且单位为米。记录输入 SHA256、脚本 SHA256、用户 prompt 和开始时间。
2. 离线渲染 mesh 的三幅正交视图。Agent 同时看 RGB 和参考视图；RGB 提供物体语义，尺寸、轴和局部坐标以 mesh 为准。首轮要求结构化 JSON；程序校验 part 数量、primitive、正尺寸、有限 pose、明显越界参数，再转成 `T_object_part`。
3. 生成 mesh/Concept 叠图；默认让 Agent 看叠图复核一次，可用 `--review-rounds 0` 跳过，最多两次复核。复核 prompt 附带每个 part 的 object-frame AABB，要求仅修正明显影响功能的错误。
4. 将每轮参数、叠图、模型 transcript 和耗时留在独立 run 目录；最终产物为 `concept.json`、`overlay_final.png` 和 `annotation_run.json`。编辑器另存 `concept.tuned.json`，保留原始自动结果。

后端目前是只读沙箱中的 Codex CLI，使用输出 schema；本地 Python 负责几何校验和落盘。调用模型前须确认这些图像和 mesh 派生视图允许交给该后端处理。`--sandbox read-only` 仅限制模型端工具写入，不代表数据不会离开本机。

## 锤子试验给出的经验

RoboTwin 锤子的米制 GT mesh 用一个圆柱锤柄和一个长方体敲击端即可描述当前操作功能。首次自动运行中，Agent 把三视图的坐标轴看错，锤头长轴偏离。明确“三图依次为 X-Y、Z-Y、X-Z，第二图横轴为 Z”，并回传 part AABB 后，从空目录重新运行，得到可用的功能近似；记录在 `runs/concept_annotation/robotwin_hammer_agent_20261017_v2/`。两次模型调用分别约 22.16 秒和 15.53 秒，完整自动标注约 37.87 秒。这个时间只代表该锤子和当时环境；不包含感知重建，也不说明批量成功率。第一次失败记录保留在 `robotwin_hammer_agent_20261017/`。

实践中最重要的是先固定 mesh frame 和单位，再对三视图的轴、part 中心及两端作交叉检查；单凭 RGB 的视觉方向容易把局部轴判断错。功能 part 只需覆盖可抓取段和实际作用面，无须补出锤子的装饰或全表面。若模型输出虽然符合 JSON schema，但接触面或抓取轴错了，应在编辑器中修正并保留自动版用于追溯。有限轮反馈提升速度，也意味着结果**没有自动收敛保证**。

## 清扫场景的实际边界

`data/realworld/brush_cube_dustpan_0/concepts/` 已有刷子、方块、簸箕的功能标注，坐标对应同场景 `assets/` 内的规范化 mesh，位姿关系为 `T_world_part = T_world_object @ T_object_part`。这组三件**不是自动 Agent 生成**：向外部 Codex CLI 上传真实 RGB 及 mesh 派生图的请求因缺少明确上传授权而被自动审批拒绝，随后使用本地查看和几何标注完成。来源、尺寸和查看命令见[清扫场景标注](../../user/scenes/sweep_scene_concepts.md)。将来若仍无该数据处理授权，继续用本地编辑器/几何方法；不得把这组标注计入自动 Agent 的质量样本。

## 用于后续 Knowledge 的约束

- Concept part 是**任务功能假设**：`grip_handle` 供抓取，`bristle_face` 供推扫，`receiving_tray` 定义接收区域；名字和语义应在任务配置中显式引用。实际可抓取性和物理接触须另验。
- 始终保存来源 RGB、mesh、规范化变换、prompt、版本/哈希及人工修订版。若 scene builder 重建 mesh 或更换 object frame，旧 Concept 不能无条件复用。
- 静态刚体先用现有有限轮标注器；对 articulated object，需要先定义 link/joint frame 与可动范围，再评估 Articraft 重建和各 link 的功能 primitive。当前运行器不具备此能力。
