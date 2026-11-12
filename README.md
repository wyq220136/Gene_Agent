# Gene_Agent

快速导航：[仓库索引](index.md) · [用户文档](docs/user/index.md) · [开发参考](docs/developer/index.md)。

Gene_Agent 是一个面向机器人操作开发的可检查框架。它把物体几何、操作知识、Skill、运动规划和 Task 分开，使人和 Codex 可以在 MuJoCo 或已有 benchmark 的原生仿真环境中调试，并在需要时迁移到真机。

项目的两个主要目标：

- **快速部署新任务**：复用 Knowledge、Skill 和规划接口，减少为每个场景重写代码。
- **提高特定场景成功率**：允许调整物体摆位和操作参数，并在声明的位姿扰动范围内验证鲁棒性。

## 核心流程

```text
场景资产与 Concept
        ↓
Knowledge：生成参数化操作目标
        ↓
Skill：组织接近、操作、撤离和失败处理
        ↓
Motion planner：IK、路径、直线 TCP、碰撞检查
        ↓
Task：组合多个 Skill，并传递实际执行状态
        ↓
Viser / MuJoCo / 原生 benchmark / 真机反馈
```

当前主要场景是微波炉任务，包括抓碗、抓把手拉门、推门开关、旋转 `knob_0`。单 Skill、子任务和完整 Task 已有在线纯视觉规划入口；运动学规划成功不等于 MuJoCo 物理成功或真机成功。

## 选择开发入口

- **已有仿真 benchmark / 新 embodiment**：先读[原生接入与恢复流程](docs/task_development/native_benchmark.md)，按模型描述、规划 profile、原生控制分别接入。RoboTwin/ALOHA 已接通共享 Pick 与 Hit，hammer 的实际验收范围和复现命令见[操作说明](docs/user/novel_tool/hammer.md)。
- **真实场景 / 自建 MuJoCo 任务**：按[阶段流程](docs/task_development/index.md)准备资产、Concept 和 Skill，下面保留已有微波炉入口。

可用 GT 当前 pose 替代感知；严格外部题目保留原初始化、物理、预算与评分。新增能力的代码位置与限制见[接口说明](docs/developer/architecture/native_benchmark_and_embodiment.md)。

## 快速开始

以下是既有 MuJoCo 路线的 `curobo` 环境示例；原生路线按上述流程选择实际解释器：

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
```

先检查环境：

```bash
python tools/diagnostics/check_environment.py --output runs/environment-check
```

查看和调整微波炉 Knowledge：

```bash
python -u tools/editors/view_microwave_knowledge.py --mode handle_pull --port 8082
```

在线查看单个 Skill，选择后可调整场景并重新规划：

```bash
python tools/viewers/view_skill.py --skill pick_bowl
```

查看由多个 Skill 组成的子任务：

```bash
python tools/viewers/view_task.py --task microwave_object_insert
```

单 Skill 还可选择 `handle_pull`、`push_open`、`push_close`、`insert_bowl`、
`extract_bowl`、`place_bowl`、`rotate_knob`。子任务还可选择
`microwave_door_open`、`microwave_door_close`、`microwave_door_cycle`、
`microwave_object_extract`、`microwave_knob_rotate`。

查看显式 25° 拉门、90° 推门的完整任务：

```bash
python tools/viewers/view_task.py --task microwave_task1 \
  --config data/task_1/visualization/task1_visual_explicit90_verified.json
```

浏览器打开终端打印的网址。界面可调整场景并重新规划，逐段查看连续轨迹、碰撞快照、附着状态和失败信息。完整任务的这一配置使用当前校准场景的调试摆位与旋钮抓取参数；具体差异记录在配置文件中。旧 `view_skill_run.py` 用于回放已保存记录。

批量录制所有 Skill、子任务和完整任务的前上方视频，并额外录制单 Skill 的夹爪侧面视频：

```bash
python tools/viewers/recording/record_visual.py --all --fps 6
```

视频和同名规划清单保存在 `artifacts/viser_acceptance/`；它们来自在线规划的实际轨迹，不代表门或旋钮已经通过 MuJoCo 接触动力学验收。

更多入口见[常用命令](常用命令.md)。

## 目录结构

| 目录 | 内容 |
|---|---|
| `gene_agent/knowledge/` | Concept、grasp 与门/旋钮操作知识 |
| `gene_agent/skills/` | 可复用的分阶段操作逻辑 |
| `gene_agent/motion/` | 规划器抽象和 cuRobo 后端 |
| `gene_agent/applications/` | 微波炉等具体任务装配 |
| `gene_agent/simulation/` | MuJoCo 场景和执行接口 |
| `gene_agent/visualization/` | Viser、截图、轨迹与运行检查 |
| `gene_agent/feedback/` | 结构化运行记录和失败信息 |
| `data/task_1/` | 当前任务的场景与 Knowledge/Skill 参数 |
| `tools/` | 长期使用的 viewer、editor、diagnostic 和运行入口 |
| `runs/` | 每次运行生成的结果、日志、截图和视频 |

## 从哪里继续了解

- [核心说明](核心说明.md)：项目目标、方法和长期原则。
- [常用命令](常用命令.md)：最常用的运行入口。
- [在线 Skill/Task 检查](docs/user/visualization/viser_skill_task_debugging.md)：场景编辑、连续轨迹和批量录像。
- [开关门 Skill/Task 检查](docs/user/microwave/door_skill_live.md)：在线规划和连续轨迹界面。
- [微波炉 Knowledge 调参](docs/user/microwave/microwave_knowledge_viser.md)：把手、门和旋钮参数。
- [抓碗 Skill 检查](docs/user/visualization/skill_viser_inspection.md)：Viser 与 MuJoCo 的运行记录检查。
- [开发规则](AGENTS.md)：供 Codex 和开发者使用的模块边界与验收要求。
- [开发环境](docs/developer/environment/environment.md)：环境版本与重建说明。

旧项目 `/home/ubuntu/Documents/yixuan/AffordToolAct` 只作为已验证行为、参数和资产的参考。迁移功能时应沿实际运行入口核对，不能默认所有同名旧脚本都属于同一实现。
