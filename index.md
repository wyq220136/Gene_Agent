# 仓库索引

先读 README，再按工作目的选择入口；不需要逐层读取索引。

| 入口 | 用途 |
|---|---|
| [README.md](README.md) | 安装环境、核心流程和常用运行入口 |
| [AGENTS.md](AGENTS.md) | 开发规则与验收边界 |
| [核心说明.md](核心说明.md) | 项目目标与工作方法 |
| [new_task.md](new_task.md) | 新任务、机器人与夹爪的扩展流程 |
| [常用命令.md](常用命令.md) | 当前运行命令速查 |
| [human_experience.md](human_experience.md) | 人工确认的经验与约束 |
| [prompts.md](prompts.md) | 用户维护的工作提示，不作为运行输入或成功证据 |
| [pyproject.toml](pyproject.toml) | Python 包、命令和默认测试目录 |
| [.gitignore](.gitignore) | 临时数据、现场副本、缓存与归档的跟踪边界 |
| [用户文档](docs/user/index.md) | 运行、调参与验收 |
| [开发参考](docs/developer/index.md) | 接口、坐标与实现边界 |
| [tests/](tests/index.md) | 当前自动回归 |
| [config/](config/index.md) | 运行配置 |
| [data/](data/index.md) | 任务输入与原始采集的保存边界 |
| [gene_agent/](gene_agent/index.md) | 当前代码模块与接口 |
| [tools/](tools/index.md) | 长期执行、编辑、查看和诊断命令 |
| [assets/](assets/) | 机器人与物体模型，见下方资产说明 |
| [environment/](environment/) | Linux/Windows 环境定义与依赖冻结 |
| `runs/`（本地） | 执行产物与用途索引，全部不跟踪 |
| [experiments/](docs/developer/maintenance/file_layout.md) | 研究代码归档边界 |

`docs/tmp/` 放临时说明与报告，`archive/` 放一次性代码和开发产物，均不跟踪。
第三方源码和安装产生的 build/cache 由依赖环境管理，不作为项目任务输入维护。

## 运行资产

- `assets/objects/bowl_visual.obj` 与 `bowl_concept.json`：碗外观和功能几何。
- `assets/objects/microwave_concept.json`：当前场景/编辑器读取的门把手几何。
- `assets/objects/microwave_calibrated/`：微波炉 URDF、MJCF、Concept 和关联 mesh，整包保留。
- `assets/robots/realman/`：Realman 模型、原 mesh 与规划配置；机器人选择见 config/index.md。

资产包内部相对路径不为减少文件数而拆散；visual、规划碰撞和物理接触用途不同。
