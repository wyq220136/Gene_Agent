param([string]$OutputDir = 'artifacts/agent_development_2026-10-27')
$ErrorActionPreference = 'Stop'
# 复用既有矢量绘图与 PowerPoint 原生形状导出器，只替换本次图的内容。
$base = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'export_framework_diagrams.ps1'),[Text.Encoding]::UTF8)
$prefix = $base.Substring($base.IndexOf('$ErrorActionPreference'), $base.IndexOf("StartPage 'Gene_Agent 当前框架总览'")-$base.IndexOf('$ErrorActionPreference'))
$suffix = $base.Substring($base.IndexOf('function Escape'))
$suffix = $suffix.Replace('Gene_Agent-framework.pptx','Gene_Agent-development-loop.pptx')
Invoke-Expression $prefix

StartPage 'Agent 自动开发任务的工作闭环' '前提：仿真 benchmark 已就绪。此图展示开发行为，场景重建与部署不在本图范围内。' '01'
Box 65 156 1470 112 '已有 benchmark：开发输入与固定裁判' "任务目标、reset / step / 状态接口、机器人与场景、成功判定、初态分布、开发与验收样本。实现与调参须遵守同一契约。" '#F1F3F7'
Box 65 328 340 155 'A  读取任务契约' "理解操作目标与允许修改范围`n检查接口、判定与现有可复用模块" '#FFF3DA' '#E8C780'
Box 440 328 340 155 'B  编写首版实现' "复用或新增 Knowledge / Skill`n组织 Task，形成可运行基线" '#FFF3DA' '#E8C780'
Box 815 328 340 155 'C  运行开发试验' "固定 seed，从 reset 或检查点开始`n调用 cuRobo 规划与 MuJoCo 执行"
Box 1190 328 345 155 'D  读取失败证据' "阶段结果、实际状态、planner 返回`n接触、错误、截图与视频"
Arrow 405 405 433 405
Arrow 780 405 808 405
Arrow 1155 405 1183 405
Arrow 1360 483 1360 555
Box 1190 565 345 155 'E  诊断与选择修改' "找最早失败环节，提出原因假设`n决定改参数、代码或补充诊断" '#FFF3DA' '#E8C780'
Box 815 565 340 155 'F  实施最小修改' "修改候选、阶段、约束或编排`n同样本对照，检查假设是否成立" '#FFF3DA' '#E8C780'
Box 440 565 340 155 'G  完整回归与验收' "局部通过后，fresh reset 重跑`n在固定分布 / seed 上统计成功率"
Box 65 565 340 155 'H  固化开发结果' "保存代码、有效配置与证据`n交付命令、验证范围和失败边界" '#EAF4EE' '#B5D1BE'
Arrow 1190 642 1162 642
Arrow 815 642 787 642
Arrow 440 642 412 642
Arrow 985 565 985 490 '#BC852A' $true
Text 1005 512 175 30 '修改后重试' 18 '#956616'
Arrow 610 720 610 770 '#BC852A' $true
Arrow 610 770 1360 770 '#BC852A' $true
Arrow 1360 770 1360 727 '#BC852A' $true
Text 790 775 570 30 '验收未达标：携失败样本返回诊断' 18 '#956616'
Text 65 285 1470 30 '黄色：当前由外层 Codex 决策与修改    蓝色：仓库工具支持，覆盖依任务而异    绿色：开发交付' 19 '#53667D'
EndPage '01-development-loop' '当前工作方式示意，非已经实现的统一自主控制器。benchmark 已就绪是假设。Agent 按现有会话规则执行读取、修改与实验；工具：gene_agent/feedback；tools/run/run_experiment.py；tools/run/evaluate_experiment.py；tools/run/evaluate_sweep_cube.py。不得擅自改成功条件或剔除困难样本。'

StartPage '失败后，Agent 如何决定下一步' '自动开发的核心是用证据选择修改，而不是重复运行同一条任务。以下是当前规则要求的决策方式。' '02'
Box 65 164 1470 110 'D  同一轮试验的输入证据' "配置与版本、seed、目标和实际状态、最早失败阶段、原始 planner 返回、接触对、截图 / 视频。证据缺失时先补诊断。"
Arrow 800 274 800 308
Text 65 311 390 35 'E  诊断分支' 23 '#172B45' $true
Text 510 311 475 35 '证据指向什么？' 23 '#172B45' $true
Text 1040 311 475 35 'F  下一轮改哪里？' 23 '#172B45' $true
$rows=@(
 @('观测不足 / 归因未知','状态、图片或碰撞对象不完整','补日志、渲染或局部探测，再判断'),
 @('目标几何 / IK 问题','候选、TCP、实际 FK 与目标误差','改 Knowledge / 阶段误差策略'),
 @('起点 / 目标 / 路径碰撞','碰撞分类、对象与阶段允许接触对','改阶段目标、碰撞策略或路径约束'),
 @('执行 / 接触 / 物体终态失败','跟踪、双侧接触、滑移、物体位移','改 Skill、控制参数或接触逻辑'),
 @('单段成功 / 全程串联失败','前后段实际状态及后置条件','改 Task 状态交接和阶段过渡')
)
$y=365
foreach($r in $rows) {
 $script:items+=@{kind='box';x=65;y=$y;w=1470;h=67;fill='#F1F5F9';stroke='#E2E8F0'}
 Text 84 ($y+16) 390 42 $r[0] 22 '#172B45' $true
 Text 510 ($y+16) 475 42 $r[1] 21
 Text 1040 ($y+16) 475 42 $r[2] 21
 $y+=78
}
Text 80 777 1450 45 '每轮输出：原因假设、证据引用、修改差异、预期效果、对照样本与结果。此记录当前尚未统一自动生成。' 21 '#956616' $true
EndPage '02-feedback-to-development-action' '基于 AGENTS.md 的开发诊断要求，以及 feedback/motion_diagnostics.py、feedback/failure.py、tasks/runner.py、tasks/checks.py。表格是决策规则，不表示已编码成自动分类器。近似 IK 需显式阶段策略，不得把失败 IK 关节直接当成功轨迹。benchmark 允许修改范围外的场景和判定不能擅改。'

StartPage '当前 Agent 框架：已有内容与待补连接' '研发对齐视图。左侧是当前依赖外层会话的工作；中间是可调用工具；右侧是建议补齐的开发控制层。' '03'
Box 65 177 440 360 '当前：外层 Codex 会话' "A  读取任务与代码，制定开发步骤`nB  编写或修改任务实现`nD  阅读结构化证据与实际图像`nE  推理原因、选择下一轮动作`nF  修改并调用实验工具`nG  判断是否继续或交付" '#FFF3DA' '#E8C780'
Box 580 177 440 360 '当前：仓库提供实验能力' "C  规划与 MuJoCo 实际执行`nD  阶段状态、失败证据和回放`nF  门实验检查点与参数对照`nG  部分任务固定 seed 评测`n`n各任务接入程度不同，非统一接口"
Arrow 505 330 570 330
Arrow 580 425 515 425 '#BC852A' $true
Box 1095 177 440 360 '建议：统一开发控制层' "任务契约与允许编辑范围`n迭代历史：证据、假设、修改、结果`n实验队列、版本和预算管理`n统一结果读取与失败样本索引`n开发 / 回归 / 验收的执行门槛`n停止、恢复与交付条件" '#F1F3F7' '#C8CFD9'
Text 1100 553 430 34 '建议项：当前未见统一实现' 21 '#956616' $true
Box 65 638 710 155 '先对齐的关键连接' "D → E：Agent 能否拿到足够且一致的失败证据？`nE → F：每次修改是否有明确原因与可验证假设？`nF → C：新实验能否与上一版在同样本上比较？" '#EEF4FA'
Box 815 638 720 155 '验收边界' "G → H：局部恢复成功后，仍须完整 reset 验收。`n固定 benchmark 判定与分母，保留全部失败样本。`n仿真辅助、CPU 诊断与正式 cuRobo 结果分别标记。" '#EAF4EE' '#B5D1BE'
EndPage '03-current-tools-and-missing-links' '当前与建议分开。未见统一开发控制器是本次已检查核心入口的结论，并非对全部仓库文件作形式证明。现有 Concept 模型复核属于独立局部自动循环；此页聚焦 benchmark 上的操作任务开发，不将其混同完整自主研发。'

Invoke-Expression $suffix
Compress-Archive -LiteralPath (Get-ChildItem $out -Filter '*.svg').FullName -DestinationPath (Join-Path $out 'Gene_Agent-development-SVG.zip') -Force
