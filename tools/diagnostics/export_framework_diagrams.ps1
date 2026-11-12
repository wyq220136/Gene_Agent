param([string]$OutputDir = 'artifacts/framework_2026-10-27')
$ErrorActionPreference = 'Stop'
$out = [IO.Path]::GetFullPath((Join-Path (Get-Location) $OutputDir))
[IO.Directory]::CreateDirectory($out) | Out-Null
$script:pages = @()
$script:items = @()
function Text($x,$y,$w,$h,$s,$size=22,$color='#263449',$bold=$false) {
  $script:items += @{kind='text';x=$x;y=$y;w=$w;h=$h;text=$s;size=$size;color=$color;bold=$bold}
}
function Box($x,$y,$w,$h,$title,$body,$fill='#EEF4FA',$stroke='#CAD7E5') {
  $script:items += @{kind='box';x=$x;y=$y;w=$w;h=$h;fill=$fill;stroke=$stroke}
  Text ($x+20) ($y+15) ($w-40) 35 $title 25 '#172B45' $true
  Text ($x+20) ($y+59) ($w-40) ($h-65) $body 20
}
function Arrow($x1,$y1,$x2,$y2,$color='#6D8199',$dash=$false) {
  $script:items += @{kind='arrow';x=$x1;y=$y1;x2=$x2;y2=$y2;color=$color;dash=$dash}
}
function StartPage($title,$subtitle,$num) {
  $script:items=@()
  Text 65 36 1470 55 $title 38 '#172B45' $true
  Text 65 102 1470 38 $subtitle 21 '#53667D'
  Text 65 856 1400 25 'Gene_Agent  /  当前代码与规则的结构快照  /  2026-10-27  /  非运行验收报告' 15 '#66788E'
  Text 1490 852 50 28 $num 18 '#66788E'
}
function EndPage($name,$notes) {$script:pages += @{name=$name;items=$script:items;notes=$notes}}

StartPage 'Gene_Agent 当前框架总览' '外层 Codex 负责开发决策；仓库提供场景、操作、规划执行与反馈工具。编号用于跨页定位。' '01'
Box 65 165 310 125 '输入与验收定义' "操作要求、RGB / 视频、RGB-D`n成功条件、初态分布与约束" '#F1F3F7'
Box 435 165 1100 125 'A  开发 Agent：Codex 会话' "理解任务、组合模块、编写缺失逻辑；读取证据、提出假设、修改与重试。当前未见统一端到端自主调度器。" '#FFF3DA' '#E8C780'
Arrow 375 225 430 225
Box 65 345 340 150 'B  场景与机器人' "重建资产、物体摆位、TCP / 坐标`nvisual、规划碰撞、接触几何"
Box 440 345 340 150 'C  Concept 功能部件' "语义、尺寸、局部坐标`n已有有限轮 Agent 视觉复核"
Box 815 345 340 150 'D  Knowledge 候选' "抓取、接触、操作几何`n可选纯 pose IK 筛选"
Box 1190 345 345 150 'E  Skill 阶段' "接近、操作、撤离与失败处理`n路径、接触及误差预算"
Arrow 405 417 435 417
Arrow 780 417 810 417
Arrow 1155 417 1185 417
Box 1190 565 345 150 'F  Task 编排' "组合 Skill 与检查前后置条件`n逐段传递实际执行状态"
Box 815 565 340 150 'G  Motion / cuRobo' "IK、阶段碰撞与轨迹规划`n正式仿真规划默认使用 cuRobo"
Box 440 565 340 150 'H  MuJoCo 执行' "actuator、夹爪、接触与物体运动`n按实际终态检查操作结果"
Box 65 565 340 150 'I  反馈与证据' "阶段报告、原始返回与错误`n状态、接触、截图 / 视频"
Arrow 1360 495 1360 560
Arrow 1190 640 1160 640
Arrow 815 640 785 640
Arrow 440 640 410 640
Arrow 985 290 985 330 '#BC852A' $true
Arrow 235 565 235 530 '#BC852A' $true
Arrow 235 530 410 530 '#BC852A' $true
Arrow 410 530 410 290 '#BC852A' $true
Text 66 761 1460 60 '实线：构建与运行主链     虚线：反馈交回 Agent / Agent 修改模块     阶段通过后，实际状态继续进入后续阶段' 21 '#53667D'
EndPage '01-framework-overview' '来源：README.md；AGENTS.md；核心说明.md；gene_agent/tasks/runner.py；gene_agent/tasks/policies/microwave_task/simulation_door/session.py；gene_agent/perception_modules/concept_agent.py。主链为职责示意，不表示各后端共用一个完整执行器。'

StartPage 'Agent 如何根据反馈迭代' '区分开发决策、局部实验与最终验收。局部恢复成功不能替代 fresh reset 的完整试验。' '02'
Box 65 178 430 150 '1  运行并收集证据  [G–I]' "保存目标与实际终态、planner 返回`n阶段状态、接触与失败画面"
Box 585 178 430 150 '2  定位原因并提出假设  [A]' "区分几何、IK、碰撞、路径、执行`n和任务后置条件；未知归因保持未知" '#FFF3DA' '#E8C780'
Box 1105 178 430 150 '3  修改最小必要部分  [B–G]' "修改候选、阶段逻辑、配置或实现`n记录改动、原因与预期改善" '#FFF3DA' '#E8C780'
Arrow 495 252 575 252
Arrow 1015 252 1095 252
Arrow 1320 328 1320 400
Box 1105 410 430 155 '4  局部对照实验  [J]' "同检查点比较少量参数候选`n或只探测下一阶段；保留失败样本"
Box 585 410 430 155 '5  完整重跑与分布评测  [K]' "fresh reset、固定 seed 与判定标准`n统计成功率、失败类型和验证范围"
Box 65 410 430 155 '6  固化可复现交付' "代码与最终生效配置、查看命令`n输入分布、证据与能力边界" '#EAF4EE' '#B5D1BE'
Arrow 1105 487 1025 487
Arrow 585 487 505 487
Arrow 800 410 800 345 '#BC852A' $true
Text 603 348 390 36 '未达标：证据返回第 2 步' 20 '#956616'
Arrow 1105 550 1050 550 '#BC852A' $true
Arrow 1050 550 1050 370 '#BC852A' $true
Arrow 1050 370 840 370 '#BC852A' $true
Text 1080 585 450 35 '局部通过后，再进入完整评测' 20 '#53667D'
Box 65 660 720 135 '当前已有的自动化工具' "阶段执行与检查点；门实验参数比较；部分任务多 seed 评测。`nConcept：生成、校验、叠图、最多两轮视觉复核。" '#EEF4FA'
Box 815 660 720 135 '当前仍由外层 Agent 承担' "失败归因、下一轮修改与实验选择、验收停止决策。`n工具覆盖范围依任务而异，尚非统一的新任务自治流水线。" '#FFF3DA' '#E8C780'
EndPage '02-agent-feedback-loop' '来源：tools/run/run_experiment.py；tools/run/evaluate_experiment.py；gene_agent/tasks/policies/microwave_task/simulation_door/evaluation.py；tools/run/evaluate_sweep_cube.py；gene_agent/feedback/failure.py。黄色表示当前主要依靠外层开发 Agent 的职责。门参数比较一次1–20组，补丁范围受限。'

StartPage '问题定位与修改对齐' '左侧是当前问题的排查入口；右侧是建议的目标行为，用于后续讨论，尚未全部实现。' '03'
Text 80 166 390 40 '问题表现' 24 '#172B45' $true
Text 510 166 470 40 '定位模块与证据' 24 '#172B45' $true
Text 1040 166 480 40 '建议改成什么样' 24 '#172B45' $true
$rows = @(
 @('抓错部位 / 普遍不可达','B–D：坐标、TCP、功能部件、候选','检查场景、FK、叠图与候选淘汰原因'),
 @('接近碰撞 / 轨迹规划失败','E、G：阶段约束与原始 planner 返回','显式路径类型、接触白名单和失败分类'),
 @('规划通过，但滑落 / 推不动','H：实际接触、跟踪和物体位移','检查物理后置条件，记录仿真辅助'),
 @('单 Skill 成功，串联后失败','F：上段实际终态与下段起点','校验状态连续性，完整传递操作状态'),
 @('失败后反复调参，原因不清','I、A：证据完整性与迭代历史','保存证据、假设、修改和对照结果'),
 @('局部成功，完整任务不稳定','J、K：检查点试验与全程评测','固定样本和标准，fresh reset 完整评测')
)
$y=223
foreach($r in $rows) {
  $script:items += @{kind='box';x=65;y=$y;w=1470;h=77;fill= $(if(($y-223)/90 % 2 -eq 0){'#F1F5F9'}else{'#FFFFFF'});stroke='#E2E8F0'}
  Text 82 ($y+18) 390 50 $r[0] 23 '#263449' $true
  Text 510 ($y+18) 490 50 $r[1] 21
  Text 1040 ($y+18) 475 50 $r[2] 21
  $y+=90
}
Text 80 790 1450 40 '讨论记录：模块编号 / 当前行为 / 原始证据 / 原因假设 / 修改内容 / 目标行为 / 验收方法' 22 '#956616' $true
EndPage '03-diagnosis-and-changes' '对齐建议，非已实现功能清单。来源：AGENTS.md；gene_agent/tasks/runner.py；gene_agent/tasks/checks.py；gene_agent/feedback/failure.py；gene_agent/tasks/policies/microwave_task/simulation_door/session.py。'

function Escape($s) { [Security.SecurityElement]::Escape([string]$s) }
foreach($p in $pages) {
  $svg = [Text.StringBuilder]::new()
  [void]$svg.AppendLine('<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img">')
  [void]$svg.AppendLine('<title>Gene_Agent 当前框架与反馈迭代</title><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto-start-reverse"><path d="M0,0 L9,4.5 L0,9" fill="context-stroke"/></marker></defs><rect width="1600" height="900" fill="#FFFFFF"/>')
  foreach($i in $p.items) {
    if($i.kind -eq 'box') { [void]$svg.AppendLine("<rect x='$($i.x)' y='$($i.y)' width='$($i.w)' height='$($i.h)' rx='12' fill='$($i.fill)' stroke='$($i.stroke)'/>") }
    elseif($i.kind -eq 'arrow') {
      $dash=if($i.dash){"stroke-dasharray='8 6'"}else{''}
      [void]$svg.AppendLine("<line x1='$($i.x)' y1='$($i.y)' x2='$($i.x2)' y2='$($i.y2)' stroke='$($i.color)' stroke-width='2.5' $dash marker-end='url(#arrow)'/>")
    } else {
      $weight=if($i.bold){600}else{400}
      $lines=([string]$i.text).Split("`n")
      [void]$svg.AppendLine("<text font-family='Microsoft YaHei, Noto Sans CJK SC, sans-serif' font-size='$($i.size)' font-weight='$weight' fill='$($i.color)'>")
      $lineY=$i.y+$i.size
      foreach($line in $lines) {[void]$svg.AppendLine("<tspan x='$($i.x)' y='$lineY'>$(Escape $line)</tspan>");$lineY+=$i.size*1.48}
      [void]$svg.AppendLine('</text>')
    }
  }
  [void]$svg.AppendLine('</svg>')
  [IO.File]::WriteAllText((Join-Path $out ($p.name+'.svg')),$svg.ToString(),[Text.UTF8Encoding]::new($false))
}

function Rgb($hex) { $h=$hex.TrimStart('#'); return [Convert]::ToInt32($h.Substring(0,2),16)+256*[Convert]::ToInt32($h.Substring(2,2),16)+65536*[Convert]::ToInt32($h.Substring(4,2),16) }
$app=New-Object -ComObject PowerPoint.Application
$pres=$app.Presentations.Add(0)
try {
  $pres.PageSetup.SlideWidth=1200
  $pres.PageSetup.SlideHeight=675
  foreach($p in $pages) {
    $slide=$pres.Slides.Add($pres.Slides.Count+1,12)
    foreach($i in $p.items) {
      $x=$i.x*.75;$y=$i.y*.75
      if($i.kind -eq 'box') {
        $shape=$slide.Shapes.AddShape(5,$x,$y,$i.w*.75,$i.h*.75)
        $shape.Fill.ForeColor.RGB=(Rgb $i.fill);$shape.Line.ForeColor.RGB=(Rgb $i.stroke)
      } elseif($i.kind -eq 'arrow') {
        $shape=$slide.Shapes.AddLine($x,$y,$i.x2*.75,$i.y2*.75)
        $shape.Line.ForeColor.RGB=(Rgb $i.color);$shape.Line.Weight=1.8;$shape.Line.EndArrowheadStyle=3
        if($i.dash){$shape.Line.DashStyle=4}
      } else {
        $shape=$slide.Shapes.AddTextbox(1,$x,$y,$i.w*.75,$i.h*.75)
        $tf=$shape.TextFrame;$tf.MarginLeft=0;$tf.MarginRight=0;$tf.MarginTop=0;$tf.MarginBottom=0
        $tf.WordWrap=0
        $tf.TextRange.Text=$i.text
        $tf.TextRange.Font.Name='Microsoft YaHei'
        $tf.TextRange.Font.NameFarEast='Microsoft YaHei'
        $tf.TextRange.Font.Size=$i.size*.75
        $tf.TextRange.Font.Color.RGB=(Rgb $i.color)
        if($i.bold){$tf.TextRange.Font.Bold=-1}
      }
    }
    $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$p.notes
    $slide.Export((Join-Path $out ($p.name+'.png')),'PNG',1600,900)
  }
  $pres.SaveAs((Join-Path $out 'Gene_Agent-framework.pptx'),24)
} finally { $pres.Close();$app.Quit() }
Get-ChildItem $out | Select-Object Name,Length
