# 自动读屏软件 —— 本机环境复检脚本
# 用法： powershell -ExecutionPolicy Bypass -File .\scripts\环境体检.ps1
# 用途： 加内存 / 装完工具后复核，输出与《本机环境体检报告.md》同结构的信息

$ErrorActionPreference = 'SilentlyContinue'
function Head($t) { "`n=== $t ===" }

Head 'OS'
Get-CimInstance Win32_OperatingSystem |
    Select-Object Caption, Version, BuildNumber, OSArchitecture,
        @{n='TotalRAM_GB'; e={[math]::Round($_.TotalVisibleMemorySize/1MB,1)}},
        @{n='FreeRAM_GB';  e={[math]::Round($_.FreePhysicalMemory/1MB,1)}} | Format-List

Head 'CPU'
Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List

Head 'GPU'
Get-CimInstance Win32_VideoController |
    Select-Object Name, DriverVersion, CurrentHorizontalResolution, CurrentVerticalResolution | Format-Table -AutoSize

Head 'nvidia-smi'
$smi = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
if (Test-Path $smi) { & $smi } else { 'nvidia-smi 不存在' }

Head '内存条（关注：数量 / 容量 / 实际频率 是否已达 6000）'
Get-CimInstance Win32_PhysicalMemory |
    Select-Object BankLabel, DeviceLocator, @{n='GB'; e={$_.Capacity/1GB}},
        Speed, ConfiguredClockSpeed, Manufacturer, PartNumber | Format-Table -AutoSize
"DIMM 插槽总数: " + (Get-CimInstance Win32_PhysicalMemoryArray).MemoryDevices

Head '磁盘'
Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' |
    Select-Object DeviceID, @{n='Size_GB'; e={[math]::Round($_.Size/1GB,1)}},
        @{n='Free_GB'; e={[math]::Round($_.FreeSpace/1GB,1)}} | Format-Table -AutoSize

Head '工具链'
foreach ($t in 'python', 'py', 'git', 'ffmpeg', 'nvcc', 'nvidia-smi', 'docker', 'node', 'wsl') {
    $c = Get-Command $t -ErrorAction SilentlyContinue
    '{0,-14} {1}' -f $t, $(if ($c) { $c.Source } else { '-- 未安装 --' })
}
'python 默认版本  ' + (& python --version 2>&1)
'py 已装版本:'
& py -0p 2>&1

Head 'CUDA Toolkit 版本'
Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Name 2>$null

Head '音频端点（播放）'
Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render\*\Properties' -ErrorAction SilentlyContinue |
    ForEach-Object { $_.'{a45c254e-df1c-4efd-8020-67d146a850e0},2' } | Sort-Object -Unique

Head '音频端点（录音，关注 CABLE Output / 立体声混音）'
Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture\*\Properties' -ErrorAction SilentlyContinue |
    ForEach-Object { $_.'{a45c254e-df1c-4efd-8020-67d146a850e0},2' } | Sort-Object -Unique

Head '虚拟声卡 / 虚拟显示器 / 安全软件'
Get-CimInstance Win32_PnPEntity |
    Where-Object { $_.Name -match 'CABLE|VB-Audio|VoiceMeeter|Oray|ToDesk|Parsec|Virtual Display|Idd' } |
    Select-Object Name, Status | Format-Table -AutoSize
Get-CimInstance Win32_Service |
    Where-Object { $_.Name -match 'hips|huorong|defender|360' } |
    Select-Object Name, State | Format-Table -AutoSize

Head '会话类型（Console=本地，截屏安全；RDP/远控=可能黑屏）'
"SESSIONNAME = $env:SESSIONNAME"

Head 'Checklist'
$ram = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize / 1MB
$dimms = @(Get-CimInstance Win32_PhysicalMemory).Count
$spd = (Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1).Speed
$cfg = (Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1).ConfiguredClockSpeed
'{0} 内存 >= 32GB           : {1}' -f $(if ($ram -ge 31) { '[OK]  ' } else { '[TODO]' }), ([math]::Round($ram,1))
'{0} 内存条 >= 2 条(双通道)  : {1}' -f $(if ($dimms -ge 2) { '[OK]  ' } else { '[TODO]' }), $dimms
'{0} XMP 已开(频率=标称)     : 配置 {1} / 标称 {2}' -f $(if ($cfg -ge $spd -and $cfg -gt 4800) { '[OK]  ' } else { '[TODO]' }), $cfg, $spd
'{0} Python 3.11 可用       : {1}' -f $(if ((& py -0p 2>&1) -match '3\.1[012]') { '[OK]  ' } else { '[TODO]' }), '见上方列表'
'{0} Git                    : {1}' -f $(if (Get-Command git -EA SilentlyContinue) { '[OK]  ' } else { '[TODO]' }), '需要'
'{0} FFmpeg                 : {1}' -f $(if (Get-Command ffmpeg -EA SilentlyContinue) { '[OK]  ' } else { '[TODO]' }), 'GPT-SoVITS 需要'
'{0} Docker 未安装(符合预期) : {1}' -f $(if (-not (Get-Command docker -EA SilentlyContinue)) { '[OK]  ' } else { '[WARN]' }), '本项目不需要容器'
'{0} 本地 Console 会话       : {1}' -f $(if ($env:SESSIONNAME -eq 'Console') { '[OK]  ' } else { '[WARN]' }), $env:SESSIONNAME
