<#
训练看门狗：训练进程被中断时自动断点续训。

为什么需要（2026-10-09）：
  v9 停在第 59 轮、v9b 停在第 17 轮，两次的 .err 都是空的——不是崩溃，
  是**外部中断**（这台笔记本会休眠/关机，后台进程随之消失）。
  训练本身支持断点续训，所以只要有人盯着"进程没了就重启"，就能把训练跑完。

用法：
  # 在项目根目录
  powershell -ExecutionPolicy Bypass -File tools\train_watchdog.ps1 `
      -Script bench\train_v9.py -ScriptArgs "120 --data datasets\train_v9b --name v9b" `
      -TargetEpochs 60

它会：
  · 启动训练；
  · 每 60 秒看一次 results.csv 是否在推进、训练进程是否还在；
  · 发现"进程没了但轮数没到目标"就重新用 --resume 拉起；
  · 达到目标轮数或超过最大重启次数就退出，并写一份汇总日志。
#>
param(
    [string]$Python     = "trafficlab-3d\env\Scripts\python.exe",
    [string]$Script     = "bench\train_v9.py",
    [string]$ScriptArgs = "120 --data datasets\train_v9b --name v9b",
    [string]$RunName    = "v9b",
    # 默认设得比训练脚本的 epochs 更大：让训练跑到它自己的早停（patience）为止。
    # 早前默认 60，结果看门狗在 61 轮就退出了，训练跑到 75 轮被外部中断时
    # 已经没人看守——这是设计缺陷，已修正。
    [int]$TargetEpochs  = 1000,
    [int]$MaxRestarts   = 30,
    [int]$PollSeconds   = 60,
    [int]$StallMinutes  = 8
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
$env:YOLO_CONFIG_DIR = Join-Path $root ".cache\ultralytics"

$csv     = Join-Path $root "outputs\train\$RunName\results.csv"
$summary = Join-Path $root "outputs\train_${RunName}_watchdog.log"

function Write-Log($msg) {
    $line = "{0}  {1}" -f (Get-Date -Format "MM-dd HH:mm:ss"), $msg
    $line | Tee-Object -FilePath $summary -Append
}

function Get-Epochs {
    if (-not (Test-Path $csv)) { return 0 }
    $n = (Get-Content $csv | Measure-Object -Line).Lines - 1
    if ($n -lt 0) { return 0 }
    return $n
}

function Get-CsvAgeMinutes {
    if (-not (Test-Path $csv)) { return 9999 }
    return ((Get-Date) - (Get-Item $csv).LastWriteTime).TotalMinutes
}

function Get-TrainerProcesses {
    # 只认"命令行里带着本训练脚本 + 本次 RunName"的进程。
    # 早前用 Get-Process python 判断，会把同时跑的评测脚本也算进来，
    # 于是训练死了却以为它还活着，看门狗形同虚设。
    $marker = (Split-Path -Leaf $Script)
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($marker) -and $_.CommandLine.Contains($RunName) }
}

Write-Log "看门狗启动：目标 $TargetEpochs 轮，最多重启 $MaxRestarts 次"

for ($i = 0; $i -lt $MaxRestarts; $i++) {
    $done = Get-Epochs
    if ($done -ge $TargetEpochs) {
        Write-Log "已达到 $done 轮（目标 $TargetEpochs），看门狗退出"
        break
    }

    $trainers = @(Get-TrainerProcesses)
    $running  = $trainers.Count -gt 0
    $fresh   = (Get-CsvAgeMinutes) -lt $StallMinutes

    if ($running -and $fresh) {
        # 正常训练中，等下一轮检查
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    if ($running -and -not $fresh) {
        Write-Log "训练进程在，但 results.csv 已 $([int](Get-CsvAgeMinutes)) 分钟没更新 —— 视为卡死"
    } else {
        Write-Log "训练进程不在了（已完成 $done 轮）—— 续训"
    }

    # 只有确认没有训练进程时才拉起，避免两个进程抢 GPU
    if (-not $running) {
        $log = Join-Path $root "outputs\train_${RunName}_watchdog_run$i.log"
        $err = Join-Path $root "outputs\train_${RunName}_watchdog_run$i.err"
        $args = @($Script) + ($ScriptArgs -split "\s+") + "--resume"
        Write-Log "启动：$Python $($args -join ' ')"
        Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $root `
            -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 90          # 给它时间起来
    } else {
        Start-Sleep -Seconds $PollSeconds
    }
}

$final = Get-Epochs
Write-Log "看门狗结束：实际 $final 轮（目标 $TargetEpochs）"
Write-Log "汇总见 $summary；训练日志见 outputs\train\$RunName\results.csv"
