<#
  把某个批次里复核过的标注打包，交回给负责人。
  由 打包复核结果.bat 调用：
      powershell -ExecutionPolicy Bypass -File tools\dataset_batches\pack_results.ps1 -Batch batch_01

  只打包 labels\ + review_order.txt + 标注说明.txt（图片不用回传），
  产物放在仓库根目录：复核结果_<批次>_<日期>.zip
  刻意不用 Compress-Archive：它在部分机器上会因 PSModulePath 冲突加载失败，
  这里直接用 .NET 的 ZipFile，最省事也最稳。
#>

param(
    [string]$Batch = '',
    [string]$DataName = 'videoset_02_review'
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

function Good($m) { Write-Host "[完成] $m" -ForegroundColor Green }
function Bad($m) { Write-Host "[失败] $m" -ForegroundColor Red }

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $scriptDir '..\..')).Path
$reviewRoot = Join-Path $root "datasets\$DataName"

if (-not $Batch) {
    $found = @(Get-ChildItem -LiteralPath $reviewRoot -Directory -ErrorAction SilentlyContinue)
    if ($found.Count -eq 1) {
        $Batch = $found[0].Name
        Write-Host "只找到 $Batch，就打包它。" -ForegroundColor Yellow
    } else {
        Bad "没说清楚是哪个批次，而且 $reviewRoot 下不是刚好一个批次目录。"
        Write-Host "      现有批次：$(($found | ForEach-Object Name) -join ', ')"
        exit 1
    }
}

$src = Join-Path $reviewRoot $Batch
$labels = Join-Path $src 'labels'
if (-not (Test-Path -LiteralPath $labels)) {
    Bad "找不到 $labels —— 批次号对吗？或者这批数据还没下载？"
    exit 1
}

$today = Get-Date -Format 'yyyyMMdd'
$out = Join-Path $root "复核结果_${Batch}_${today}.zip"
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out -Force }

Add-Type -AssemblyName System.IO.Compression | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null

$files = @(Get-ChildItem -LiteralPath $labels -File -Filter *.txt)
if ($files.Count -eq 0) {
    Bad "$labels 里没有标注文件，没什么可打包的。"
    exit 1
}

$zip = [IO.Compression.ZipFile]::Open($out, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($f in $files) {
        [IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $f.FullName, "labels/$($f.Name)") | Out-Null
    }
    foreach ($extra in @('review_order.txt', '标注说明.txt')) {
        $p = Join-Path $src $extra
        if (Test-Path -LiteralPath $p) {
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $p, $extra) | Out-Null
        }
    }
} finally {
    $zip.Dispose()
}

$mb = [math]::Round((Get-Item -LiteralPath $out).Length / 1MB, 2)
Good "$Batch 的复核结果已打包：$out （$($files.Count) 个标注文件，$mb MB）"
Write-Host "    把这个 zip 发回给负责人即可（不含图片）。"
