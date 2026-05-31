# Windows Görev Zamanlayıcı — günlük pipeline (19:00)
# Yönetici olarak çalıştırmanız gerekebilir.

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "scripts\run_daily.py"
$TaskName = "BorsaAI-DailyPipeline"

if (-not (Test-Path $Python)) {
    Write-Error "venv bulunamadi: $Python — once: python -m venv .venv"
    exit 1
}

$Action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Script`"" -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At "19:00"
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Force

Write-Host "Gorev olusturuldu: $TaskName"
Write-Host "Her gun 19:00 — $Script"
Write-Host "Loglar: $ProjectRoot\logs\"
Write-Host "Kaldirmak icin: Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
