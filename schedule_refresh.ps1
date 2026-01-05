# schedule_refresh.ps1
# Registers the "MovieRecsRefresh" task to run daily at 3:00 AM.
# Make sure you run this as Administrator!

$TaskName = "MovieRecsRefresh"
$PythonPath = (Get-Command python).Source
$ScriptPath = "$PSScriptRoot\daily_refresh.py"


$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument "$ScriptPath" -WorkingDirectory $PSScriptRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At 3am
$Structure = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Structure -Description "Daily TMDB Refresh & Model Rebuild"

Write-Host "Task '$TaskName' registered successfully!"
Write-Host "It will run daily at 3:00 AM."
Write-Host "To run manually: Start-ScheduledTask -TaskName '$TaskName'"
