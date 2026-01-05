# setup_hadoop_env.ps1
# Sets HADOOP_HOME permanently for the user.
# Run this once.

$HadoopPath = "C:\Users\pavan\hadoops"

Write-Host "Setting HADOOP_HOME to $HadoopPath..."
[System.Environment]::SetEnvironmentVariable("HADOOP_HOME", $HadoopPath, [System.EnvironmentVariableTarget]::User)

$BinPath = Join-Path $HadoopPath "bin"
$CurrentPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)

if ($CurrentPath -notlike "*$BinPath*") {
    Write-Host "Adding $BinPath to User Path..."
    $NewPath = "$CurrentPath;$BinPath"
    [System.Environment]::SetEnvironmentVariable("Path", $NewPath, [System.EnvironmentVariableTarget]::User)
}
else {
    Write-Host "Bin path already in User Path."
}

Write-Host "Done! Please Restart your Terminal for changes to take effect."
