# setup_spark.ps1
# Helper script to set up Spark environment on Windows

Write-Host "==================================================="
Write-Host "       Setting up PySpark Environment              "
Write-Host "==================================================="

# 1. Install PySpark via pip
Write-Host "1. Installing PySpark library..."
pip install pyspark

# 2. Check for Java (Crucial!)
try {
    $javaVer = java -version 2>&1
    Write-Host "2. Java found: $javaVer"
}
catch {
    Write-Host "X Java NOT found. Spark requires Java 8, 11, or 17."
    Write-Host "  Please download OpenJDK: https://adoptium.net/"
    Write-Host "  AFTER installing, restart your terminal."
    exit
}

# 3. Setup Winutils (Critical for Windows Spark)
$hadoopDir = "$env:USERPROFILE\hadoop"
if (-not (Test-Path $hadoopDir)) {
    Write-Host "3. Setting up Hadoop compatibility (winutils)..."
    New-Item -ItemType Directory -Force -Path "$hadoopDir\bin" | Out-Null
    
    # Download winutils.exe from a reliable github repo (cdarlint/winutils)
    $url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.2.2/bin/winutils.exe"
    $output = "$hadoopDir\bin\winutils.exe"
    
    Write-Host "   Downloading winutils.exe..."
    Invoke-WebRequest -Uri $url -OutFile $output
    
    # Set Environment Variable
    [System.Environment]::SetEnvironmentVariable("HADOOP_HOME", $hadoopDir, [System.EnvironmentVariableTarget]::User)
    $currentPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
    if ($currentPath -notlike "*$hadoopDir\bin*") {
        [System.Environment]::SetEnvironmentVariable("Path", "$currentPath;$hadoopDir\bin", [System.EnvironmentVariableTarget]::User)
    }
    Write-Host "   HADOOP_HOME set to $hadoopDir"
}
else {
    Write-Host "3. Hadoop utils already present."
}

Write-Host "==================================================="
Write-Host "Spark Setup Steps Completed."
Write-Host "NOTE: You may need to RESTART your terminal for Env Vars to update."
Write-Host "==================================================="
