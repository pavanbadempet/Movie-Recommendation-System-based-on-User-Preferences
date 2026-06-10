"""
Local PySpark / Hadoop Environment Setup for Windows.

PySpark is notoriously difficult to run natively on Windows due to Hadoop
dependencies (winutils) and modern Java (17+) reflection restrictions.

This script acts as an auto-configuring bootstrapper. Import it at the top
of your local ETL jobs to dynamically inject all required environment variables
and download Hadoop binaries if they are missing.
"""

import logging
import os
from pathlib import Path
import sys
import urllib.request

logger = logging.getLogger(__name__)

HADOOP_VERSION = "3.2.2"
WINUTILS_BASE_URL = f"https://raw.githubusercontent.com/cdarlint/winutils/master/hadoop-{HADOOP_VERSION}/bin/"


def configure_local_spark():
    """Automatically configures PySpark for the local OS."""

    # 1. Fix Java 17+ / Java 26+ Reflection Errors
    # PySpark's memory manager relies on internal JVM APIs blocked in modern Java.
    java_opens = [
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED",
    ]

    java_options_str = " ".join(java_opens)
    os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-java-options "{java_options_str}" pyspark-shell'
    os.environ["JAVA_HOME"] = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"

    # 2. Windows specific Hadoop setup
    if sys.platform == "win32":
        hadoop_home = Path(__file__).resolve().parent.parent / ".hadoop"
        bin_dir = hadoop_home / "bin"

        # Download winutils.exe and hadoop.dll if missing
        if not (bin_dir / "winutils.exe").exists() or not (bin_dir / "hadoop.dll").exists():
            logger.info(f"Windows detected: Downloading Hadoop {HADOOP_VERSION} binaries to {bin_dir}...")
            bin_dir.mkdir(parents=True, exist_ok=True)

            try:
                urllib.request.urlretrieve(WINUTILS_BASE_URL + "winutils.exe", bin_dir / "winutils.exe")
                urllib.request.urlretrieve(WINUTILS_BASE_URL + "hadoop.dll", bin_dir / "hadoop.dll")
                logger.info("Hadoop binaries downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download winutils. PySpark may crash when writing files: {e}")
                return

        # Inject into OS Environment
        os.environ["HADOOP_HOME"] = str(hadoop_home)
        os.environ["PATH"] = f"{bin_dir!s};{os.environ.get('PATH', '')}"
        logger.info(f"HADOOP_HOME dynamically set to {hadoop_home}")
        logger.info(f"JAVA_HOME dynamically set to {os.environ['JAVA_HOME']}")

    logger.info("Local PySpark environment configured successfully.")


# Auto-configure when imported (not just when run directly)
configure_local_spark()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    configure_local_spark()
