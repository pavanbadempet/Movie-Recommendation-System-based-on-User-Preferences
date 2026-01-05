import os
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)

def install():
    hadoop_home = r"C:\Users\pavan\hadoops"
    bin_dir = os.path.join(hadoop_home, "bin")
    
    if not os.path.exists(bin_dir):
        os.makedirs(bin_dir)
        logging.info(f"Created directory: {bin_dir}")
        
    url = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.2.0/bin/winutils.exe"
    dest = os.path.join(bin_dir, "winutils.exe")
    
    logging.info(f"Downloading winutils.exe from {url}...")
    try:
        urllib.request.urlretrieve(url, dest)
        logging.info(f"Success! Saved to {dest}")
        print("WINUTILS_INSTALLED")
    except Exception as e:
        logging.error(f"Failed to download: {e}")

if __name__ == "__main__":
    install()
