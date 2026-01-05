#!/usr/bin/env python
"""
Single entry point to run both FastAPI backend and Streamlit frontend.
Usage: python run.py
"""
import subprocess
import sys
import time
import os
import socket

# Colors for output
GREEN = "\033[92m"
BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def is_port_in_use(port):
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def kill_process_on_port(port):
    """Kill any process using the specified port (Windows)."""
    try:
        # Find PID using netstat
        result = subprocess.run(
            f'netstat -ano | findstr :{port}',
            shell=True, capture_output=True, text=True
        )
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 5 and f':{port}' in parts[1]:
                    pid = parts[-1]
                    print(f"{YELLOW}[Cleanup]{RESET} Killing process {pid} on port {port}")
                    subprocess.run(f'taskkill /F /PID {pid}', shell=True, 
                                 capture_output=True)
                    time.sleep(1)
                    break
    except Exception as e:
        print(f"{YELLOW}Warning: Could not kill process on port {port}: {e}{RESET}")


def main():
    print(f"{GREEN}🚀 Starting Movie Recommendation System...{RESET}\n")
    
    # Get the directory of this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check and free ports
    for port in [8000, 8501]:
        if is_port_in_use(port):
            print(f"{YELLOW}[Warning]{RESET} Port {port} is in use, attempting to free...")
            kill_process_on_port(port)
    
    time.sleep(1)
    
    processes = []
    
    try:
        # Start FastAPI backend
        print(f"{BLUE}[Backend]{RESET} Starting FastAPI on http://localhost:8000")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", 
             "--host", "0.0.0.0", "--port", "8000"],
            cwd=base_dir,
        )
        processes.append(("Backend", backend))
        
        # Wait for backend to start
        print(f"{BLUE}[Backend]{RESET} Waiting for backend to initialize...")
        time.sleep(4)
        
        if backend.poll() is not None:
            print(f"{RED}[Backend]{RESET} Failed to start! Check for errors above.")
            return
        
        # Start Streamlit frontend
        print(f"{BLUE}[Frontend]{RESET} Starting Streamlit on http://localhost:8501")
        frontend = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
             "--server.port", "8501", "--server.headless", "true"],
            cwd=base_dir,
        )
        processes.append(("Frontend", frontend))
        
        time.sleep(2)
        
        print(f"\n{GREEN}{'='*50}{RESET}")
        print(f"{GREEN}✅ Both services started!{RESET}")
        print(f"   📺 Frontend: http://localhost:8501")
        print(f"   🔌 Backend:  http://localhost:8000/docs")
        print(f"{GREEN}{'='*50}{RESET}")
        print(f"\n{YELLOW}Press Ctrl+C to stop both services{RESET}\n")
        
        # Wait for processes
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"{RED}[{name}] Process exited with code {proc.returncode}{RESET}")
                    raise KeyboardInterrupt
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Shutting down...{RESET}")
        
    finally:
        # Clean up all processes
        for name, proc in processes:
            if proc.poll() is None:
                print(f"{BLUE}[{name}]{RESET} Stopping...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        print(f"{GREEN}✅ All services stopped.{RESET}")


if __name__ == "__main__":
    main()
