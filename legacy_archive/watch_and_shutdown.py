#!/usr/bin/env python3
"""
watch_and_shutdown.py
=====================
Monitors the Qwen2.5-Coder-32B download process.
Once the download finishes and all 14 shards are verified 100% intact,
it flushes disk buffers and issues a scheduled system shutdown (with a 1-minute grace period).
"""

import os
import sys
import time
import subprocess

TARGET_DIR = "./models/Qwen2.5-Coder-32B-Instruct"
LOG_FILE = "./watch_and_shutdown.log"

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(formatted + "\n")

def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def verify_download_complete() -> bool:
    """Runs download_model.py once to verify all files are present and match remote metadata."""
    log("[*] Verifying download completeness via download_model.py...")
    cmd = [sys.executable, "download_model.py", "--model", "32b"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        log("[✓] Model download verification PASSED!")
        return True
    else:
        log(f"[!] Verification returned code {res.returncode}. Output:\n{res.stdout}\n{res.stderr}")
        return False

def main():
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    log("=" * 60)
    log("   AUTO-SHUTDOWN WATCHER ACTIVATED")
    log("=" * 60)
    if pid:
        log(f"[*] Monitoring download process PID: {pid}")

    while True:
        if pid and is_pid_running(pid):
            time.sleep(15)
            continue

        # If PID is no longer running or was not specified, verify model completeness
        log("[*] Process is no longer running or checking status...")
        if verify_download_complete():
            log("[✓] Download is 100% complete and fully verified.")
            log("[*] Flushing filesystem cache (sync)...")
            subprocess.run(["sync"], check=False)
            
            log("[*] Executing system shutdown in 60 seconds (shutdown -h +1)...")
            shutdown_res = subprocess.run(
                ["shutdown", "-h", "+1", "Model download 100% complete. System shutting down in 1 minute."],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            log(f"[*] Shutdown command output: {shutdown_res.stdout} {shutdown_res.stderr}")
            log("[✓] Auto-shutdown watcher finished its mission. Good night!")
            break
        else:
            log("[!] Download not fully complete yet. Sleeping 15s before re-checking/retrying...")
            time.sleep(15)

if __name__ == "__main__":
    main()
