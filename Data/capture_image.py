#!/usr/bin/env python
"""
Pull the newest phone photo via ADB and save it under a filename
that is exactly the FEN (with / → - so Windows accepts it).

Run once per position; re‑run for the next shot.
Tested on Windows 10 + OnePlus 11R + ADB in PATH.
"""

import os
import re
import subprocess
import sys

# ---------- EDIT ME ---------- #
OUTPUT_DIR = r""# Replace with your actual path
CAMERA_DIR = "/sdcard/DCIM/Camera"        # OnePlus default (Edit if different on your phone)
ADB        = "adb"                        # full path if adb.exe isn't on PATH
# --------------------------------

ILLEGAL = r'<>:"/\\|?*'                   # Windows‑forbidden characters

def adb_shell(cmd: str) -> str:
    out = subprocess.check_output([ADB, "shell"] + cmd.split())
    return out.decode(errors="ignore").strip()

def newest_photo() -> str:
    for name in adb_shell(f'ls -t "{CAMERA_DIR}"').splitlines():
        if name.lower().endswith((".jpg", ".jpeg")):
            return name
    raise RuntimeError("No JPG files found in phone camera directory.")

def adb_pull(remote: str, local: str):
    subprocess.check_call([ADB, "pull", remote, local])

def fen_to_filename(fen: str) -> str:
    """Return a Windows‑safe filename derived from the FEN (no extension)."""
    name = fen.replace("/", "-")          # only change required for Windows
    if any(ch in ILLEGAL for ch in name):
        bad = [ch for ch in name if ch in ILLEGAL]
        raise ValueError(f"Illegal character(s) in FEN for Windows filenames: {bad}")
    return name

def check_device():
    lines = subprocess.check_output([ADB, "devices"]).decode().splitlines()[1:]
    if not any(l.strip().endswith("device") for l in lines):
        raise RuntimeError("No ADB device found – enable USB debugging and trust the PC.")

if __name__ == "__main__":
    check_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get FEN from stdin or first CLI arg
    fen = sys.argv[1] if len(sys.argv) > 1 else input("Enter FEN for this position:\n> ").strip()
    if not fen:
        sys.exit("No FEN given – aborting.")

    try:
        base = fen_to_filename(fen)
    except ValueError as e:
        sys.exit(str(e))

    remote = f"{CAMERA_DIR}/{newest_photo()}"
    local  = os.path.join(OUTPUT_DIR, f"{base}.jpg")

    print(f"Pulling → {local}")
    adb_pull(remote, local)
    print("Saved ✔  (filename is exactly the FEN, with / turned into -)")