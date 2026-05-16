import os
import glob
from scipy.io import loadmat
from datetime import datetime

log_delay_files = sorted(glob.glob("data/eager1_log_delay_*.mat"))

for f in log_delay_files:
    # Get file modification timestamp
    mtime = os.path.getmtime(f)
    dt = datetime.fromtimestamp(mtime)
    
    # Load and check what's inside
    d = loadmat(f)
    
    print(f"{os.path.basename(f):<30} Modified: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Keys in file: {[k for k in d.keys() if not k.startswith('__')]}")
    print()