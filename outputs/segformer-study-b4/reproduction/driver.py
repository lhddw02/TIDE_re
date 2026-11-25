"""Supervise the requested training chain and restart interrupted stages."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path('work/segstudy-b4')
for stage in ('real','synthetic','finetuned'):
    if stage=='synthetic':
        print('Waiting for the verified FreeMask-style filtered synthetic manifest',flush=True)
        while not (root/'filtered/complete.json').exists():
            time.sleep(5)
    attempts=0
    while True:
        result=subprocess.run([sys.executable,'-Bu',str(root/'train.py'),'--stage',stage])
        if result.returncode == 0:
            break
        attempts+=1
        if result.returncode not in (-15,-9,137,143) or attempts>=5:
            raise SystemExit(f'{stage} exited with code {result.returncode}, inspect the training log')
        print(f'Restarting {stage} from its last checkpoint after exit {result.returncode}',flush=True)
        time.sleep(3)
