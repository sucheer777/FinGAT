#!/usr/bin/env python3
"""
Wrapper to run training and produce a small JSON report with the best checkpoint path and metric.

Usage: python train_and_report.py [--train-args "--epochs 1"] [--checkpoints-dir /app/checkpoints] [--out-report /data/automation_report.json]

This script runs the repository's `train.py` with any provided args, then searches for .ckpt files in `checkpoints_dir`,
parses `val_mrr` from filenames when available, picks the best, and writes the report JSON with fields:
  {"best_checkpoint": "/path/to/ckpt", "best_metric": 0.123}

It prints the JSON to stdout as well so automation tools can parse it.
"""
import argparse
import subprocess
import json
import sys
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--train-args', default='', help='Extra args to pass to train.py')
parser.add_argument('--checkpoints-dir', default=None, help='Directory to search for checkpoints (default: ./checkpoints then /app/checkpoints then /data/checkpoints)')
parser.add_argument('--out-report', default='/data/automation_report.json', help='Path to write JSON report')
args = parser.parse_args()

# Run training
train_cmd = ['python', 'train.py']
if args.train_args:
    # split safely
    import shlex
    train_cmd += shlex.split(args.train_args)

print('Running training:', ' '.join(train_cmd), file=sys.stderr)
ret = subprocess.call(train_cmd)
train_error = None
if ret != 0:
    # don't exit immediately; still attempt to find any checkpoints produced and write a report
    # capture the train return code so downstream automation can detect failure
    train_error = {'error': 'train_failed', 'return_code': ret}

# locate checkpoints
possible = []
if args.checkpoints_dir:
    possible.append(Path(args.checkpoints_dir))
possible += [Path('./checkpoints'), Path('/app/checkpoints'), Path('/data/checkpoints')]
ckpt_files = []
for p in possible:
    if p.is_dir():
        ckpt_files += [p / f for f in os.listdir(p) if f.endswith('.ckpt')]

best = None
best_metric = None
metric_name = 'val_mrr'
pattern = re.compile(r'val_mrr=([0-9]*\.?[0-9]+)')
for f in ckpt_files:
    m = pattern.search(f.name)
    if m:
        try:
            v = float(m.group(1))
        except Exception:
            continue
        if best is None or v > best_metric:
            best = f
            best_metric = v

# fallback: if no metric found, use newest file
if best is None and ckpt_files:
    best = max(ckpt_files, key=lambda p: p.stat().st_mtime)
    best_metric = None

report = {
    'best_checkpoint': str(best) if best else None,
    'best_metric': best_metric,
}

# ensure output dir exists
out_path = Path(args.out_report)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as fh:
    # include any train error details in the report for debugging
    out = dict(report)
    if train_error:
        out.update({'train_error': train_error})
    json.dump(out, fh)

# If training failed, print the error JSON and exit with non-zero so orchestrators detect failure.
if train_error:
    print(json.dumps({'train_error': train_error}), flush=True)
    # also print the report for additional context
    print(json.dumps(out), flush=True)
    # attempt to copy checkpoint if present (best may have been produced before error)
    try:
        if best:
            src_path = Path(best)
            if not src_path.is_absolute():
                src_path = Path.cwd() / src_path
            dest_dir = out_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / src_path.name
            import shutil
            if src_path.exists():
                shutil.copy2(src_path, dest_path)
                print(f"COPIED_BEST_CKPT: {dest_path}")
    except Exception as e:
        print(f"WARN: failed copying best checkpoint after error: {e}")
    sys.exit(train_error.get('return_code', 1))

# Success path: print token and report, copy best checkpoint for convenience
print('TRAIN_OK')
print(json.dumps(report))
try:
    if best:
        src_path = Path(best)
        if not src_path.is_absolute():
            src_path = Path.cwd() / src_path
        dest_dir = out_path.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name
        import shutil
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            # update report file with copied path
            report['best_checkpoint_copied'] = str(dest_path)
            with open(out_path, 'w') as fh:
                json.dump(report, fh)
            print(f"COPIED_BEST_CKPT: {dest_path}")
        else:
            print(f"WARN: best checkpoint not found at expected path: {src_path}")
except Exception as e:
    print(f"WARN: failed copying best checkpoint: {e}")

sys.exit(0)
