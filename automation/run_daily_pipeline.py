#!/usr/bin/env python3
"""Run the daily pipeline: fetch (optional), train, predict, and produce a single JSON payload for automation.

Produces: ./data/automation_payload.json with keys:
 - generated_at
 - report (contents of automation_report.json)
 - top_stocks (list from top_stocks_YYYY-MM-DD.json)
 - top_stocks_text (string contents of top_stocks.txt)
 - best_checkpoint_copied (path if available)

This script is safe to run without an AlphaVantage key (it will skip fetch).
"""
import subprocess
import os
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
INDIAN_DATA = ROOT / 'indian_data'


def run_cmd(cmd, cwd=None, env=None):
    print('RUN:', ' '.join(cmd))
    # Capture output using UTF-8 and replace errors so Windows CP1252 doesn't raise
    res = subprocess.run(cmd, cwd=cwd or ROOT, env=env or os.environ, capture_output=True, text=True, encoding='utf-8', errors='replace')
    print('EXIT', res.returncode)
    if res.stdout:
        print('STDOUT:\n', res.stdout[:10000])
    if res.stderr:
        print('STDERR:\n', res.stderr[:10000])
    return res.returncode, res.stdout, res.stderr


def maybe_fetch():
    # Fetcher now uses yfinance and does not require an API key.
    cmd = ['python', str(ROOT / 'automation' / 'fetch_live_data.py'), '--tickers-from-dir', str(INDIAN_DATA), '--outdir', str(INDIAN_DATA), '--pause', '1']
    return run_cmd(cmd)


def run_trainer():
    cmd = ['python', str(ROOT / 'automation' / 'train_and_report.py'), '--out-report', str(DATA_DIR / 'automation_report.json')]
    return run_cmd(cmd)


def run_predictor():
    cmd = ['python', str(ROOT / 'predict_now.py')]
    return run_cmd(cmd)


def collate_payload():
    payload = {'generated_at': datetime.utcnow().isoformat()}

    report_path = DATA_DIR / 'automation_report.json'
    if report_path.exists():
        try:
            payload['report'] = json.loads(report_path.read_text(encoding='utf-8'))
        except Exception as e:
            payload['report'] = {'error': f'could not read report: {e}'}
    else:
        payload['report'] = None

    # Find latest top_stocks_*.json in data/
    top_json = None
    for p in sorted(DATA_DIR.glob('top_stocks_*.json'), key=lambda p: p.name, reverse=True):
        top_json = p
        break
    if top_json and top_json.exists():
        try:
            payload['top_stocks'] = json.loads(top_json.read_text(encoding='utf-8')).get('top_stocks')
        except Exception as e:
            payload['top_stocks'] = {'error': f'could not read {top_json}: {e}'}
    else:
        payload['top_stocks'] = None

    top_txt = DATA_DIR / 'top_stocks.txt'
    if top_txt.exists():
        payload['top_stocks_text'] = top_txt.read_text(encoding='utf-8')
    else:
        payload['top_stocks_text'] = None

    # Propagate best_checkpoint_copied if present
    try:
        if payload.get('report') and isinstance(payload['report'], dict):
            payload['best_checkpoint_copied'] = payload['report'].get('best_checkpoint_copied')
        else:
            payload['best_checkpoint_copied'] = None
    except Exception:
        payload['best_checkpoint_copied'] = None

    out = DATA_DIR / 'automation_payload.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print('WROTE', out)
    return out


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # 1) fetch if key present
    maybe_fetch()

    # 2) trainer
    run_trainer()

    # 3) predictor
    run_predictor()

    # 4) collate
    collate_payload()


if __name__ == '__main__':
    main()
