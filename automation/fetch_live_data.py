#!/usr/bin/env python3
"""
Fetcher for Indian stock daily data using yfinance.

Behavior:
 - If a ticker CSV exists in the target outdir, the script fetches only new rows since the last date and appends them.
 - If no CSV exists for a ticker, the script downloads the full history (period='max').
 - By default the script will read tickers from the filenames found in the `--tickers-from-dir` directory (strip .csv).

Usage examples:
  python fetch_live_data.py --tickers "RELIANCE.NS,TCS.NS" --outdir indian_data
  python fetch_live_data.py --tickers-from-dir indian_data --outdir indian_data

Notes:
 - This script uses the `yfinance` package (pip install yfinance).
 - It is safe to run daily; it will only append new rows when CSVs already exist.
"""
import os
import sys
import argparse
import time
import logging
from datetime import datetime, timedelta
import pandas as pd

try:
    import yfinance as yf
except Exception:
    print("ERROR: yfinance is required. Install with `pip install yfinance`.")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_tickers_from_dir(dirpath: str):
    """Read .csv filenames from a directory and return cleaned ticker names."""
    import re
    if not os.path.isdir(dirpath):
        return []
    files = os.listdir(dirpath)
    found = set()
    for fname in files:
        if not fname.lower().endswith('.csv'):
            continue
        base = os.path.splitext(fname)[0]
        # strip trailing _YYYY-MM-DD if present
        m = re.search(r'_(\d{4}-\d{2}-\d{2})$', base)
        if m:
            ticker = base[:m.start()]
        else:
            ticker = base
        if ticker:
            found.add(ticker)
    return sorted(found)


def fetch_and_update(ticker: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    target = os.path.join(outdir, f"{ticker}.csv")

    if os.path.exists(target):
        try:
            # Read without forcing parse_dates to avoid ValueError when the
            # 'date' column is named differently (e.g. 'Date') or missing.
            existing = pd.read_csv(target)

            # Normalize date column name (case-insensitive)
            date_col = None
            for col in existing.columns:
                if col.lower() == 'date':
                    date_col = col
                    break
            if date_col and date_col != 'date':
                existing = existing.rename(columns={date_col: 'date'})

            if 'date' not in existing.columns:
                logger.warning('%s exists but has no date column; will replace', target)
                existing = None
            else:
                # ensure date values are converted to date objects
                try:
                    existing['date'] = pd.to_datetime(existing['date']).dt.date
                except Exception:
                    logger.exception('Could not parse dates in existing %s; will replace', target)
                    existing = None
        except Exception:
            logger.exception('Failed to read existing %s; will replace', target)
            existing = None
    else:
        existing = None

    if existing is not None and len(existing) > 0:
        # Be defensive: coerce to datetime, drop NaT and non-parseable values
        try:
            parsed_dates = pd.to_datetime(existing['date'], errors='coerce').dt.date
            parsed_dates = parsed_dates.dropna()
            if parsed_dates.empty:
                logger.warning('%s existing date column could not be parsed; will replace', target)
                existing = None
            else:
                last_date = parsed_dates.max()
        except Exception:
            logger.exception('Error parsing dates for existing %s; will replace', target)
            existing = None
        
    if existing is not None and len(existing) > 0:
        # At this point parsed last_date should be available
        start = last_date + timedelta(days=1)
        if start >= datetime.utcnow().date():
            logger.info('%s is up-to-date (last_date=%s)', ticker, last_date)
            return target
        logger.info('Fetching %s from %s to today', ticker, start)
        df = yf.download(ticker, start=start.isoformat(), end=(datetime.utcnow().date() + timedelta(days=1)).isoformat(), progress=False)
    else:
        logger.info('No existing data for %s, downloading full history', ticker)
        df = yf.download(ticker, period='max', progress=False)

    if df is None or df.empty:
        logger.warning('No data returned for %s', ticker)
        return None

    df.reset_index(inplace=True)
    # yfinance uses 'Date' index -> 'Date' column, normalize to 'date' lower-case
    # Keep open, high, low, close, adj close, volume
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    if 'Adj Close' in df.columns:
        df = df.rename(columns={'Adj Close': 'adjusted_close'})
    if 'Adj Close' not in df.columns and 'adjusted_close' not in df.columns:
        df['adjusted_close'] = df.get('Close', df.iloc[:, 0])

    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Keep necessary columns
    keep = ['date', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    df = df[[c for c in keep if c in df.columns]]
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values('date')

    if existing is not None and len(existing) > 0:
        # append any new rows
        existing_dates = set(pd.to_datetime(existing['date']).dt.date.tolist())
        new_rows = df[~df['date'].isin(existing_dates)]
        if new_rows.empty:
            logger.info('No new rows for %s', ticker)
        else:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.sort_values('date')
            combined.to_csv(target, index=False)
            logger.info('Appended %d rows to %s', len(new_rows), target)
    else:
        df.to_csv(target, index=False)
        logger.info('Wrote %s (%d rows)', target, len(df))

    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fetch daily stock data using yfinance')
    parser.add_argument('--tickers', required=False, help="Comma-separated tickers")
    parser.add_argument('--outdir', required=False, default='indian_data', help='Directory to write per-ticker CSVs (default: indian_data)')
    parser.add_argument('--single_csv', required=False, help='Path to write a combined latest CSV (optional)')
    parser.add_argument('--pause', type=float, default=1.0, help='Seconds to pause between API calls (default 1s)')
    parser.add_argument('--tickers-from-dir', dest='tickers_from_dir', required=False, help='Read existing tickers from CSV filenames in this directory')
    args = parser.parse_args(argv)

    tickers = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',') if t.strip()]
    elif args.tickers_from_dir:
        tickers = read_tickers_from_dir(args.tickers_from_dir)
    else:
        logger.error('No tickers provided and no tickers_from_dir specified')
        sys.exit(2)

    if not tickers:
        logger.error('No tickers found')
        sys.exit(2)

    combined = []
    errors = []
    for i, ticker in enumerate(tickers):
        try:
            logger.info('Processing %s (%d/%d)', ticker, i + 1, len(tickers))
            path = fetch_and_update(ticker, args.outdir)
            if path:
                df = pd.read_csv(path)
                df['ticker'] = ticker
                combined.append(df)
        except Exception as e:
            logger.exception('Failed to fetch %s: %s', ticker, e)
            errors.append({'ticker': ticker, 'error': str(e)})
        time.sleep(args.pause)

    if args.single_csv and combined:
        all_df = pd.concat(combined, ignore_index=True)
        try:
            all_df.to_csv(args.single_csv, index=False)
            logger.info('Wrote combined CSV %s', args.single_csv)
        except Exception:
            logger.exception('Failed to write combined CSV %s', args.single_csv)

    if errors:
        logger.error('Finished with errors: %s', errors)
        sys.exit(1)

    logger.info('Fetch completed successfully')


if __name__ == '__main__':
    main()
