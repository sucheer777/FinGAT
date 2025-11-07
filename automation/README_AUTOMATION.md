# Automation Build and Validation

This folder contains the automation for daily fetch → train → notify workflows for FinGAT.

## Files added
- `automation/fetch_live_data.py` - script to fetch stock data and write per-ticker CSVs and a combined CSV.
- `automation/requirements-fetch.txt` - pip deps for fetcher.
- `automation/docker/Dockerfile.fetcher` - Dockerfile for the fetcher container.
- `automation/docker/Dockerfile.trainer` - Dockerfile for the trainer container (copies project's `train.py`).
- `automation/n8n_workflow.json` - example n8n workflow to orchestrate fetch → train → notify.
- `docker-compose.yml` (repo root) - brings up postgres, mlflow, n8n, fetcher, trainer.

## Build Docker Images (local tests)
# from project root
docker build -f automation/docker/Dockerfile.fetcher -t custom_fetcher:latest .
docker build -f automation/docker/Dockerfile.trainer -t custom_trainer:latest .

## Run Fetcher (test)
# Ensure ALPHA_VANTAGE_KEY is set in env; map ./data so outputs are visible
docker run --rm -e ALPHA_VANTAGE_KEY="$env:ALPHA_VANTAGE_KEY" -v ${PWD}/data:/data custom_fetcher:latest --tickers 'RELIANCE.NS,TCS.NS' --outdir /data/indian_data --single_csv /data/latest_stock_data.csv

## Run Trainer (test)
# Quick dev run to validate the trainer container
docker run --rm -v ${PWD}/data:/app custom_trainer:latest --fast_dev_run True

## Full stack
# from project root
docker-compose up -d

## Validate
- Ensure CSVs appear in `./data/indian_data/` (map to container path `/data/indian_data`).
- Confirm MLflow UI (depends on mlflow service image/config) at `http://localhost:5000` if mlflow is configured.
- Import `automation/n8n_workflow.json` into your n8n instance and test a manual run.
- Set SMTP and email addresses in n8n for notification nodes.

## Notes and next steps
- Replace `custom_fetcher:latest` and `custom_trainer:latest` with your built images or adjust the workflow to use `docker-compose run`.
- For production, store secrets (API keys, SMTP, DB credentials) securely (Vault, Secrets Manager, or Docker-compose .env file).
- Consider using a paid data provider or caching to avoid Alpha Vantage rate limits.
