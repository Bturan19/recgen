#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

echo "== Adult (binary classification) =="
uv run python -u experiments/adult.py --pooling last
echo
echo "== House Prices (regression) =="
uv run python -u experiments/house_prices.py --pooling last
echo
echo "== IMDB (text classification) =="
uv run python -u experiments/imdb.py --pooling last
echo
echo "== Results =="
uv run python scripts/summarize.py
