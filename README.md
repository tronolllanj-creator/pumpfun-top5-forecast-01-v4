# pumpfun-top5-forecast-01

A deterministic forecast entry for the five largest new pump.fun SOL coins by market cap at the future answer reads.

`main.py` reads `task.json` and `history.ndjson`, scores the latest ranked coins using current log-cap, 30-minute and one-hour momentum, and a graduation bonus. It assigns place probabilities with a rank-shifted softmax over the top 12 candidates while reserving 35% unnamed mass for new arrivals. Market-cap size bins use a horizon-widened normal distribution in log10 SOL market cap.

Run locally:

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/tmp/out python3 main.py
```

The program writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (or `/output/forecast.json`) and uses only the Python standard library.
