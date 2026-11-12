#!/usr/bin/env python3
"""Forecast pump.fun top-five market caps deterministically.

The hello input is processed as one batch of orders per slot.  This program
uses only local marks, so network access, randomness, and wall-clock time do
not affect its forecast.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
NAMED_MASS = 0.65
SHORTLIST_SIZE = 12
TEMPERATURE = 0.22
RANK_SHIFT = 0.35
MICRO = 1_000_000


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def cap_log_sol(lamports):
    return math.log10(int(lamports)) - 9.0


def round_down(value):
    return math.floor(max(value, 0.0) * MICRO) / MICRO


def normal_survival(value, mean, deviation):
    if value == -math.inf:
        return 1.0
    if value == math.inf:
        return 0.0
    scaled = (value - mean) / (deviation * math.sqrt(2.0))
    return 0.5 * math.erfc(scaled)


def load_marks(path):
    marks = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            mark = json.loads(line)
            if mark.get("ok") is True and isinstance(mark.get("rows"), list):
                marks.append(mark)
    marks.sort(key=lambda mark: (mark.get("at", ""), str(mark.get("slot", ""))))
    return marks


def row_rank(row):
    try:
        return int(row["rank"])
    except (KeyError, TypeError, ValueError):
        return 10**9


def prior_cap(points, cutoff):
    chosen = None
    for observed_at, cap in points:
        if observed_at > cutoff:
            break
        chosen = cap
    return chosen


def validate_task(task):
    try:
        places = int(task["places"])
        edges = [float(edge) for edge in task["size"]["edges"]]
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit("task lacks valid places or size edges") from error
    if places < 1:
        raise SystemExit("task requests no forecast places")
    if not edges or any(not math.isfinite(edge) for edge in edges):
        raise SystemExit("task has no finite size edges")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        raise SystemExit("task size edges must be strictly increasing")
    return places, edges


def output_name(task):
    name = task.get("output", "forecast.json")
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise SystemExit("task output must be a simple filename")
    return name


def build_history(marks):
    history = {}
    for mark in marks:
        observed_at = parse_time(mark["at"])
        for row in mark["rows"]:
            mint = row.get("mint")
            try:
                cap = int(row.get("market_cap_lamports"))
            except (TypeError, ValueError):
                continue
            if isinstance(mint, str) and mint and cap > 0:
                history.setdefault(mint, []).append((observed_at, cap))
    for points in history.values():
        points.sort()
    return history


def score_coins(last_mark, history, last_time):
    coins = []
    seen = set()
    rows = sorted(last_mark["rows"], key=lambda row: (row_rank(row), row.get("mint", "")))
    for row in rows:
        mint = row.get("mint")
        if not isinstance(mint, str) or not mint or mint in seen or mint not in history:
            continue
        try:
            current_cap = int(row.get("market_cap_lamports"))
        except (TypeError, ValueError):
            continue
        if current_cap <= 0:
            continue
        seen.add(mint)
        points = history[mint]
        cap_30m = prior_cap(points, last_time - timedelta(minutes=30))
        cap_1h = prior_cap(points, last_time - timedelta(hours=1))
        growth_30m = math.log10(current_cap / cap_30m) if cap_30m else 0.0
        growth_1h = math.log10(current_cap / cap_1h) if cap_1h else 0.0
        momentum = 0.5 * clamp(growth_30m, -1.0, 1.0)
        momentum += 0.5 * clamp(growth_1h, -1.0, 1.0)
        graduated = row.get("venue") == "pool" and bool(row.get("complete"))
        current_size = cap_log_sol(current_cap)
        coins.append({
            "mint": mint,
            "score": current_size + 0.7 * momentum + 0.12 * float(graduated),
            "predicted_size": current_size + 0.5 * clamp(growth_1h, -1.0, 1.0),
        })
    coins.sort(key=lambda coin: (-coin["score"], coin["mint"]))
    for rank, coin in enumerate(coins, start=1):
        coin["forecast_rank"] = rank
    return coins


def place_probabilities(coins, places):
    rows = []
    shortlist = coins[:SHORTLIST_SIZE]
    for place in range(1, places + 1):
        shifted = [
            coin["score"] - RANK_SHIFT * abs(coin["forecast_rank"] - place)
            for coin in shortlist
        ]
        maximum = max(shifted)
        weights = [math.exp((value - maximum) / TEMPERATURE) for value in shifted]
        total = sum(weights)
        for coin, weight in zip(shortlist, weights):
            probability = round_down(NAMED_MASS * weight / total)
            if probability:
                rows.append({"mint": coin["mint"], "place": place, "p": probability})
    return rows


def size_probabilities(coins, places, edges, deviation):
    rows = []
    for place, coin in enumerate(coins[:places], start=1):
        for bin_number in range(len(edges) + 1):
            lower = -math.inf if bin_number == 0 else edges[bin_number - 1]
            upper = math.inf if bin_number == len(edges) else edges[bin_number]
            probability = round_down(
                normal_survival(lower, coin["predicted_size"], deviation)
                - normal_survival(upper, coin["predicted_size"], deviation)
            )
            if probability:
                rows.append({"place": place, "bin": bin_number, "p": probability})
    return rows


def main():
    with (INPUT_DIR / "task.json").open(encoding="utf-8") as handle:
        task = json.load(handle)
    places, edges = validate_task(task)
    marks = load_marks(INPUT_DIR / "history.ndjson")
    if not marks:
        raise SystemExit("history contains no successful marks")

    last_mark = marks[-1]
    last_time = parse_time(last_mark["at"])
    coins = score_coins(last_mark, build_history(marks), last_time)
    if len(coins) < places:
        raise SystemExit("last mark has fewer usable coins than requested places")

    reads = task.get("reads", [])
    horizon_hours = 0.0
    if reads:
        horizon_hours = max(0.0, (parse_time(reads[0]) - last_time).total_seconds() / 3600.0)
    deviation = min(0.65, 0.35 + 0.03 * horizon_hours)

    forecast = {
        "probabilities": place_probabilities(coins, places),
        "top5": [coin["mint"] for coin in coins[:places]],
        "sizes": size_probabilities(coins, places, edges, deviation),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name(task)
    output_path.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
