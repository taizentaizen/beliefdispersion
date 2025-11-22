import math
import random
import statistics
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os

random.seed(42)

@dataclass
class Series:
    dates: List[str]
    values: List[float]

@dataclass
class IntervalForecast:
    point: float
    lower: float
    upper: float


def generate_synthetic_data(n_days: int = 800) -> Dict[str, Series]:
    """Generate synthetic GSPC and VIX series to mimic behavior when real data unavailable."""
    dates, spx, vix = [], [], []
    price = 3000.0
    vix_level = 18.0
    for i in range(n_days):
        date = f"1990-01-01+{i}"
        shock = random.gauss(0, 0.01)
        vol_shock = random.gauss(0, 1.0)
        vix_level = max(10.0, vix_level * 0.98 + abs(shock) * 220 + vol_shock * 0.1)
        price *= math.exp(0.0002 + shock)
        dates.append(date)
        spx.append(price)
        vix.append(vix_level)
    return {"spx": Series(dates, spx), "vix": Series(dates, vix)}


def compute_returns(prices: List[float]) -> List[float]:
    returns = [0.0]
    for i in range(1, len(prices)):
        returns.append(math.log(prices[i] / prices[i - 1]))
    return returns


def rolling_vol(returns: List[float], window: int) -> List[float]:
    vols = []
    win = deque(maxlen=window)
    for r in returns:
        win.append(r)
        if len(win) < window:
            vols.append(0.0)
        else:
            mean = sum(win) / len(win)
            var = sum((x - mean) ** 2 for x in win) / len(win)
            vols.append(math.sqrt(var))
    return vols


def cumulative_forward(returns: List[float], horizon: int) -> List[float]:
    cum = [0.0] * len(returns)
    for i in range(len(returns) - horizon):
        total = 0.0
        for j in range(1, horizon + 1):
            total += returns[i + j]
        cum[i] = total
    return cum


def build_features(spx: Series, vix: Series, horizon: int = 1, n_lags: int = 5):
    rets = compute_returns(spx.values)
    rv5 = rolling_vol(rets, 5)
    rv21 = rolling_vol(rets, 21)
    cum_target = cumulative_forward(rets, horizon)
    features, targets, dates = [], [], []
    for i in range(n_lags, len(rets) - horizon):
        feat = []
        for k in range(n_lags):
            feat.append(rets[i - k])
        feat.append(rv5[i])
        feat.append(rv21[i])
        feat.append(vix.values[i])
        feat.append(vix.values[i] - vix.values[i - 1] if i > 0 else 0.0)
        features.append(feat)
        targets.append(cum_target[i])
        dates.append(spx.dates[i])
    return dates, features, targets


def split_features(features: List[List[float]]):
    base = [f[:-2] for f in features]
    belief = features
    return base, belief


def fit_linear(x: List[List[float]], y: List[float]) -> List[float]:
    if not x:
        return []
    d = len(x[0])
    xtx = [[0.0 for _ in range(d)] for _ in range(d)]
    xty = [0.0 for _ in range(d)]
    for row, target in zip(x, y):
        for i in range(d):
            xty[i] += row[i] * target
            for j in range(d):
                xtx[i][j] += row[i] * row[j]
    lam = 1e-3
    for i in range(d):
        xtx[i][i] += lam
    aug = [xtx[i] + [xty[i]] for i in range(d)]
    for i in range(d):
        pivot = aug[i][i] if aug[i][i] != 0 else 1e-12
        for j in range(i, d + 1):
            aug[i][j] /= pivot
        for k in range(d):
            if k == i:
                continue
            factor = aug[k][i]
            for j in range(i, d + 1):
                aug[k][j] -= factor * aug[i][j]
    beta = [aug[i][-1] for i in range(d)]
    return beta


def predict_linear(beta: List[float], row: List[float]) -> float:
    return sum(b * r for b, r in zip(beta, row))


def bagged_linear(train_x, train_y, test_rows, n_models=4):
    preds = []
    for _ in range(n_models):
        idx = [random.randrange(len(train_x)) for _ in range(len(train_x))]
        sample_x = [train_x[i] for i in idx]
        sample_y = [train_y[i] for i in idx]
        beta = fit_linear(sample_x, sample_y)
        preds.append([predict_linear(beta, row) for row in test_rows])
    averaged = []
    for j in range(len(test_rows)):
        averaged.append(sum(pred[j] for pred in preds) / n_models)
    return averaged


def interval_from_residuals(residuals: List[float], alpha: float = 0.1) -> Tuple[float, float]:
    sorted_res = sorted(residuals)
    lower_idx = max(0, int(alpha / 2 * len(sorted_res)) - 1)
    upper_idx = min(len(sorted_res) - 1, int((1 - alpha / 2) * len(sorted_res)))
    return sorted_res[lower_idx], sorted_res[upper_idx]


def evaluate_model(dates, feats, targets, predictor, train_start, step=30):
    results = []
    n = len(targets)
    start = train_start
    while start < n - 1:
        train_x = feats[:start]
        train_y = targets[:start]
        test_slice = slice(start, min(start + step, n))
        test_x = feats[test_slice]
        test_y = targets[test_slice]
        base_beta = fit_linear(train_x, train_y)
        base_preds = [predict_linear(base_beta, row) for row in train_x]
        residuals = [y - p for y, p in zip(train_y, base_preds)]
        lower_res, upper_res = interval_from_residuals(residuals)
        preds = predictor(train_x, train_y, test_x)
        for point, y_true, date in zip(preds, test_y, dates[test_slice]):
            results.append((date, y_true, IntervalForecast(point, point + lower_res, point + upper_res)))
        start += step
    return results


def metrics(results: List[Tuple[str, float, IntervalForecast]], nominal=0.9):
    errors = [y - r.point for _, y, r in results]
    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    coverage = sum(1 for (_, y, r) in results if r.lower <= y <= r.upper) / len(results)
    widths = [r.upper - r.lower for _, _, r in results]
    avg_width = sum(widths) / len(widths)
    alpha = 1 - nominal
    score = 0.0
    for (_, y, r) in results:
        width = r.upper - r.lower
        if y < r.lower:
            score += width + 2 / alpha * (r.lower - y)
        elif y > r.upper:
            score += width + 2 / alpha * (y - r.upper)
        else:
            score += width
    score /= len(results)
    half_widths = [w / 2 for w in widths]
    corr = statistics.correlation(half_widths, [abs(e) for e in errors]) if len(results) > 1 else 0.0
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Coverage": coverage,
        "AvgWidth": avg_width,
        "IntervalScore": score,
        "UncertaintyAbsErrCorr": corr,
    }


def strategy_stats(results: List[Tuple[str, float, IntervalForecast]]):
    cum_point = 1.0
    cum_interval = 1.0
    max_draw_point = 0.0
    max_draw_interval = 0.0
    peak_point = 1.0
    peak_interval = 1.0
    rets_point = []
    rets_interval = []
    for _, y, r in results:
        pos_p = 1 if r.point > 0 else 0
        strat_ret_p = pos_p * y
        cum_point *= math.exp(strat_ret_p)
        peak_point = max(peak_point, cum_point)
        max_draw_point = min(max_draw_point, (cum_point - peak_point) / peak_point)
        rets_point.append(strat_ret_p)
        if r.lower > 0:
            pos_i = 1
        elif r.upper < 0:
            pos_i = -1
        else:
            pos_i = 0
        strat_ret_i = pos_i * y
        cum_interval *= math.exp(strat_ret_i)
        peak_interval = max(peak_interval, cum_interval)
        max_draw_interval = min(max_draw_interval, (cum_interval - peak_interval) / peak_interval)
        rets_interval.append(strat_ret_i)
    def sharpe(ret_list):
        if not ret_list:
            return 0.0
        avg = sum(ret_list) / len(ret_list)
        var = sum((r - avg) ** 2 for r in ret_list) / len(ret_list)
        vol = math.sqrt(var)
        return avg / vol * math.sqrt(252) if vol > 0 else 0.0
    return {
        "CumulativePoint": cum_point,
        "CumulativeInterval": cum_interval,
        "SharpePoint": sharpe(rets_point),
        "SharpeInterval": sharpe(rets_interval),
        "MaxDDPoint": max_draw_point,
        "MaxDDInterval": max_draw_interval,
    }


def summarize_table(label, metrics_dict):
    lines = [f"# {label}", "Metric,Value"]
    for k, v in metrics_dict.items():
        lines.append(f"{k},{v:.4f}")
    return "\n".join(lines)


def ascii_plot(results: List[Tuple[str, float, IntervalForecast]], path: str, n: int = 60):
    sample = results[-n:]
    lines = []
    for date, y, r in sample:
        scale = 100
        center = 50
        point = center + int(r.point * scale)
        actual = center + int(y * scale)
        lower = center + int(r.lower * scale)
        upper = center + int(r.upper * scale)
        width = ["."] * 101
        lower_i = max(0, min(100, lower))
        upper_i = max(0, min(100, upper))
        for i in range(lower_i, upper_i + 1):
            width[i] = "-"
        if 0 <= point <= 100:
            width[point] = "P"
        if 0 <= actual <= 100:
            width[actual] = "A"
        lines.append(f"{date} |" + "".join(width))
    with open(path, "w") as f:
        f.write("\n".join(lines))


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def run_experiment():
    data = generate_synthetic_data()
    dates, feats_all, targets1 = build_features(data["spx"], data["vix"], horizon=1)
    _, feats_all5, targets5 = build_features(data["spx"], data["vix"], horizon=5)
    base1, belief1 = split_features(feats_all)
    base5, belief5 = split_features(feats_all5)
    train_start = 200
    models = {
        "AR1": lambda tx, ty, rows: (lambda b=fit_linear(tx, ty): [predict_linear(b, r) for r in rows])(),
        "BaggedLinear": lambda tx, ty, rows: bagged_linear(tx, ty, rows),
    }
    summaries = []
    ensure_dir("outputs")
    for horizon, feats_base, feats_belief, targets in [
        (1, base1, belief1, targets1),
        (5, base5, belief5, targets5),
    ]:
        for feat_name, feat_set in [("Base", feats_base), ("Belief", feats_belief)]:
            for model_name, fn in models.items():
                res = evaluate_model(dates, feat_set, targets, fn, train_start)
                met = metrics(res)
                strat = strategy_stats(res)
                label = f"h{horizon}_{feat_name}_{model_name}"
                out_lines = summarize_table(label, {**met, **strat})
                path = f"outputs/{label}_metrics.csv"
                with open(path, "w") as f:
                    f.write(out_lines)
                ascii_plot(res, f"outputs/{label}_interval_plot.txt")
                summaries.append((label, met, strat))
    with open("outputs/summary.txt", "w") as f:
        for label, met, strat in summaries:
            f.write(label + "\n")
            for k, v in met.items():
                f.write(f"  {k}: {v:.4f}\n")
            for k, v in strat.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    run_experiment()
