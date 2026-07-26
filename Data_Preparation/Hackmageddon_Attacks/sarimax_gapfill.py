#!/usr/bin/env python
"""SARIMAX gap-fill for the 2025 Hackmageddon monthly attack counts.

This script fills gap with per-series SARIMAX models using exogenous regressors derived from PCA of the 37 Reddit War_Conflict_* monthly mention-count series.

MODES:
    holdout     Train on history through --holdout-train-end (default 2024-04). Then,
                forecast May-2024 -> Jan-2025 (the last months with ground truth) and compare MAE/RAE for SARIMAX + Reddit-PCA exogenous regressors (sarimax-exog), same model family, no exogenous regressors (sarima), and seasonal-naive.

    fill        Refit on the full valid history (2011-07 -> 2025-01) and forecast Feb-Dec 2025.
                Wrutes to NoI_monthly_v2_filled_sarimax.csv. With --extend-dataset, additionally writes Cyber_Trend_Forecasting_All_v2_2_sarimax.csv.
"""

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ATTACKS_CSV = SCRIPT_DIR / "NoI_monthly_v2.csv"
REDDIT_CSV = REPO_ROOT / "Data_Preparation" / "Reddit" / "RedditData_v2.csv"
V2_1_CSV = REPO_ROOT / "Data_Preparation" / "Cyber_Trend_Forecasting_All_v2_1.csv"
V2_2_CSV = REPO_ROOT / "Data_Preparation" / "Cyber_Trend_Forecasting_All_v2_2_sarimax.csv"
FILLED_CSV = SCRIPT_DIR / "NoI_monthly_v2_filled_sarimax.csv"

FIRST_MONTH = pd.Period("2011-07", freq="M")
LAST_VALID = pd.Period("2025-01", freq="M")   # last month with real Hackmageddon data
GAP = pd.period_range("2025-02", "2025-12", freq="M")
SEASONAL_PERIOD = 12
N_EXPECTED_ROWS = 174
N_EXPECTED_TARGETS = 988


def load_attacks(path=ATTACKS_CSV):
    df = pd.read_csv(path)
    idx = pd.PeriodIndex(pd.to_datetime(df["Attack-Country"], format="%m/%Y"), freq="M")
    df = df.drop(columns=["Attack-Country"]).astype(float)
    df.index = idx
    assert len(df) == N_EXPECTED_ROWS, f"expected {N_EXPECTED_ROWS} rows, got {len(df)}"
    assert df.index.is_monotonic_increasing
    assert df.index[0] == FIRST_MONTH and df.index[-1] == GAP[-1]
    assert df.shape[1] == N_EXPECTED_TARGETS, f"expected {N_EXPECTED_TARGETS} cols, got {df.shape[1]}"
    assert not df.isna().any().any(), "NaNs in attack data"
    assert (df.loc[GAP[0]:] == 0).all().all(), "gap rows are not all zero -- data changed?"
    assert (df.loc[:LAST_VALID].sum(axis=1) > 0).all(), \
        f"an all-zero month exists before {LAST_VALID} -- the gap has widened?"
    return df


def load_reddit(path=REDDIT_CSV):
    df = pd.read_csv(path)
    idx = pd.PeriodIndex(df["Date"], freq="M")
    df = df.drop(columns=["Date"])
    df.index = idx
    cols = [c for c in df.columns if c.startswith("War_Conflict_")]
    assert len(cols) == 37, f"expected 37 War_Conflict_* cols, got {len(cols)}"
    df = df[cols].astype(float)
    assert df.index.is_monotonic_increasing
    assert df.index[0] <= FIRST_MONTH and df.index[-1] >= GAP[-1], "Reddit data does not cover the needed range"
    assert not df.isna().any().any(), "NaNs in Reddit data"
    return df


def slice_train(series_or_frame, train_start, train_end):
    assert train_end <= LAST_VALID, f"training window may not extend past {LAST_VALID}"
    return series_or_frame.loc[train_start:train_end]


# --- Exogenous regressors: PCA of the 37 Reddit War_Conflict_* series --
def build_exog(reddit, train_start, train_end, full_end, *, n_components=5,
               variance_target=None, standardize=True, output_dir=None,
               pkl_name="pca_model.pkl"):
    """Scaler + PCA fit on the training rows only, transform train_start..full_end."""
    train = slice_train(reddit, train_start, train_end)
    full = reddit.loc[train_start:full_end]
    train_vals = train.to_numpy(dtype=float)
    full_vals = full.to_numpy(dtype=float)

    scaler = None
    if standardize:
        scaler = StandardScaler().fit(train_vals)
        train_vals = scaler.transform(train_vals)
        full_vals = scaler.transform(full_vals)

    n_comp = variance_target if variance_target is not None else n_components
    pca = PCA(n_components=n_comp, svd_solver="full").fit(train_vals)
    scores = pca.transform(full_vals)
    exog = pd.DataFrame(scores, index=full.index,
                        columns=[f"PC{i + 1}" for i in range(scores.shape[1])])
    info = {
        "exog_source": str(REDDIT_CSV.relative_to(REPO_ROOT)),
        "n_input_series": reddit.shape[1],
        "standardize": standardize,
        "n_components": int(pca.n_components_),
        "variance_target": variance_target,
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cumulative_explained_variance": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        "pca_fit_rows": f"{train_start}..{train_end}",
    }
    if output_dir is not None:
        import pickle
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / pkl_name, "wb") as f:
            pickle.dump({"pca": pca, "scaler": scaler, "info": info}, f)
    return exog, info


# -- Per-series models --
def candidate_orders():
    """Small candidate list. Min-AIC on the training window picks one."""
    return [
        dict(order=(0, 1, 1), seasonal_order=(0, 1, 1, 12), trend=None),
        dict(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12), trend=None),
        dict(order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), trend=None),
        dict(order=(2, 1, 2), seasonal_order=(1, 0, 1, 12), trend=None),
        dict(order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), trend="c"),
        dict(order=(1, 1, 0), seasonal_order=(1, 0, 0, 12), trend=None),
        dict(order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), trend=None),
        dict(order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend="c"),
    ]


FALLBACK_ORDERS = [
    dict(order=(0, 1, 1), seasonal_order=(0, 0, 0, 12), trend=None),
    dict(order=(0, 1, 0), seasonal_order=(0, 0, 0, 12), trend=None),
]


def parse_orders_override(spec):
    """Parse --orders like '0,1,1|0,1,1,12 ; 1,1,1|1,0,1,12|c'."""
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = [p.strip() for p in part.split("|")]
        order = tuple(int(x) for x in pieces[0].split(","))
        seasonal = tuple(int(x) for x in pieces[1].split(","))
        trend = pieces[2] if len(pieces) > 2 and pieces[2] else None
        assert len(order) == 3 and len(seasonal) == 4
        out.append(dict(order=order, seasonal_order=seasonal, trend=trend))
    if not out:
        raise ValueError(f"could not parse any orders from {spec!r}")
    return out


@dataclass
class SeriesResult:
    name: str
    method: str          # sarimax-exog | sarima | sparse-seasonal-mean | mean-fallback
    order: "tuple | None" = None
    seasonal_order: "tuple | None" = None
    trend: "str | None" = None
    aic: "float | None" = None          # raw statsmodels AIC of the chosen model
    selection_score: "float | None" = None  # common-sample AIC used for selection
    fallback_used: bool = False
    n_candidates_failed: int = 0
    n_convergence_warnings: int = 0
    forecast: object = None  # pd.Series over the forecast PeriodIndex


def sparse_seasonal_mean(y_train, fc_index):
    """Per-calendar-month mean of the last 36 training months."""
    last36 = y_train.iloc[-36:]
    by_month = last36.groupby(last36.index.month).mean()
    overall = float(last36.mean())
    vals = np.array([float(by_month.get(p.month, overall)) for p in fc_index])
    return np.clip(vals, 0.0, None)


def seasonal_naive(y_train, fc_index):
    """y[t-12]; every needed lag must lie inside the training window."""
    lags = [p - 12 for p in fc_index]
    assert all(lag <= y_train.index[-1] for lag in lags)
    return y_train.loc[lags].to_numpy(dtype=float)


def fit_one_series(name, y_train, exog_train, exog_future, fc_index, *,
                   use_log1p=True, orders=None, sparse_threshold=24.0):
    """Fit one SARIMAX (exog optional) and forecast len(fc_index) steps."""
    method = "sarimax-exog" if exog_train is not None else "sarima"
    if float(y_train.sum()) < sparse_threshold:
        vals = sparse_seasonal_mean(y_train, fc_index)
        return SeriesResult(name=name, method="sparse-seasonal-mean",
                            forecast=pd.Series(vals, index=fc_index))

    orders = orders or candidate_orders()
    z = np.log1p(y_train.to_numpy(dtype=float)) if use_log1p else y_train.to_numpy(dtype=float)

    def build(cand):
        return SARIMAX(z, exog=exog_train, order=cand["order"],
                       seasonal_order=cand["seasonal_order"], trend=cand["trend"],
                       enforce_stationarity=False, enforce_invertibility=False)

    def try_fit(model):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            res = model.fit(disp=False, maxiter=200)
        n_conv = sum(1 for w in wlist if issubclass(w.category, ConvergenceWarning))
        return res, n_conv

    n_failed, n_convwarn, fallback_used = 0, 0, False
    built = []
    for cand in orders:
        try:
            built.append((cand, build(cand)))
        except Exception:
            n_failed += 1
    common_burn = max((m.k_states for _, m in built), default=0)
    assert common_burn < len(z), "training window shorter than the largest state vector"

    best = None  # (cand, res, score)
    for cand, model in built:
        try:
            res, n_conv = try_fit(model)
            n_convwarn += n_conv
            llf_obs = np.asarray(res.llf_obs, dtype=float)
            score = 2.0 * len(res.params) - 2.0 * float(np.sum(llf_obs[common_burn:]))
            if not np.isfinite(score):
                n_failed += 1
                continue
            if best is None or score < best[2]:
                best = (cand, res, score)
        except Exception:
            n_failed += 1
    if best is None:
        fallback_used = True
        for cand in FALLBACK_ORDERS:
            try:
                res, n_conv = try_fit(build(cand))
                n_convwarn += n_conv
                if np.isfinite(res.aic):
                    best = (cand, res, float(res.aic))
                    break
            except Exception:
                continue
    if best is None:
        vals = sparse_seasonal_mean(y_train, fc_index)
        return SeriesResult(name=name, method="mean-fallback", fallback_used=True,
                            n_candidates_failed=n_failed,
                            n_convergence_warnings=n_convwarn,
                            forecast=pd.Series(vals, index=fc_index))

    cand, res, score = best
    fc = np.asarray(res.forecast(steps=len(fc_index), exog=exog_future), dtype=float)
    if use_log1p:
        fc = np.expm1(fc)
    fc = np.clip(fc, 0.0, None)
    blowup_cap = max(50.0 * max(float(y_train.max()), 1.0), 100.0)
    if not np.all(np.isfinite(fc)) or float(np.max(fc)) > blowup_cap:
        fc = sparse_seasonal_mean(y_train, fc_index)
        method += "+blowup-fallback"
        fallback_used = True
    return SeriesResult(name=name, method=method, order=cand["order"],
                        seasonal_order=cand["seasonal_order"], trend=cand["trend"],
                        aic=round(float(res.aic), 2), selection_score=round(score, 2),
                        fallback_used=fallback_used,
                        n_candidates_failed=n_failed, n_convergence_warnings=n_convwarn,
                        forecast=pd.Series(fc, index=fc_index))


# --- Metrics (same as compute_metrics_per_horizon in Transformer_Pipeline/Evaluate_VisionPP.py) ---
def mae(preds, trues):
    return float(np.mean(np.abs(preds - trues)))


def rae(preds, trues):
    denom = float(np.mean(np.abs(trues - np.mean(trues))))
    return float(np.mean(np.abs(preds - trues)) / denom) if denom > 0 else float("nan")


def metrics_rows(scope, series, method, preds, trues, horizons):
    """preds/trues: (n_series, n_steps). Rows for each horizon + Overall."""
    rows = []
    for label, h in [(str(h), h) for h in horizons] + [("Overall", preds.shape[1])]:
        assert h <= preds.shape[1], f"horizon {h} exceeds forecast window {preds.shape[1]}"
        p, t = preds[:, :h].ravel(), trues[:, :h].ravel()
        rows.append(dict(scope=scope, series=series, method=method, horizon=label,
                         mae=round(mae(p, t), 4), rae=round(rae(p, t), 4)))
    return rows


# --- Runs ---
def resolve_targets(attacks, cfg):
    if cfg.columns:
        names = [c.strip() for c in cfg.columns.split(",") if c.strip()]
        missing = [c for c in names if c not in attacks.columns]
        if missing:
            raise SystemExit(f"unknown columns: {missing}")
        return names
    if cfg.targets == "all":
        return list(attacks.columns)
    return [c for c in attacks.columns if c.endswith("-ALL")]


def sanitize(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def run_series_batch(attacks, targets, exog, train_start, train_end, fc_index, cfg):
    """Fit sarimax-exog + sarima for every target; returns {name: {method: SeriesResult}}."""
    exog_train = slice_train(exog, train_start, train_end).to_numpy(dtype=float)
    exog_future = exog.loc[fc_index[0]:fc_index[-1]].to_numpy(dtype=float)
    assert exog_future.shape[0] == len(fc_index)
    orders = parse_orders_override(cfg.orders) if cfg.orders else None

    def worker(col):
        y_train = slice_train(attacks[col], train_start, train_end)
        r_exog = fit_one_series(col, y_train, exog_train, exog_future, fc_index,
                                use_log1p=not cfg.no_log1p, orders=orders,
                                sparse_threshold=cfg.sparse_threshold)
        r_noexog = fit_one_series(col, y_train, None, None, fc_index,
                                  use_log1p=not cfg.no_log1p, orders=orders,
                                  sparse_threshold=cfg.sparse_threshold)
        return col, {"sarimax-exog": r_exog, "sarima": r_noexog}

    if cfg.n_jobs and cfg.n_jobs != 1:
        pairs = Parallel(n_jobs=cfg.n_jobs, verbose=5)(delayed(worker)(c) for c in targets)
    else:
        pairs = []
        for i, c in enumerate(targets, 1):
            pairs.append(worker(c))
            if i % 25 == 0 or i == len(targets):
                print(f"  fitted {i}/{len(targets)} series", flush=True)
    return dict(pairs)


def make_plots(attacks, targets, results, naive_preds, fc_index, truth, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    hist_start = fc_index[0] - 48
    for col in targets:
        fig, ax = plt.subplots(figsize=(10, 4))
        hist = attacks[col].loc[hist_start:fc_index[0] - 1]
        ax.plot(hist.index.to_timestamp(), hist.values, color="black", lw=1.2, label="history")
        if truth is not None:
            ax.plot(fc_index.to_timestamp(), truth[col].values, color="black", lw=1.2,
                    ls="--", label="truth")
        for method, color in [("sarimax-exog", "tab:blue"), ("sarima", "tab:orange")]:
            ax.plot(fc_index.to_timestamp(), results[col][method].forecast.values,
                    color=color, lw=1.4, label=method)
        if naive_preds is not None:
            ax.plot(fc_index.to_timestamp(), naive_preds[col], color="tab:green",
                    lw=1.0, label="seasonal-naive")
        ax.set_title(col)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{sanitize(col)}.png", dpi=110)
        plt.close(fig)
    print(f"wrote plots to {plot_dir}")


def run_holdout(cfg):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attacks, reddit = load_attacks(), load_reddit()
    targets = resolve_targets(attacks, cfg)

    train_start = pd.Period(cfg.train_start, freq="M") if cfg.train_start else pd.Period("2012-01", freq="M")
    train_end = pd.Period(cfg.holdout_train_end, freq="M")
    fc_index = pd.period_range(train_end + 1, LAST_VALID, freq="M")
    print(f"[holdout] train {train_start}..{train_end}, test {fc_index[0]}..{fc_index[-1]} "
          f"({len(fc_index)} months), {len(targets)} series")

    exog, exog_info = build_exog(reddit, train_start, train_end, LAST_VALID,
                                 n_components=cfg.pca_components,
                                 variance_target=cfg.pca_variance,
                                 standardize=not cfg.no_standardize,
                                 output_dir=out_dir, pkl_name="pca_model_holdout.pkl")
    print(f"[holdout] exog: {exog_info['n_components']} PCs, "
          f"cum. explained variance {exog_info['cumulative_explained_variance']}")

    results = run_series_batch(attacks, targets, exog, train_start, train_end, fc_index, cfg)
    naive_preds = {col: np.clip(seasonal_naive(slice_train(attacks[col], train_start, train_end),
                                               fc_index), 0.0, None)
                   for col in targets}
    truth = attacks.loc[fc_index, targets]

    # Long-format predictions
    pred_rows = []
    for col in targets:
        for method in ("sarimax-exog", "sarima"):
            fc = results[col][method].forecast
            for p in fc_index:
                pred_rows.append(dict(period=str(p), series=col, method=method,
                                      prediction=float(fc.loc[p]), truth=float(truth.loc[p, col])))
        for p, v in zip(fc_index, naive_preds[col]):
            pred_rows.append(dict(period=str(p), series=col, method="seasonal-naive",
                                  prediction=float(v), truth=float(truth.loc[p, col])))
    pd.DataFrame(pred_rows).to_csv(out_dir / "holdout_predictions.csv", index=False)

    # Metrics: per-series + aggregates (full target set and -ALL).
    # Horizons longer than the test window are dropped.
    horizons = [h for h in (3, 6, 9) if h <= len(fc_index)]
    truth_mat = truth.T.to_numpy(dtype=float)  # (n_series, n_steps)
    preds_by_method = {
        "sarimax-exog": np.vstack([results[c]["sarimax-exog"].forecast.to_numpy() for c in targets]),
        "sarima": np.vstack([results[c]["sarima"].forecast.to_numpy() for c in targets]),
        "seasonal-naive": np.vstack([naive_preds[c] for c in targets]),
    }
    rows = []
    agg_slices = {"all-targets": np.arange(len(targets))}
    all_idx = [i for i, c in enumerate(targets) if c.endswith("-ALL")]
    if all_idx:
        agg_slices["ALL-totals"] = np.array(all_idx)
    for method, preds in preds_by_method.items():
        for slice_name, idx in agg_slices.items():
            rows += metrics_rows("aggregate", slice_name, method, preds[idx], truth_mat[idx], horizons)
        for i, col in enumerate(targets):
            rows += metrics_rows("per-series", col, method, preds[i:i + 1], truth_mat[i:i + 1], horizons)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "holdout_metrics.csv", index=False)

    summary = metrics[(metrics.scope == "aggregate")].pivot_table(
        index=["series", "method"], columns="horizon", values="mae")
    print("\n[holdout] aggregate MAE by method and horizon:")
    print(summary.to_string())

    print(f"[holdout] wrote {out_dir}/holdout_metrics.csv, holdout_predictions.csv")

    if cfg.plots:
        make_plots(attacks, targets, results, naive_preds, fc_index, truth, out_dir)
    return metrics


def run_fill(cfg):
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attacks, reddit = load_attacks(), load_reddit()
    targets = resolve_targets(attacks, cfg)

    train_start = pd.Period(cfg.train_start, freq="M") if cfg.train_start else FIRST_MONTH
    train_end = LAST_VALID
    fc_index = GAP
    print(f"[fill] train {train_start}..{train_end}, fill {fc_index[0]}..{fc_index[-1]} "
          f"({len(fc_index)} months), {len(targets)} series")

    exog, exog_info = build_exog(reddit, train_start, train_end, GAP[-1],
                                 n_components=cfg.pca_components,
                                 variance_target=cfg.pca_variance,
                                 standardize=not cfg.no_standardize,
                                 output_dir=out_dir, pkl_name="pca_model_fill.pkl")
    print(f"[fill] exog: {exog_info['n_components']} PCs, "
          f"cum. explained variance {exog_info['cumulative_explained_variance']}")

    results = run_series_batch(attacks, targets, exog, train_start, train_end, fc_index, cfg)

    # Ensure partial runs don't override full runs.
    partial = len(targets) != N_EXPECTED_TARGETS
    filled_csv = SCRIPT_DIR / "NoI_monthly_v2_filled_sarimax_PARTIAL.csv" if partial else FILLED_CSV
    if partial:
        print(f"[fill] WARNING: only {len(targets)}/{N_EXPECTED_TARGETS} columns targeted; "
              f"writing PARTIAL fill to {filled_csv.name} -- untargeted columns keep "
              f"their gap-zero placeholders")

    # Rewrite the CSV: header + all rows through Jan-2025; only the 11 gap lines are regenerated.
    # Values are written as floats to match the original file's formatting.
    original_lines = ATTACKS_CSV.read_text().splitlines()
    assert len(original_lines) == N_EXPECTED_ROWS + 1
    filled = attacks.copy()
    for col in targets:
        filled.loc[fc_index, col] = np.rint(results[col]["sarimax-exog"].forecast.to_numpy())
    n_history = int((attacks.index <= LAST_VALID).sum())
    new_lines = original_lines[:1 + n_history]
    for p in fc_index:
        vals = ",".join(f"{float(v):.1f}" for v in filled.loc[p].to_numpy())
        new_lines.append(f"{p.strftime('%m/%Y')},{vals}")
    filled_csv.write_text("\n".join(new_lines) + "\n")

    # Post-fill assertions
    refill = pd.read_csv(filled_csv)
    assert len(refill) == N_EXPECTED_ROWS
    assert refill.columns.tolist() == ["Attack-Country"] + attacks.columns.tolist()
    check = refill.drop(columns=["Attack-Country"]).astype(float)
    check.index = attacks.index
    assert (check.loc[:LAST_VALID].to_numpy() == attacks.loc[:LAST_VALID].to_numpy()).all(), \
        "history rows changed"
    assert (check.to_numpy() >= 0).all()
    assert (check.to_numpy() == np.rint(check.to_numpy())).all(), "non-integer fills"
    if not partial:
        assert not (check.loc[GAP] == 0).all(axis=1).any(), "an all-zero gap row exists after fill"
    print(f"[fill] wrote {filled_csv}")

    if cfg.plots:
        make_plots(attacks, targets, results, None, fc_index, None, out_dir)

    if cfg.extend_dataset:
        if len(targets) != N_EXPECTED_TARGETS:
            raise SystemExit("--extend-dataset requires filling all 988 columns "
                             "(drop --columns / use --targets all)")
        extend_assembled_dataset(check)
    return results


def extend_assembled_dataset(filled_attacks):
    """Cyber_Trend_Forecasting_All_v2_2_sarimax.csv = v2.1 base with the 988 attack
    columns replaced by Hackmageddon values (real through Jan-25, imputed after)."""
    v21 = pd.read_csv(V2_1_CSV)
    assert len(v21) == N_EXPECTED_ROWS
    v21_idx = pd.PeriodIndex(pd.to_datetime(v21["Date"], format="%b-%y"), freq="M")
    assert (v21_idx == filled_attacks.index).all(), \
        "v2_1 rows are not date-aligned with the attack data"
    attack_cols = filled_attacks.columns.tolist()
    missing = [c for c in attack_cols if c not in v21.columns]
    assert not missing, f"attack columns missing from v2_1: {missing[:5]}"
    v22 = v21.copy()
    v22[attack_cols] = filled_attacks.to_numpy().astype(int)
    v22.to_csv(V2_2_CSV, index=False)

    # Verification
    reread = pd.read_csv(V2_2_CSV)
    assert reread.shape == v21.shape
    assert (reread[attack_cols].to_numpy() == filled_attacks.to_numpy().astype(int)).all()
    other_cols = [c for c in v21.columns if c not in attack_cols and c != "Date"]
    assert np.allclose(reread[other_cols].to_numpy(dtype=float),
                       v21[other_cols].to_numpy(dtype=float))
    assert (reread["Date"] == v21["Date"]).all()
    pd.to_datetime(reread["Date"], format="%b-%y")  # loader-compatible dates
    print(f"[extend] wrote {V2_2_CSV} ({reread.shape[0]} rows x {reread.shape[1]} cols)")


# -- CLI ---
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["holdout", "fill", "both"])
    p.add_argument("--targets", choices=["all", "all-agg"], default="all",
                   help="all = 988 columns; all-agg = the 26 '-ALL' totals")
    p.add_argument("--columns", default=None,
                   help="comma-separated column names (overrides --targets)")
    p.add_argument("--train-start", default=None,
                   help="YYYY-MM; default 2012-01 for holdout, 2011-07 for fill")
    p.add_argument("--holdout-train-end", default="2024-04")
    p.add_argument("--pca-components", type=int, default=5)
    p.add_argument("--pca-variance", type=float, default=None,
                   help="variance-target mode (e.g. 0.95); overrides --pca-components")
    p.add_argument("--no-standardize", action="store_true",
                   help="skip StandardScaler before PCA")
    p.add_argument("--no-log1p", action="store_true")
    p.add_argument("--sparse-threshold", type=float, default=24.0)
    p.add_argument("--orders", default=None,
                   help="override candidates: 'p,d,q|P,D,Q,s[|trend] ; ...'")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "gapfill_outputs"))
    p.add_argument("--extend-dataset", action="store_true",
                   help="after fill, write Cyber_Trend_Forecasting_All_v2_2_sarimax.csv")
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.mode in ("holdout", "both"):
        run_holdout(cfg)
    if cfg.mode in ("fill", "both"):
        run_fill(cfg)


if __name__ == "__main__":
    main()
