from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
# Results subdir per dataset
_DATASET_SUBDIR = {"csis": "csis", "cyber_trend": "cyber_trend", "mark3": "mark3", "v2_1": "v2_1"}
HORIZONS = [3, 6, 12, 24]


def _canonical_dir(dataset: str) -> Path:
    return REPO_ROOT / "Transformer_Pipeline" / "Results" / _DATASET_SUBDIR[dataset]


def _horizon_corr(preds: np.ndarray, trues: np.ndarray, h: int) -> float:
    p = preds[:, :h, :].flatten()
    t = trues[:, :h, :].flatten()
    if p.std() > 0 and t.std() > 0:
        return float(np.corrcoef(p, t)[0, 1])
    return 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", type=Path, required=True,
                   help="Sweep directory containing <cell>/seed_<S>/ snapshots.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for the ensemble predictions + summary.")
    p.add_argument("--cells", nargs="+", default=["attn", "mlp"],
                   help="Cell names under sweep-dir (default: attn mlp).")
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789],
                   help="Seeds to include (default: 123 456 789).")
    p.add_argument("--no-promote-canonical", action="store_true",
                   help="Skip copying outputs over Results/<dataset>/visionpp_*.npy.")
    p.add_argument("--dataset", type=str, default="csis", choices=list(_DATASET_SUBDIR.keys()),
                   help="Which dataset's canonical Results dir to promote to (default: csis).")
    args = p.parse_args()

    stack: list[np.ndarray] = []
    gt_ref: np.ndarray | None = None
    for cell in args.cells:
        for s in args.seeds:
            seed_dir = args.sweep_dir / cell / f"seed_{s}"
            preds = np.load(seed_dir / "visionpp_predictions.npy")
            gt = np.load(seed_dir / "visionpp_ground_truth.npy")
            print(f"  loaded {cell}/seed_{s}: {preds.shape}")
            stack.append(preds)
            if gt_ref is None:
                gt_ref = gt
            else:
                assert np.array_equal(gt, gt_ref), f"Ground truth mismatch in {seed_dir}"
    assert gt_ref is not None, "No predictions loaded"
    ens = np.mean(np.stack(stack, axis=0), axis=0)
    print(f"Ensemble shape: {ens.shape}  ({len(stack)} predictions averaged)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "visionpp_predictions.npy", ens)
    np.save(args.out_dir / "visionpp_ground_truth.npy", gt_ref)

    summary_path = args.out_dir / "ensemble_summary.csv"
    horizons_corr = {f"{h}m_corr": _horizon_corr(ens, gt_ref, h) for h in HORIZONS}
    overall_corr = float(np.corrcoef(ens.flatten(), gt_ref.flatten())[0, 1])
    pf_corrs = [
        float(np.corrcoef(ens[:, :, i].flatten(), gt_ref[:, :, i].flatten())[0, 1])
        for i in range(ens.shape[2])
        if ens[:, :, i].std() > 0 and gt_ref[:, :, i].std() > 0
    ]
    summary = {
        "cells": ",".join(args.cells),
        "seeds": ",".join(str(s) for s in args.seeds),
        "n_predictions_averaged": len(stack),
        **{k: round(v, 4) for k, v in horizons_corr.items()},
        "overall_corr": round(overall_corr, 4),
        "per_feat_corr_mean": round(float(np.mean(pf_corrs)), 4),
    }
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("\nEnsemble metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if not args.no_promote_canonical:
        canonical_dir = _canonical_dir(args.dataset)
        canonical_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.out_dir / "visionpp_predictions.npy",
                     canonical_dir / "visionpp_predictions.npy")
        shutil.copy2(args.out_dir / "visionpp_ground_truth.npy",
                     canonical_dir / "visionpp_ground_truth.npy")
        print(f"\nPromoted ensemble to canonical: {canonical_dir}/visionpp_*.npy")


if __name__ == "__main__":
    main()
