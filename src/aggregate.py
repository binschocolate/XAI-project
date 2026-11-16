"""모든 run의 metrics.json을 모아 표로 정리.

사용: python src/aggregate.py
결과: <results>/sri_runs.csv (run별), <results>/sri_summary.csv (arch x rho 평균/표준편차)
기본 <results> = /data/$USER/results/XAI-project
"""
import argparse
import json
import os
from pathlib import Path

DEFAULT_RESULTS = f"/data/{os.environ.get('USER', 'user')}/results/XAI-project"

import pandas as pd

KEYS = ["avg_acc", "worst_group_acc", "group_gap", "sri"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default=f"{DEFAULT_RESULTS}/models")
    ap.add_argument("--out_dir", default=DEFAULT_RESULTS)
    args = ap.parse_args()

    rows = [json.loads(p.read_text()) for p in sorted(Path(args.models_dir).glob("*/metrics.json"))]
    if not rows:
        raise SystemExit(f"no metrics.json under {args.models_dir}")
    df = pd.DataFrame(rows).sort_values(["arch", "rho", "seed"])
    df.to_csv(Path(args.out_dir) / "sri_runs.csv", index=False)

    summ = df.groupby(["arch", "rho"])[KEYS].agg(["mean", "std", "count"])
    summ.to_csv(Path(args.out_dir) / "sri_summary.csv")

    pd.set_option("display.width", 200)
    print(df[["run_name"] + KEYS].round(4).to_string(index=False))
    print()
    # PROGRESS.md에 붙여넣기 좋은 형태: mean ± std
    print("| arch | rho | n | avg acc | worst-group acc | group gap | SRI |")
    print("|---|---|---|---|---|---|---|")
    for (arch, rho), g in df.groupby(["arch", "rho"]):
        cells = [f"{g[k].mean():.3f} ± {g[k].std(ddof=1) if len(g) > 1 else 0:.3f}" for k in KEYS]
        print(f"| {arch} | {rho} | {len(g)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
