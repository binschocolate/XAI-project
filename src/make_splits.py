"""B안 train split 생성.

공식 train+val pool에서 seed마다
  - clean    (rho=0.5) : 1,678장 = 650 / 650 / 189 / 189
  - shortcut (rho=0.95): 1,678장 = 1235 / 65 / 19 / 359
를 뽑는다. shortcut은 clean에 들어간 이미지를 최대한 재사용한다.
같은 seed가 subsample과 학습 초기화를 모두 결정한다.

추가로 공통 test(5,794장)와 XAI 평가용 subset(그룹당 200장)도 저장한다.

사용: python src/make_splits.py --data_dir $DATA_DIR --out_dir splits --seeds 0 1 2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import GROUPS, group_counts

COLS = ["img_filename", "y", "place", "split"]
EXPECTED_POOL = {"landbird_land": 3965, "landbird_water": 650,
                 "waterbird_land": 189, "waterbird_water": 1190}
EXPECTED_TEST = {"landbird_land": 2255, "landbird_water": 2255,
                 "waterbird_land": 642, "waterbird_water": 642}


def idx(df, y, p):
    return df[(df.y == y) & (df.place == p)].index.to_numpy()


def make_pair(pool: pd.DataFrame, seed: int, minority_frac: float = 0.05):
    rng = np.random.default_rng(seed)
    LL, LW, WL, WW = idx(pool, 0, 0), idx(pool, 0, 1), idx(pool, 1, 0), idx(pool, 1, 1)

    # clean (rho=0.5): 소수 그룹 전부 + 같은 수의 다수 그룹
    ll_c = rng.choice(LL, len(LW), replace=False)
    ww_c = rng.choice(WW, len(WL), replace=False)
    clean = np.concatenate([ll_c, LW, WL, ww_c])

    # shortcut (rho=0.95): 클래스별 개수는 clean과 동일
    n_land_cls, n_water_cls = 2 * len(LW), 2 * len(WL)          # 1300, 378
    n_lw = round(minority_frac * n_land_cls)                    # 65
    n_wl = round(minority_frac * n_water_cls)                   # 19
    n_ll, n_ww = n_land_cls - n_lw, n_water_cls - n_wl          # 1235, 359

    ll_s = np.concatenate([ll_c, rng.choice(np.setdiff1d(LL, ll_c), n_ll - len(ll_c), replace=False)])
    ww_s = np.concatenate([ww_c, rng.choice(np.setdiff1d(WW, ww_c), n_ww - len(ww_c), replace=False)])
    lw_s = rng.choice(LW, n_lw, replace=False)
    wl_s = rng.choice(WL, n_wl, replace=False)
    shortcut = np.concatenate([ll_s, lw_s, wl_s, ww_s])

    assert len(clean) == len(shortcut)
    return pool.loc[clean, COLS], pool.loc[shortcut, COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="splits")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--xai_per_group", type=int, default=200)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(args.data_dir) / "metadata.csv")
    assert len(df) == 11788, f"metadata rows {len(df)} != 11788"

    pool = df[df.split.isin([0, 1])]
    test = df[df.split == 2]
    assert group_counts(pool) == EXPECTED_POOL, group_counts(pool)
    assert group_counts(test) == EXPECTED_TEST, group_counts(test)

    for seed in args.seeds:
        clean, shortcut = make_pair(pool, seed)
        clean.to_csv(out / f"train_rho050_seed{seed}.csv", index=False)
        shortcut.to_csv(out / f"train_rho095_seed{seed}.csv", index=False)
        shared = len(set(clean.img_filename) & set(shortcut.img_filename))
        print(f"[seed {seed}] clean {group_counts(clean)}")
        print(f"[seed {seed}] shortcut {group_counts(shortcut)}  shared={shared}")

    test[COLS].to_csv(out / "test.csv", index=False)

    # XAI 평가용 고정 subset (모든 모델 공통, seed 0 고정)
    rng = np.random.default_rng(0)
    parts = [test.loc[rng.choice(idx(test, y, p), args.xai_per_group, replace=False), COLS]
             for (y, p) in GROUPS]
    pd.concat(parts).to_csv(out / f"test_xai{args.xai_per_group * 4}.csv", index=False)
    print(f"test: {group_counts(test)}")
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
