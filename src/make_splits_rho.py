"""추가 ρ 수준 (0.75, 0.99) 의 B안 train split 생성 — 팀 src/make_splits.py 의 make_pair 를 그대로 사용.

make_pair(pool, seed, minority_frac) 는 같은 seed 에서 clean(ρ=0.5) 이미지를 최대한 재사용하며
클래스별 개수(landbird 1,300 / waterbird 378, 총 1,678장)를 고정하고 소수 그룹 비율만 minority_frac(= 1 − ρ)으로 바꾼다.
  ρ=0.75 :  975 / 325 / 94 / 284   (landbird-land / landbird-water / waterbird-land / waterbird-water)
  ρ=0.99 : 1287 /  13 /  4 / 374

검증: 같은 함수로 ρ=0.95, 0.5 를 다시 만들어 저장소에 커밋된 splits/*.csv 와 완전히 같은지 확인한다
      (같으면 ariel 데이터의 metadata 순서까지 팀과 동일 → 새 split 도 팀 방식과 일관).
사용: python src/make_splits_rho.py --data_dir $DATA_DIR
출력: splits/train_rho075_seed{0,1,2}.csv, splits/train_rho099_seed{0,1,2}.csv (기존 파일은 건드리지 않음)
"""
import argparse
from pathlib import Path

import pandas as pd

from common import group_counts, rho_tag
from make_splits import make_pair

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.75, 0.99])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    df = pd.read_csv(Path(args.data_dir) / "metadata.csv")
    pool = df[df.split.isin([0, 1])]
    sd = REPO / "splits"

    # 1) 재현 검증
    for seed in args.seeds:
        clean, short = make_pair(pool, seed, 0.05)
        for rho, new in [(0.5, clean), (0.95, short)]:
            old = pd.read_csv(sd / f"train_rho{rho_tag(rho)}_seed{seed}.csv")
            same = (old.img_filename.values == new.img_filename.values).all() and len(old) == len(new)
            print(f"[check] rho={rho} seed={seed}: 커밋된 split 과 {'동일' if same else '다름!'}")
            assert same, "팀 split 재현 실패 — 중단"

    # 2) 새 ρ
    for rho in args.rhos:
        for seed in args.seeds:
            clean, new = make_pair(pool, seed, round(1 - rho, 4))
            out = sd / f"train_rho{rho_tag(rho)}_seed{seed}.csv"
            assert not out.exists() or pd.read_csv(out).img_filename.equals(new.img_filename.reset_index(drop=True)), out
            new.to_csv(out, index=False)
            shared = len(set(clean.img_filename) & set(new.img_filename))
            print(f"[rho={rho} seed={seed}] {len(new)}장 {group_counts(new)} clean 과 공유 {shared}장 -> {out.name}")


if __name__ == "__main__":
    main()
