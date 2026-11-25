"""ρ 수준 경향 분석 (dose-response, CPU). 설계: docs/RHO_LEVELS_DESIGN.md

질문: 배경 의존도(ρ → SRI)가 높을수록 baseline 에 따른 평가(순위·점수)가 더 민감하게 변하는가?
모델 24개 = 2 arch × ρ {0.5, 0.75, 0.95, 0.99} × seed 0–2 (같은 seed 는 같은 clean 이미지를 최대한 공유하는 split)

입력: outputs/faithfulness_m6/<run>/ (6개 기법), outputs/attributions/<run>/meta.csv, models/<run>/metrics_ariel.json (SRI)
지표 (run 별)
  S = 1 − 평균 Kendall τ-b (baseline 5종 10쌍) — 기법 3개(주) / 5개(보조), MIF(주)·LIF·SRG, scope all·aligned·conflicting
  점수 변동 = baseline 5종에 대한 평균 점수 SD (기법별, MIF·SRG)
경향 (사전 고정)
  주   : arch × seed 마다 SRI(4개 모델)와 S(기법 3개, MIF, all)의 Spearman ρ_s → 6개 중 양수 개수, 평균
         + arch 별 기울기 (S ~ SRI + seed 고정효과, OLS) 와 이미지 bootstrap 95% 구간 (그룹별 200장 복원추출 2,000회)
  보조 : 같은 방식으로 기법 5개 S, aligned·conflicting, SRG, 점수 변동 SD
출력: outputs/dose_response/{per_run.csv, trend.csv, summary.md, figs/*.png}
사용: python src/dose_response.py
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analysis_stage7_9 as A

REPO = Path(__file__).resolve().parent.parent
SETS = {"3": ["gradcam", "ig", "rise"], "5": ["gradcam", "ig", "rise", "smoothgrad", "lime"]}
SCOPES = ["all", "aligned", "conflicting"]
RHO = {"rho050": 0.5, "rho075": 0.75, "rho095": 0.95, "rho099": 0.99}


def slope_with_seed_fe(x, y, seed):
    """y ~ x + seed 더미 의 x 계수 (arch 안에서)"""
    X = np.column_stack([x] + [(seed == s).astype(float) for s in np.unique(seed)])
    return np.linalg.lstsq(X, y, rcond=None)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--froot", default=str(REPO / "outputs" / "faithfulness_m6"))
    ap.add_argument("--aroot", default=str(REPO / "outputs" / "attributions"))
    ap.add_argument("--out", default=str(REPO / "outputs" / "dose_response"))
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figs").mkdir(parents=True, exist_ok=True)

    A._selftest()
    first = next(Path(args.froot).glob("*/curves.npz"))
    A.METHODS = [str(m) for m in np.load(first)["methods"]]
    report = []
    runs, data = A.load_all(Path(args.froot), Path(args.aroot), report, n_expected=24)
    fails = [m for s, m in report if s == "FAIL"]
    (out / "validation_report.md").write_text("# 입력 검증 (24개 모델)\n\n" + "\n".join(f"- {s}: {m}" for s, m in report))
    print(f"[검증] checks={len(report)} fail={len(fails)}")
    if fails:
        raise SystemExit("검증 실패: " + "; ".join(fails[:5]))

    info = {r: dict(arch=r.split("_")[0], rho=RHO[r.split("_")[1]], seed=int(r[-1]), sri=data[r]["sri"]) for r in runs}
    rng = np.random.default_rng(0)
    g0 = data[runs[0]]["g"]
    boot = [p[rng.integers(0, len(p), (args.n_boot, len(p)))] for p in [np.where(g0 == k)[0] for k in range(4)]]

    rows, Sboot = [], {}
    for r in runs:
        d = data[r]
        for sname, xai in SETS.items():
            xi = [A.METHODS.index(m) for m in xai]
            for q in ["mif", "lif", "srg"]:
                Q = A.qarr(d, q)[:, xi]
                GM = A.group_mean(Q, d["g"])
                GMb = np.stack([Q[b].mean(1) for b in boot], 1)  # (R,4,m,5)
                for sc in SCOPES:
                    S = A.rank_sensitivity(A.scope_mean(GM, sc))[0]
                    Sboot[(r, sname, q, sc)] = A.rank_sensitivity(GMb[:, A.SCOPES[sc]].mean(1))[0]
                    rows.append(dict(run=r, **info[r], methods=sname, metric=q, scope=sc, S=S))
        for q in ["mif", "lif", "srg"]:
            sc_vals = A.scope_mean(A.group_mean(A.qarr(d, q), d["g"]), "all")  # (method, base)
            for mi, m in enumerate(A.METHODS):
                rows.append(dict(run=r, **info[r], methods=m, metric=q, scope="all", S=np.nan,
                                 score_sd=np.std(sc_vals[mi], ddof=0)))
    pr = pd.DataFrame(rows)
    pr.to_csv(out / "per_run.csv", index=False)

    # 경향
    trend = []
    S = pr[pr.S.notna()]
    for (arch, sname, q, sc), g in S.groupby(["arch", "methods", "metric", "scope"]):
        rhos = [spearmanr(gg.sri, gg.S)[0] for _, gg in g.groupby("seed")]
        g = g.sort_values(["seed", "rho"])
        slope = slope_with_seed_fe(g.sri.values, g.S.values, g.seed.values)
        bs = np.array([slope_with_seed_fe(g.sri.values, np.array([Sboot[(r, sname, q, sc)][b] for r in g.run]),
                                          g.seed.values) for b in range(args.n_boot)])
        trend.append(dict(kind="rank_S", arch=arch, methods=sname, metric=q, scope=sc,
                          spearman_mean=np.nanmean(rhos), spearman_pos=f"{int(np.sum(np.array(rhos) > 0))}/{len(rhos)}",
                          slope_per_SRI=slope, slope_lo=np.nanpercentile(bs, 2.5), slope_hi=np.nanpercentile(bs, 97.5),
                          **{f"S_rho{k}": g[g.rho == k].S.mean() for k in sorted(RHO.values())}))
    V = pr[pr.S.isna() & pr.methods.isin(["gradcam", "rise", "lime", "smoothgrad", "ig"])]
    for (arch, m, q), g in V.groupby(["arch", "methods", "metric"]):
        rhos = [spearmanr(gg.sri, gg.score_sd)[0] for _, gg in g.groupby("seed")]
        trend.append(dict(kind="score_sd", arch=arch, methods=m, metric=q, scope="all",
                          spearman_mean=np.nanmean(rhos), spearman_pos=f"{int(np.sum(np.array(rhos) > 0))}/{len(rhos)}",
                          slope_per_SRI=slope_with_seed_fe(g.sri.values, g.score_sd.values, g.seed.values),
                          **{f"S_rho{k}": g[g.rho == k].score_sd.mean() for k in sorted(RHO.values())}))
    T = pd.DataFrame(trend)
    T.to_csv(out / "trend.csv", index=False)

    pd.set_option("display.width", 230)
    sri = pr.drop_duplicates("run").pivot_table(index="arch", columns="rho", values="sri", aggfunc=["mean", "min", "max"])
    main_ = T[(T.kind == "rank_S") & (T.methods == "3") & (T.metric == "mif") & (T.scope == "all")]
    lines = ["# ρ 수준 경향 분석 (dose-response)\n", "## SRI (ρ 별, seed 평균·최소·최대)\n", "```\n" + sri.round(3).to_string() + "\n```\n",
             "## 주 분석: 기법 3개, MIF, 전체 800장 — S 와 SRI 의 관계\n", "```\n" + main_.round(3).to_string(index=False) + "\n```\n",
             "## 순위 민감도 S 경향 (전체)\n", "```\n" + T[T.kind == "rank_S"].round(3).to_string(index=False) + "\n```\n",
             "## 점수 변동 SD 경향\n", "```\n" + T[T.kind == "score_sd"].round(4).to_string(index=False) + "\n```\n"]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    make_fig(pr, out)


def make_fig(pr, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    AC = {"resnet18": "#2a78d6", "resnet50": "#eb6834"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.edgecolor": "#e4e3df", "font.size": 9, "legend.frameon": False})
    panels = [("3", "mif", "all"), ("3", "mif", "aligned"), ("5", "mif", "all"), ("5", "mif", "aligned"),
              ("3", "srg", "all"), ("5", "srg", "all")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for ax, (s, q, sc) in zip(axes.ravel(), panels):
        d = pr[(pr.methods == s) & (pr.metric == q) & (pr.scope == sc)]
        for (arch, seed), g in d.groupby(["arch", "seed"]):
            g = g.sort_values("sri")
            ax.plot(g.sri, g.S, color=AC[arch], lw=1, alpha=0.6, marker="o", ms=4,
                    label=arch.replace("resnet", "ResNet-") if seed == 0 else None)
        ax.set_title(f"{s} methods · {q.upper()} · {sc}")
        ax.set_ylabel("rank sensitivity S")
    for ax in axes[1]:
        ax.set_xlabel("SRI (background reliance; ρ = 0.5 → 0.99)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2)
    fig.suptitle("Dose-response: SRI vs rank sensitivity (line = one seed across ρ levels)", x=0.01, y=0.99, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "figs" / "fig_sri_vs_S.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
