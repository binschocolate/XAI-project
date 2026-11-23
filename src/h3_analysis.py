"""H3-1 · H3-2 분석·판정 (CPU). 설계와 판정 기준: docs/H3_DESIGN.md

입력: outputs/h3/pairs/<arch>_seed<k>/raw.npz (src/h3_experiment.py)
      outputs/h3/oracle_faith/<clean run>/{auc.csv, summary.csv} (src/faithfulness.py + oracle map)
출력: outputs/h3/analysis/{expA.csv, expB.csv, h32_selection.csv, h32.csv, summary.md, figs/*.png}
사용: python src/h3_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parent.parent
H3 = REPO / "outputs" / "h3"
OUT = H3 / "analysis"
BASES = ["black", "mean", "blur", "inpaint", "trainset"]
N_BOOT, SEED = 1000, 0
TAUS = [0.02, 0.05]


def auroc(orig, new):
    ok = np.isfinite(new)
    return roc_auc_score(np.r_[np.zeros(len(orig)), np.ones(ok.sum())], np.r_[orig, new[ok]]) if ok.any() else np.nan


def boot_mean(v, g, rng, strat=True):
    """그룹 층화(가능하면) bootstrap 평균의 95% 구간 (NaN 제외)"""
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if len(v) == 0:
        return np.nan, np.nan
    if strat and all((g == k).sum() > 1 for k in range(4)):
        parts = [v[g == k] for k in range(4)]
        bt = np.mean([[p[rng.integers(0, len(p), len(p))].mean() for p in parts] for _ in range(N_BOOT)], 1)
    else:
        bt = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)])
    return np.percentile(bt, 2.5), np.percentile(bt, 97.5)


def gmean(v, g):
    """그룹 동일 가중 평균 (NaN 제외, 빈 그룹 제외)"""
    return np.nanmean([np.nanmean(v[g == k]) for k in range(4) if np.isfinite(v[g == k]).any()])


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pairs = sorted(p.parent.name for p in (H3 / "pairs").glob("*/raw.npz"))
    A_rows, S_rows, H_rows = [], [], []
    for pair in pairs:
        arch, seed = pair.split("_seed")
        z = np.load(H3 / "pairs" / pair / "raw.npz", allow_pickle=True)
        g, n = z["group"], len(z["group"])
        ref0 = z["ref_maha_orig"]

        # ---------- 실험 A (clean)
        p0 = z["p0_clean"]
        dp = p0[:, None] - z["A_p"]
        dp_same, dp_opp = dp[:, 5:9].mean(1), dp[:, 9:13].mean(1)
        ood_same = np.mean([auroc(ref0, z["A_ref_maha"][:, 5 + j]) for j in range(4)])
        ood_opp = np.mean([auroc(ref0, z["A_ref_maha"][:, 9 + j]) for j in range(4)])
        for bi, b in enumerate(BASES):
            art = dp[:, bi] - dp_same
            lo, hi = boot_mean(art, g, rng)
            A_rows.append(dict(arch=arch, seed=int(seed), baseline=b, ood_auroc=auroc(ref0, z["A_ref_maha"][:, bi]),
                               dp_mask=gmean(dp[:, bi], g), dp_same=gmean(dp_same, g), dp_opp=gmean(dp_opp, g),
                               artifact=gmean(art, g), artifact_lo=lo, artifact_hi=hi,
                               flip_mask=gmean((z["A_p"][:, bi] < 0.5).astype(float), g),
                               flip_same=gmean((z["A_p"][:, 5:9] < 0.5).mean(1), g),
                               flip_opp=gmean((z["A_p"][:, 9:13] < 0.5).mean(1), g),
                               ood_same=ood_same, ood_opp=ood_opp))

        # ---------- H3-2
        area = z["bird_area"]
        for tau in TAUS:
            cm = z[f"S2_common_tau{tau}"]
            cnt = cm.sum(1)
            S_rows.append(dict(arch=arch, seed=int(seed), tau=tau, n_images=n, share_ge2=(cnt >= 2).mean(),
                               mean_cells=cnt.mean(), share_zero=(cnt == 0).mean(),
                               unused_clean=(z["S1_maxdp_clean"] < tau).mean(),
                               unused_shortcut=(z["S1_maxdp_shortcut"] < tau).mean(),
                               rho_cells_birdarea=spearmanr(cnt, area)[0],
                               **{f"loc{c:02d}": cm[:, c].mean() for c in range(16)}))
            valid = cnt > 0
            refS = z[f"S2_ref_maha_tau{tau}"]
            dps = {k: z[f"p0_{k}"][:, None] - z[f"S2_p_{k}_tau{tau}"] for k in ["clean", "shortcut"]}
            for bi, b in enumerate(BASES):
                diff = dps["shortcut"][:, bi] - dps["clean"][:, bi]
                lo, hi = boot_mean(np.where(valid, diff, np.nan), g, rng, strat=False)
                art = {k: dps[k][:, bi] - np.nanmean(dps[k][:, 5:9], 1) for k in dps}
                H_rows.append(dict(arch=arch, seed=int(seed), tau=tau, baseline=b, n_valid=int(valid.sum()),
                                   ood_auroc=auroc(ref0[valid], refS[valid, bi]),
                                   dp_clean=np.nanmean(dps["clean"][valid, bi]), dp_shortcut=np.nanmean(dps["shortcut"][valid, bi]),
                                   diff=np.nanmean(diff[valid]), diff_lo=lo, diff_hi=hi,
                                   artifact_clean=np.nanmean(art["clean"][valid]), artifact_shortcut=np.nanmean(art["shortcut"][valid]),
                                   dp_nat_clean=np.nanmean(dps["clean"][valid, 5:9]),
                                   dp_nat_shortcut=np.nanmean(dps["shortcut"][valid, 5:9]),
                                   cos_clean=np.nanmean(z[f"S2_cos_clean_tau{tau}"][valid, bi]),
                                   cos_shortcut=np.nanmean(z[f"S2_cos_shortcut_tau{tau}"][valid, bi])))
    A, S, Hh = pd.DataFrame(A_rows), pd.DataFrame(S_rows), pd.DataFrame(H_rows)
    A.to_csv(OUT / "expA.csv", index=False)
    S.to_csv(OUT / "h32_selection.csv", index=False)
    Hh.to_csv(OUT / "h32.csv", index=False)

    # ---------- 실험 B (oracle 설명 평가)
    B_rows = []
    for d in sorted((H3 / "oracle_faith").glob("*/auc.csv")):
        run = d.parent.name
        arch, _, seed = run.split("_")
        a = pd.read_csv(d)
        for b in BASES:
            w = a[a.baseline == b].pivot(index="img_filename", columns="method", values="mif_auc")
            gg = a[(a.baseline == b) & (a.method == "random")].set_index("img_filename").loc[w.index].group.values
            unfair = (w["random"] - w["background"]).values
            lo, hi = boot_mean(unfair, gg, rng)
            oa = A[(A.arch == arch) & (A.seed == int(seed[-1])) & (A.baseline == b)]
            B_rows.append(dict(arch=arch, seed=int(seed[-1]), baseline=b, mif_bird=gmean(w["bird"].values, gg),
                               mif_background=gmean(w["background"].values, gg), mif_random=gmean(w["random"].values, gg),
                               unfair=gmean(unfair, gg), unfair_lo=lo, unfair_hi=hi,
                               bird_beats_background=bool(gmean(w["bird"].values, gg) < gmean(w["background"].values, gg)),
                               ood_auroc=oa.ood_auroc.iloc[0] if len(oa) else np.nan))
    B = pd.DataFrame(B_rows)
    B.to_csv(OUT / "expB.csv", index=False)
    write_summary(A, B, S, Hh)
    make_figs(A, B, Hh)


def rho_by(df, x, y, keys):
    return df.groupby(keys).apply(lambda t: spearmanr(t[x], t[y])[0]).rename(f"rho_{x}_{y}").reset_index()


def write_summary(A, B, S, Hh):
    pd.set_option("display.width", 220)
    L = ["# H3 실험 결과 (설계: docs/H3_DESIGN.md)\n"]
    # H3-1
    premise = A.groupby(["arch", "seed"]).dp_opp.first()
    rA = rho_by(A, "ood_auroc", "artifact", ["arch", "seed"])
    L += ["## H3-1 실험 A — clean 모델 배경 개입 (seed 평균)\n",
          "```\n" + A.groupby(["arch", "baseline"])[["ood_auroc", "dp_mask", "dp_same", "dp_opp", "artifact",
                                                    "flip_mask", "flip_same", "flip_opp"]].mean().round(4).to_string() + "\n```\n",
          "artifact 95% 구간이 0 보다 큰 seed 수 (/3):\n",
          "```\n" + A.assign(pos=A.artifact_lo > 0).groupby(["arch", "baseline"]).pos.sum().unstack().to_string() + "\n```\n",
          "ρ(OOD AUROC, artifact) — baseline 5종, seed 별:\n", "```\n" + rA.round(3).to_string(index=False) + "\n```\n",
          f"전제 확인 (반대 장소 교체 Δp, 기준 0.05 이하): max {premise.max():.4f}\n"]
    if len(B):
        rB = rho_by(B, "ood_auroc", "unfair", ["arch", "seed"])
        L += ["## H3-1 실험 B — 정답을 아는 설명의 평가 (MIF, seed 평균)\n",
              "```\n" + B.groupby(["arch", "baseline"])[["ood_auroc", "mif_bird", "mif_background", "mif_random", "unfair"]]
              .mean().round(4).to_string() + "\n```\n",
              "unfair = MIF(random) − MIF(background) > 0 이면 틀린 설명이 Random 보다 좋게 평가됨. 95% 구간이 0 보다 큰 seed 수:\n",
              "```\n" + B.assign(pos=B.unfair_lo > 0).groupby(["arch", "baseline"]).pos.sum().unstack().to_string() + "\n```\n",
              "ρ(OOD AUROC, unfair):\n", "```\n" + rB.round(3).to_string(index=False) + "\n```\n"]
    # 판정 H3-1
    if premise.max() > 0.05:
        v1 = "전제 붕괴 (clean 이 배경 교체에 반응) → 해석 보류"
    elif A.groupby("baseline").artifact.mean().abs().max() < 0.01:
        v1 = "H3-1 기각 (모든 baseline 에서 인공물 ≈ 0)"
    else:
        top = A.groupby("baseline").artifact.mean().idxmax()
        rho_pos = (rA.iloc[:, -1] > 0).mean() > 0.5
        ood_b = A.groupby("baseline").ood_auroc.mean().nlargest(2).index
        unfair_pos = len(B) and (B[B.baseline.isin(ood_b)].groupby("baseline").unfair.mean() > 0).all()
        if rho_pos and unfair_pos and A.groupby("baseline").artifact.mean().max() > 0:
            v1 = f"H3-1 지지 (인공물 > 0, OOD 와 비례, OOD baseline({', '.join(ood_b)})에서 부당한 점수 > 0)"
        else:
            v1 = f"인공물은 있으나 OOD 순서와 일치하지 않음 (최대: {top}) → 원인은 OOD 가 아니라 채운 내용 쪽"
    L += [f"**H3-1 판정 (자동, 설계 기준):** {v1}\n"]
    # H3-2
    L += ["## H3-2 단계 1 — 공통으로 안 보는 칸\n",
          "```\n" + S[["arch", "seed", "tau", "share_ge2", "mean_cells", "share_zero", "unused_clean", "unused_shortcut",
                      "rho_cells_birdarea"]].round(3).to_string(index=False) + "\n```\n",
          "칸 위치 빈도 (τ=0.02, 4×4, 모든 쌍 평균):\n",
          "```\n" + pd.DataFrame(S[S.tau == 0.02][[f"loc{c:02d}" for c in range(16)]].mean().values.reshape(4, 4))
          .round(2).to_string() + "\n```\n"]
    go = S[S.tau == 0.02].share_ge2.mean() >= 0.5
    L += ["## H3-2 단계 2 — 공통 칸 마스킹 (diff = Δp_shortcut − Δp_clean, seed 평균)\n",
          "```\n" + Hh.groupby(["tau", "arch", "baseline"])[["n_valid", "ood_auroc", "dp_clean", "dp_shortcut", "diff",
                                                            "dp_nat_clean", "dp_nat_shortcut"]].mean().round(4).to_string() + "\n```\n",
          "diff 95% 구간이 0 보다 큰 seed 수 (τ=0.02):\n",
          "```\n" + Hh[Hh.tau == 0.02].assign(pos=lambda t: t.diff_lo > 0).groupby(["arch", "baseline"]).pos.sum().unstack()
          .to_string() + "\n```\n"]
    h = Hh[Hh.tau == 0.02]
    if not go:
        v2 = f"판정 보류 — 진행 기준 미달 (공통 칸 ≥2 인 이미지 비율 {S[S.tau == 0.02].share_ge2.mean():.2f} < 0.5)"
    elif h["diff"].abs().max() < 0.01:
        v2 = "H3-2 기각 (공통 칸에서 shortcut 과 clean 의 반응 차이 ≈ 0)"
    else:
        per_arch = h.groupby(["arch", "seed"])["diff"].mean().gt(0).groupby("arch").mean()
        rh = rho_by(h, "ood_auroc", "diff", ["arch", "seed"]).iloc[:, -1]
        if (per_arch > 0.5).all() and (rh > 0).mean() > 0.5:
            v2 = "H3-2 지지 (두 arch 에서 shortcut 이 더 크게 반응, OOD 큰 baseline 에서 더 큼)"
        elif h["diff"].mean() < 0:
            v2 = "반대 방향 (clean 이 더 크게 반응)"
        else:
            v2 = "부분적 (방향이 arch·seed 에 따라 다르거나 OOD 순서와 무관)"
    L += [f"**H3-2 판정 (자동, 설계 기준, τ=0.02):** {v2}\n"]
    (OUT / "summary.md").write_text("\n".join(L))
    print("\n".join(L))


def make_figs(A, B, Hh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BC = {"black": "#2a78d6", "mean": "#eb6834", "blur": "#1baf7a", "inpaint": "#eda100", "trainset": "#e87ba4"}
    BN = {"black": "Black", "mean": "Mean", "blur": "Blur", "inpaint": "Inpaint", "trainset": "Train-set"}
    AM = {"resnet18": "o", "resnet50": "s"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.edgecolor": "#e4e3df", "font.size": 9, "legend.frameon": False})
    panels = [("H3-1 A: artifact (clean, background)", A, "artifact"),
              ("H3-1 B: unfair credit to wrong (background) map", B, "unfair"),
              ("H3-2: Δp shortcut − Δp clean (common unused cells, τ=0.02)", Hh[Hh.tau == 0.02], "diff")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, d, y) in zip(axes, panels):
        if len(d):
            for _, r in d.iterrows():
                ax.scatter(r.ood_auroc, r[y], color=BC[r.baseline], marker=AM[r.arch], s=45, edgecolor="#fcfcfb", lw=1)
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("OOD-ness of intervened images (ImageNet R50 AUROC)")
    hs = [plt.Line2D([], [], color=BC[b], marker="o", ls="", label=BN[b]) for b in BASES] + \
         [plt.Line2D([], [], color="#52514e", marker=m, ls="", label=a.replace("resnet", "ResNet-")) for a, m in AM.items()]
    fig.suptitle("H3: effect vs OOD-ness of the masking baseline (dot = seed)", x=0.01, y=0.99, ha="left")
    fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=7)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "figs" / "fig_h3_overview.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
