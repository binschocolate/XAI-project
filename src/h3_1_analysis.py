"""H3-1 검증 (탐색 분석, CPU) — "OOD 를 만드는 마스킹으로 모델이 쓰지 않는 부분을 가리면 XAI 신뢰성과 무관한 결과가 나온다"

입력 (새 실행 없음, 기존 결과만 연결)
  outputs/faithfulness/<run>/curves.npz   단계 6 곡선 (800장 × method × baseline × 50점)
  outputs/attributions/<run>/meta.csv     그룹 정보
  outputs/baseline_ood/<run>/ood.csv      baseline 별 OOD AUROC (무작위 patch 마스크)

설계 (결과 보기 전에 고정)
  · "모델이 쓰지 않는 부분" = attribution 이 가장 덜 중요하다고 한 patch → LIF 곡선의 앞부분.
    주 정의: Grad-CAM·RISE 평균 (단계 6 SRG 가 0.21–0.42 로 신뢰할 만한 두 기법).
    IG 는 LIF ≈ Random 이라 "안 쓰는 부분"을 제대로 고르지 못해 제외. Grad-CAM·RISE 각각은 민감도 분석으로 보고.
  · 지운 비율 f: 25% (주, OOD AUROC 가 baseline 을 가장 잘 구분한 비율), 10%·50% (보조). 곡선은 선형 보간.
  · 지표 (이미지별 → scope 평균; scope = all / aligned / conflicting, 그룹 평균 동일 가중)
      U   = p_orig − p_LIF(f)                안 쓰는 부분을 지웠을 때 target 확률 하락 (마스킹 왜곡의 크기)
      U_rel = U / p_orig                     원래 확신도 차이 보정
      R   = p_orig − p_Random(f)             무작위로 지웠을 때 하락 (같은 양의 정보 제거 기준)
      Sep = p_LIF(f) − p_Random(f) = R − U    "안 쓰는 곳"과 "아무 곳"을 구분하는 정도 — XAI 평가가 의미를 갖는 폭
      OOD = Mahalanobis AUROC (원본 vs 마스킹, 같은 f, 같은 scope)
  · H3-1 예측: OOD 가 큰 baseline 일수록 U 가 크고 Sep 이 작다.
      검정 1) 모델(run)마다 baseline 5종에서 Spearman ρ(OOD, U), ρ(OOD, Sep) → 12개 모델의 부호 일관성
      검정 2) 사전 지정 대비: Black − Inpaint (가장 OOD vs 가장 덜 OOD), Black − Blur
               이미지 bootstrap(그룹별 200장 복원추출, 1,000회, seed 0) 95% 구간
      민감도: Train-set 제외 ρ (Train-set 은 OOD 가 아니라 다른 이미지의 내용을 넣는 baseline 이라 성격이 다름)
  · H3-2 연결: 위 값을 shortcut / clean, aligned / conflicting 으로 나눠 비교
한계: OOD AUROC 는 무작위 patch 마스크로 잰 값이고 LIF 마스크(덜 중요한 patch)와 위치가 다르다.
      결과를 본 뒤 추가한 분석이므로 탐색 분석으로 보고.
출력: outputs/h3_1/{metrics.csv, rho.csv, contrasts.csv, summary.md, figs/*.png}
사용: python src/h3_1_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "h3_1"
BASES = ["black", "mean", "blur", "inpaint", "trainset"]
SCOPES = {"all": [0, 1, 2, 3], "aligned": [0, 3], "conflicting": [1, 2]}
FRACS = [0.10, 0.25, 0.50]
METHOD_SETS = {"gradcam+rise": ["gradcam", "rise"], "gradcam": ["gradcam"], "rise": ["rise"]}
CONTRASTS = [("black", "inpaint"), ("black", "blur")]
N_BOOT, SEED = 1000, 0


def runs():
    return sorted(p.parent.name for p in (REPO / "outputs" / "faithfulness").glob("*/curves.npz"))


def load(run):
    z = np.load(REPO / "outputs" / "faithfulness" / run / "curves.npz")
    order = np.argsort(z["img_idx"])
    meth = list(z["methods"])
    assert list(z["baselines"]) == BASES
    am = pd.read_csv(REPO / "outputs" / "attributions" / run / "meta.csv")
    x = z["x"]
    interp = lambda c, f: np.apply_along_axis(lambda v: np.interp(f, x, v), -1, c)  # (...,P) -> (...)
    lif, mif = z["lif"][order], z["mif"][order]
    p0 = mif[:, meth.index("random"), :, 0]  # (n, B) 원본 확률 (모든 곡선 시작점 동일)
    out = {"g": am.group.values, "p0": p0}
    for f in FRACS:
        out[f] = {"rand": interp(mif[:, meth.index("random")], f),  # (n,B)
                  **{m: interp(lif[:, meth.index(m)], f) for m in ["gradcam", "rise"]}}
    return out


def per_image(d, f, mset):
    """이미지별 U, U_rel, R, Sep (n,B)"""
    p_lif = np.mean([d[f][m] for m in METHOD_SETS[mset]], 0)
    U = d["p0"] - p_lif
    R = d["p0"] - d[f]["rand"]
    return {"U": U, "U_rel": U / d["p0"], "R": R, "Sep": p_lif - d[f]["rand"], "p0": d["p0"]}


def scope_mean(a, g, scope):
    return np.mean([a[g == k].mean(0) for k in SCOPES[scope]], 0)


def main():
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    rs = runs()
    ood = pd.concat([pd.read_csv(REPO / "outputs" / "baseline_ood" / r / "ood.csv") for r in rs])
    ood = ood.set_index(["run", "baseline", "fraction", "scope"]).auroc_maha
    rng = np.random.default_rng(SEED)
    data = {r: load(r) for r in rs}
    g0 = data[rs[0]]["g"]
    assert all((data[r]["g"] == g0).all() for r in rs), "run 간 이미지 순서 불일치"
    pos = [np.where(g0 == k)[0] for k in range(4)]
    boot = [p[rng.integers(0, len(p), (N_BOOT, len(p)))] for p in pos]

    rows, crows = [], []
    for r in rs:
        arch, rho, seed = r.split("_")
        model = "shortcut" if rho == "rho095" else "clean"
        d = data[r]
        for f in FRACS:
            for mset in METHOD_SETS:
                pi = per_image(d, f, mset)
                for sc in SCOPES:
                    vals = {k: scope_mean(v, d["g"], sc) for k, v in pi.items()}
                    for bi, b in enumerate(BASES):
                        of = 0.5 if f == 0.5 else 0.25  # OOD 는 25/50/75% 에서만 측정 → 10% 는 25% 값 사용
                        rows.append(dict(run=r, arch=arch, model=model, seed=int(seed[-1]), fraction=f, methods=mset,
                                         scope=sc, baseline=b, ood_auroc=ood[(r, b, of, sc)],
                                         **{k: v[bi] for k, v in vals.items()}))
                    # 사전 지정 대비 + bootstrap (주 정의만)
                    if mset == "gradcam+rise":
                        for b1, b2 in CONTRASTS:
                            i1, i2 = BASES.index(b1), BASES.index(b2)
                            for met in ["U", "Sep"]:
                                diff = pi[met][:, i1] - pi[met][:, i2]
                                est = scope_mean(diff, d["g"], sc)
                                gm = np.stack([diff[b].mean(1) for b in boot], 1)  # (N_BOOT, 4 그룹)
                                bt = gm[:, SCOPES[sc]].mean(1)
                                crows.append(dict(run=r, arch=arch, model=model, seed=int(seed[-1]), fraction=f, scope=sc,
                                                  contrast=f"{b1}-{b2}", metric=met, estimate=est,
                                                  ci_lo=np.percentile(bt, 2.5), ci_hi=np.percentile(bt, 97.5)))
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "metrics.csv", index=False)
    c = pd.DataFrame(crows)
    c.to_csv(OUT / "contrasts.csv", index=False)

    # 모델별 Spearman ρ (baseline 5종 / Train-set 제외 4종)
    rr = []
    for key, g in m.groupby(["run", "arch", "model", "fraction", "methods", "scope"]):
        for excl in [False, True]:
            gg = g[g.baseline != "trainset"] if excl else g
            rr.append(dict(zip(["run", "arch", "model", "fraction", "methods", "scope"], key),
                           baselines="4 (no trainset)" if excl else "5",
                           rho_ood_U=spearmanr(gg.ood_auroc, gg.U)[0], rho_ood_Urel=spearmanr(gg.ood_auroc, gg.U_rel)[0],
                           rho_ood_Sep=spearmanr(gg.ood_auroc, gg.Sep)[0]))
    rho = pd.DataFrame(rr)
    rho.to_csv(OUT / "rho.csv", index=False)
    write_summary(m, c, rho)
    make_figs(m)


def write_summary(m, c, rho):
    pd.set_option("display.width", 220)
    main_ = (m.methods == "gradcam+rise") & (m.fraction == 0.25)
    t = m[main_].groupby(["arch", "model", "scope", "baseline"])[["ood_auroc", "U", "U_rel", "R", "Sep"]].mean()
    r = rho[(rho.methods == "gradcam+rise")].groupby(["fraction", "scope", "baselines", "arch", "model"]).agg(
        rho_U=("rho_ood_U", "mean"), U_pos=("rho_ood_U", lambda v: f"{(v > 0).sum()}/{len(v)}"),
        rho_Sep=("rho_ood_Sep", "mean"), Sep_neg=("rho_ood_Sep", lambda v: f"{(v < 0).sum()}/{len(v)}"))
    cc = c[c.fraction == 0.25].groupby(["contrast", "metric", "scope", "arch", "model"]).agg(
        est=("estimate", "mean"), excl0_pos=("ci_lo", lambda v: int((v > 0).sum())),
        excl0_neg=("ci_hi", lambda v: int((v < 0).sum())))
    sens = rho[(rho.fraction == 0.25) & (rho.scope == "all") & (rho.baselines == "5")].groupby(
        ["methods", "arch", "model"]).rho_ood_U.mean().unstack("methods")
    lines = ["# H3-1 검증 (탐색): OOD 정도 vs '안 쓰는 부분' 마스킹 왜곡\n",
             "주 정의: Grad-CAM·RISE 평균의 LIF, 25% 제거, OOD = Mahalanobis AUROC (seed 평균)\n",
             "## 1. baseline 별 값\n", "```\n" + t.round(3).to_string() + "\n```\n",
             "## 2. 모델별 Spearman ρ(OOD, ·) — 예측: ρ(OOD,U) > 0, ρ(OOD,Sep) < 0\n",
             "```\n" + r.round(3).to_string() + "\n```\n",
             "## 3. 사전 지정 대비 (25%, 이미지 bootstrap 95% 구간; excl0_* = 구간이 0 을 제외한 seed 수 /3)\n",
             "```\n" + cc.round(3).to_string() + "\n```\n",
             "## 4. 민감도: 기법 정의별 ρ(OOD,U) (25%, scope=all, baseline 5종)\n",
             "```\n" + sens.round(3).to_string() + "\n```\n"]
    (OUT / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def make_figs(m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BC = {"black": "#2a78d6", "mean": "#eb6834", "blur": "#1baf7a", "inpaint": "#eda100", "trainset": "#e87ba4"}
    BN = {"black": "Black", "mean": "Mean", "blur": "Blur", "inpaint": "Inpaint", "trainset": "Train-set"}
    MK = {("resnet18", "clean"): "o", ("resnet18", "shortcut"): "^", ("resnet50", "clean"): "s", ("resnet50", "shortcut"): "D"}
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                         "axes.edgecolor": "#e4e3df", "font.size": 9, "legend.frameon": False})
    d = m[(m.methods == "gradcam+rise") & (m.fraction == 0.25)]
    d = d.groupby(["arch", "model", "scope", "baseline"])[["ood_auroc", "U", "Sep"]].mean().reset_index()
    for met, lab in [("U", "U = p_orig − p(LIF), 25%"),
                     ("Sep", "Sep = p(LIF) − p(Random), 25%")]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, sc in zip(axes, SCOPES):
            for (arch, model), mk in MK.items():
                for b in BASES:
                    r = d[(d.arch == arch) & (d.model == model) & (d.scope == sc) & (d.baseline == b)]
                    ax.scatter(r.ood_auroc, r[met], color=BC[b], marker=mk, s=55, edgecolor="#fcfcfb", lw=1.2)
            ax.set_title(f"scope = {sc}")
            ax.set_xlabel("OOD-ness: Mahalanobis AUROC (25% removal)")
        axes[0].set_ylabel(lab)
        hs = [plt.Line2D([], [], color=BC[b], marker="o", ls="", label=BN[b]) for b in BASES] + \
             [plt.Line2D([], [], color="#52514e", marker=mk, ls="", label=f"{a.replace('resnet', 'R')} {mo}")
              for (a, mo), mk in MK.items()]
        fig.suptitle(f"H3-1: baseline OOD-ness vs {met} (Grad-CAM·RISE LIF, seed mean)", x=0.01, y=0.99, ha="left")
        fig.legend(handles=hs, loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=9)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        fig.savefig(OUT / "figs" / f"fig_h3_1_ood_vs_{met}.png", dpi=140)
        plt.close(fig)


if __name__ == "__main__":
    main()
