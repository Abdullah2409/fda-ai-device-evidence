"""Step 7: exploratory analysis, models and robustness check (analysis plan, steps 1-4).

  1. Exploratory analysis: authorizations by year and specialty, share of summaries
     reporting clinical validation over time, recall and adverse event rates by validation.
  2. Recall model: logistic regression, recalled ~ validated + controls.
  3. Adverse event model: negative binomial regression of report counts with offset
     log(years on market), after checking overdispersion against a Poisson model.
  4. Robustness: Cox model of time to first recall, plus Kaplan-Meier curves.

Controls (analysis plan): risk class, specialty, public company status, review time,
years on market. Risk class is entered as class III vs. other because 99% of devices
are class II. The model sample excludes PMA supplements (their recalls and reports
belong to the whole product line) and devices without usable summary text.

Outputs: report/figures/*.png, report/tables/*.csv
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

from common import PROCESSED, ROOT

FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"

# Reference palette (dataviz skill): categorical slots in fixed order, text inks, surface.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})

PERFORMANCE: list[tuple[str, str, float]] = []  # (model, metric, value) -> tables/model_performance.csv

CONTROLS = "public_company + C(specialty, Treatment('Radiology')) + class_iii + review_100d"
LABELS = {
    "validated": "Clinically validated",
    "public_company": "Public company",
    "C(specialty, Treatment('Radiology'))[T.Cardiovascular]": "Cardiovascular (vs. radiology)",
    "C(specialty, Treatment('Radiology'))[T.Neurology]": "Neurology (vs. radiology)",
    "C(specialty, Treatment('Radiology'))[T.Other]": "Other specialty (vs. radiology)",
    "class_iii": "Risk class III",
    "review_100d": "Review time (per 100 days)",
    "years_on_market": "Years on market",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(PROCESSED / "devices.csv")
    df["class_iii"] = (df["risk_class"].astype(str) == "3").astype(int)
    df["review_100d"] = df["review_days"] / 100
    model = df[df["in_model_sample"] == 1].copy()
    model["validated"] = model["validated"].astype(int)
    return df, model


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# ---------------------------------------------------------------- 1. exploratory

def fig_growth(df: pd.DataFrame) -> None:
    order = ["Radiology", "Cardiovascular", "Neurology", "Other"]
    counts = (df[df["decision_year"] >= 2010].pivot_table(index="decision_year", columns="specialty",
                                                         values="submission_number", aggfunc="count")
              .reindex(columns=order).fillna(0))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bottom = np.zeros(len(counts))
    for spec, color in zip(order, [BLUE, ORANGE, AQUA, YELLOW]):
        ax.bar(counts.index, counts[spec], bottom=bottom, color=color, width=0.8,
               edgecolor="white", linewidth=1, label=spec)
        bottom += counts[spec].values
    ax.set_title("FDA authorizations of AI-enabled devices per year, by specialty")
    ax.set_ylabel("Devices authorized")
    ax.legend(ncol=4, loc="upper left")
    ax.set_ylim(0, bottom.max() * 1.22)
    ax.set_xticks(counts.index[::2])
    ax.annotate("Jan–Jun\nonly", xy=(2026, bottom[-1]), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", color=INK2, fontsize=7)
    ax.grid(axis="x", visible=False)
    save(fig, "fig1_growth_by_specialty")
    counts.assign(total=counts.sum(axis=1)).to_csv(TAB / "authorizations_by_year.csv")


def fig_validation_share(df: pd.DataFrame) -> None:
    coded = df[df["validated"].notna()].copy()
    coded["period"] = coded["decision_year"].clip(lower=2015)  # 2015 = "2015 and earlier"
    g = coded.groupby("period")["validated"].agg(["mean", "size"])
    se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.fill_between(g.index, (g["mean"] - 1.96 * se) * 100, (g["mean"] + 1.96 * se) * 100,
                    color=BLUE, alpha=0.15, linewidth=0)
    ax.plot(g.index, g["mean"] * 100, color=BLUE, linewidth=2, marker="o", markersize=5)
    for x, n in g["size"].items():  # sample sizes along the bottom, clear of the line
        ax.text(x, 3, f"n={int(n)}", ha="center", fontsize=7, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of summaries reporting\nclinical validation")
    ax.set_title("Share of device summaries reporting testing on patient data")
    ax.set_xticks(g.index, ["≤2015" if x == 2015 else str(x) for x in g.index])
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_validation_share_by_year")
    g.rename(columns={"mean": "share_validated", "size": "n"}).to_csv(TAB / "validation_share_by_year.csv")


def fig_rates(model: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v, grp in model.groupby("validated"):
        for outcome, label in (("recalled", "Recalled"), ("any_ae", "Any adverse event report")):
            p, n = grp[outcome].mean(), len(grp)
            rows.append({"validated": v, "outcome": label, "rate": p, "n": n,
                         "lo": max(p - 1.96 * np.sqrt(p * (1 - p) / n), 0),
                         "hi": p + 1.96 * np.sqrt(p * (1 - p) / n)})
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(2)
    for k, (v, color, name) in enumerate(((0, ORANGE, "No reported clinical validation"),
                                          (1, BLUE, "Clinically validated"))):
        sub = r[r["validated"] == v]
        pos = x + (k - 0.5) * 0.36
        ax.bar(pos, sub["rate"] * 100, width=0.34, color=color, label=f"{name} (n={sub['n'].iloc[0]})")
        ax.errorbar(pos, sub["rate"] * 100, yerr=[(sub["rate"] - sub["lo"]) * 100, (sub["hi"] - sub["rate"]) * 100],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for p_, val, hi in zip(pos, sub["rate"], sub["hi"]):
            ax.annotate(f"{val:.1%}", (p_, hi * 100), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, ["Recalled after clearance", "Any adverse event report"])
    ax.set_ylabel("% of devices")
    ax.set_title("Post-market outcomes by reported clinical validation (unadjusted)")
    ax.set_ylim(0, 27)
    ax.legend(loc="upper center", fontsize=8, ncol=1)
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_outcomes_by_validation")
    return r


def table_descriptives(model: pd.DataFrame) -> None:
    def summarize(g):
        return pd.Series({
            "Devices": len(g),
            "Recalled, %": 100 * g["recalled"].mean(),
            "Any adverse event report, %": 100 * g["any_ae"].mean(),
            "Adverse event reports, total": g["ae_reports"].sum(),
            "Public company, %": 100 * g["public_company"].mean(),
            "Radiology, %": 100 * (g["specialty"] == "Radiology").mean(),
            "Risk class III, %": 100 * g["class_iii"].mean(),
            "Review time, median days": g["review_days"].median(),
            "Years on market, mean": g["years_on_market"].mean(),
        })
    t = pd.concat({"No reported validation": summarize(model[model["validated"] == 0]),
                   "Clinically validated": summarize(model[model["validated"] == 1]),
                   "All": summarize(model)}, axis=1).round(1)
    t.to_csv(TAB / "table1_descriptives.csv")
    print("\nTable 1\n", t.to_string())


# ---------------------------------------------------------------- 2-3. models

def ratio_table(res, exp_name: str, cluster_res=None) -> pd.DataFrame:
    ci = res.conf_int()
    t = pd.DataFrame({exp_name: np.exp(res.params), "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]),
                      "p": res.pvalues})
    if cluster_res is not None:
        cci = cluster_res.conf_int()
        t["ci_low_cluster"], t["ci_high_cluster"], t["p_cluster"] = np.exp(cci[0]), np.exp(cci[1]), cluster_res.pvalues
    t = t.drop(index=[i for i in t.index if i in ("Intercept", "alpha")])
    t.index = [LABELS.get(i, i) for i in t.index]
    return t


def recall_model(model: pd.DataFrame) -> pd.DataFrame:
    formula = f"recalled ~ validated + {CONTROLS} + years_on_market"
    res = smf.logit(formula, data=model).fit(disp=0)
    # Robustness: one recall event can cover several devices of the same firm.
    clus = smf.logit(formula, data=model).fit(disp=0, cov_type="cluster",
                                              cov_kwds={"groups": model["company"].astype("category").cat.codes})
    base = smf.logit(f"recalled ~ {CONTROLS} + years_on_market", data=model).fit(disp=0)
    lr = 2 * (res.llf - base.llf)
    lr_p = stats.chi2.sf(lr, 1)
    auc = roc_auc_score(model["recalled"], res.predict(model))
    auc_base = roc_auc_score(model["recalled"], base.predict(model))  # same model without validation
    brier = brier_score_loss(model["recalled"], res.predict(model))
    PERFORMANCE.extend([
        ("Logistic (recall)", "AUC (ROC)", auc),
        ("Logistic (recall)", "AUC without validation variable", auc_base),
        ("Logistic (recall)", "McFadden pseudo-R2", res.prsquared),
        ("Logistic (recall)", "Brier score", brier),
        ("Logistic (recall)", "LR test chi2 for validation (1 df)", lr),
        ("Logistic (recall)", "LR test p-value", lr_p),
    ])
    t = ratio_table(res, "odds_ratio", clus)
    t.to_csv(TAB / "table2_logistic_recall.csv")
    print(f"\nLogistic regression, recall (n={int(res.nobs)}, events={model['recalled'].sum()})")
    print(t.round(3).to_string())
    print(f"LR test for validated: chi2={lr:.2f}, p={lr_p:.4g}; pseudo-R2={res.prsquared:.3f}; "
          f"AUC={auc:.3f} (without validation {auc_base:.3f}); Brier={brier:.4f}")
    forest(t, "odds_ratio", "Odds ratio of recall (95% CI, log scale)", "fig4_recall_odds_ratios")
    return t


def ae_model(model: pd.DataFrame) -> pd.DataFrame:
    formula = f"ae_reports ~ validated + {CONTROLS}"
    offset = np.log(model["years_on_market"])
    pois = smf.glm(formula, data=model, family=sm.families.Poisson(), offset=offset).fit()
    dispersion = pois.pearson_chi2 / pois.df_resid
    nb = smf.negativebinomial(formula, data=model, offset=offset).fit(disp=0, maxiter=200)
    alpha = nb.params["alpha"]
    t = ratio_table(nb, "incidence_rate_ratio")
    t.to_csv(TAB / "table3_negbin_adverse_events.csv")
    zeros = (model["ae_reports"] == 0).mean()
    print(f"\nNegative binomial, adverse event reports per year on market (n={int(nb.nobs)}, "
          f"{zeros:.1%} zeros)")
    print(f"Poisson dispersion (Pearson chi2/df) = {dispersion:.1f}  -> overdispersed; NB alpha = {alpha:.2f}; "
          f"AIC Poisson {pois.aic:.0f} vs NB {nb.aic:.0f}")
    print(t.round(3).to_string())
    PERFORMANCE.extend([
        ("Negative binomial (adverse events)", "AIC", nb.aic),
        ("Negative binomial (adverse events)", "AIC of Poisson alternative", pois.aic),
        ("Negative binomial (adverse events)", "McFadden pseudo-R2", nb.prsquared),
    ])
    pd.DataFrame({"poisson_dispersion": [dispersion], "nb_alpha": [alpha], "aic_poisson": [pois.aic],
                  "aic_nb": [nb.aic], "share_zero": [zeros]}).to_csv(TAB / "table3b_ae_model_fit.csv", index=False)
    return t


def forest(t: pd.DataFrame, col: str, xlabel: str, name: str) -> None:
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(t))
    colors = [BLUE if lbl == "Clinically validated" else INK2 for lbl in t.index]
    ax.hlines(y, t["ci_low"], t["ci_high"], color=colors, linewidth=2)
    ax.scatter(t[col], y, color=colors, s=36, zorder=3)
    ax.axvline(1, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yticks(y, t.index)
    ax.set_xlabel(xlabel)
    ax.set_title("Adjusted associations with recall (logistic regression)")
    ax.grid(axis="y", visible=False)
    save(fig, name)


# ---------------------------------------------------------------- 4. robustness

def cox_model(model: pd.DataFrame) -> pd.DataFrame:
    d = model.copy()
    d["years_observed"] = (d["days_observed"].clip(lower=1)) / 365.25
    spec = pd.get_dummies(d["specialty"], prefix="spec", dtype=int).drop(columns="spec_Radiology")
    X = pd.concat([d[["years_observed", "recalled", "validated", "public_company", "class_iii", "review_100d"]],
                   spec], axis=1)
    cph = CoxPHFitter().fit(X, duration_col="years_observed", event_col="recalled")
    t = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    t.columns = ["hazard_ratio", "ci_low", "ci_high", "p"]
    ph = proportional_hazard_test(cph, X, time_transform="rank").summary
    t["ph_test_p"] = ph["p"]
    t.to_csv(TAB / "table4_cox_time_to_recall.csv")
    PERFORMANCE.extend([
        ("Cox (time to recall)", "Concordance (Harrell's C)", cph.concordance_index_),
        ("Cox (time to recall)", "Proportional-hazards test p (validation)", ph.loc["validated", "p"]),
    ])
    print(f"\nCox model, time to first recall (n={len(X)}, events={int(X['recalled'].sum())}, "
          f"concordance={cph.concordance_index_:.3f})")
    print(t.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for v, color, name in ((0, ORANGE, "No reported clinical validation"), (1, BLUE, "Clinically validated")):
        g = d[d["validated"] == v]
        km = KaplanMeierFitter().fit(g["years_observed"], g["recalled"], label=name)
        cum = 1 - km.survival_function_[name]
        ci = 1 - km.confidence_interval_
        ax.step(cum.index, cum * 100, where="post", color=color, linewidth=2, label=f"{name} (n={len(g)})")
        ax.fill_between(ci.index, ci.iloc[:, 1] * 100, ci.iloc[:, 0] * 100, step="post", color=color,
                        alpha=0.12, linewidth=0)
    lr = logrank_test(d.loc[d.validated == 0, "years_observed"], d.loc[d.validated == 1, "years_observed"],
                      d.loc[d.validated == 0, "recalled"], d.loc[d.validated == 1, "recalled"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 35)
    ax.set_xlabel("Years since authorization")
    ax.set_ylabel("Cumulative % recalled")
    ax.set_title("Time to first recall (Kaplan–Meier)")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.05, f"log-rank p = {lr.p_value:.2g}", transform=ax.transAxes, ha="right", color=INK2, fontsize=8)
    save(fig, "fig5_time_to_recall_km")
    return t


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df, model = load()
    print(f"All devices: {len(df)}; model sample: {len(model)} "
          f"({model['recalled'].sum()} recalled, {model['any_ae'].sum()} with adverse event reports)")
    fig_growth(df)
    fig_validation_share(df)
    fig_rates(model).to_csv(TAB / "outcome_rates_by_validation.csv", index=False)
    table_descriptives(model)
    recall_model(model)
    ae_model(model)
    cox_model(model)
    pd.DataFrame(PERFORMANCE, columns=["model", "metric", "value"]).to_csv(
        TAB / "model_performance.csv", index=False)
    print(f"\nfigures -> {FIG}\ntables  -> {TAB}")


if __name__ == "__main__":
    main()
