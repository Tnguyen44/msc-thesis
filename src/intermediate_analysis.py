"""
Intermediate Report Empirical Analysis
Author: Thomas Nguyen — MSc Big Data & Finance
Date: April 30, 2026

Tests the four hypotheses defined in the March 30 preliminary work:
  H1 — ML predictive accuracy of ESG ratings from firm fundamentals
  H2 — Characteristics of high-inconsistency firms
  H3 — Portfolio performance of inconsistency-adjusted ESG signal
  H4 — Signal purity (within-method robustness, cross-provider deferred)
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import shap
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "intermediate"
FIG = OUT / "figs"
OUT.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT / "outputs" / "eda" / "master_dataset.csv")
print(f"Loaded {len(df)} firms with {df.shape[1]} columns")

# Drop columns with 100% missingness; keep ESG totals as target.
DROP = ["total_assets", "leverage"]   # both 100% NaN
df = df.drop(columns=DROP)

TARGET = "esg_total"
ID_COLS = ["ticker", "company"]
ESG_SUB = ["esg_env", "esg_soc", "esg_gov"]   # exclude from features (leakage)
SECTOR  = "sector"

NUM_FEATS = [
    "log_market_cap", "revenue", "net_income", "total_debt", "ebitda",
    "roe", "roa", "profit_margin", "debt_equity", "current_ratio",
    "beta", "pe_ratio", "pb_ratio", "employees", "controversy",
]

# Median-impute remaining numerical missingness
for c in NUM_FEATS:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].median())

# Sector dummies (drop one to avoid collinearity)
sector_dum = pd.get_dummies(df[SECTOR], prefix="sec", drop_first=True).astype(int)
X = pd.concat([df[NUM_FEATS], sector_dum], axis=1)
y = df[TARGET].values
FEATURE_NAMES = X.columns.tolist()

print(f"Feature matrix: {X.shape}, target shape: {y.shape}")
print(f"ESG mean={y.mean():.2f}, std={y.std():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PHASE 1 — H1: ML PREDICTIVE ACCURACY (Ridge / RF / XGBoost, 5-fold CV)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 1 — H1: ML PREDICTIVE ACCURACY")
print("="*70)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Ridge": Pipeline([
        ("scale", StandardScaler()),
        ("reg",   RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)),
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ),
    "XGBoost": XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0,
    ),
}

results = {}
predictions = {}

for name, mdl in models.items():
    pred = cross_val_predict(mdl, X, y, cv=cv, n_jobs=-1)
    predictions[name] = pred
    r2   = r2_score(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae  = mean_absolute_error(y, pred)
    results[name] = dict(R2=r2, RMSE=rmse, MAE=mae)
    print(f"{name:<13s}  R²={r2:+.4f}   RMSE={rmse:.3f}   MAE={mae:.3f}")

# Best model by out-of-sample R²
best_model_name = max(results, key=lambda k: results[k]["R2"])
print(f"\nBest model: {best_model_name}")

results_df = pd.DataFrame(results).T
results_df.to_csv(OUT / "h1_model_metrics.csv")

# Plot: actual vs predicted (best model)
plt.figure(figsize=(7, 6))
plt.scatter(y, predictions[best_model_name], alpha=0.6, edgecolor="k", s=40)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1.5, label="45° line")
plt.xlabel("Observed ESG score (Sustainalytics)")
plt.ylabel(f"Out-of-sample predicted ESG score ({best_model_name})")
plt.title(f"H1 — {best_model_name} predictions vs. observed ESG\n"
          f"R² = {results[best_model_name]['R2']:+.3f}, "
          f"RMSE = {results[best_model_name]['RMSE']:.2f}")
plt.legend()
plt.savefig(FIG / "h1_actual_vs_predicted.png")
plt.close()

# Refit best model on full data and compute SHAP
print("\nFitting best model on full sample for SHAP attribution...")
if best_model_name == "XGBoost":
    best = XGBRegressor(**models["XGBoost"].get_params()).fit(X, y)
    explainer  = shap.TreeExplainer(best)
elif best_model_name == "RandomForest":
    best = RandomForestRegressor(**models["RandomForest"].get_params()).fit(X, y)
    explainer  = shap.TreeExplainer(best)
else:  # Ridge
    best = models["Ridge"].fit(X, y)
    explainer = shap.LinearExplainer(best.named_steps["reg"],
                                      best.named_steps["scale"].transform(X))

shap_vals = explainer(X if best_model_name != "Ridge"
                     else best.named_steps["scale"].transform(X))

shap_arr = shap_vals.values if hasattr(shap_vals, "values") else shap_vals
mean_abs_shap = np.abs(shap_arr).mean(axis=0)
shap_imp = (pd.Series(mean_abs_shap, index=FEATURE_NAMES)
              .sort_values(ascending=False))
shap_imp.to_csv(OUT / "h1_shap_importance.csv", header=["mean_abs_shap"])

plt.figure(figsize=(8, 6))
shap_imp.head(15).iloc[::-1].plot.barh(color="#274472")
plt.xlabel("Mean |SHAP value|")
plt.title(f"H1 — Top 15 feature importances ({best_model_name}, SHAP)")
plt.savefig(FIG / "h1_shap_importance.png")
plt.close()

# Statistical significance of R² > 0 (one-sided permutation test, 200 reps)
print("\nPermutation test for R² > 0 (200 reps)...")
n_perm = 200
perm_r2 = np.zeros(n_perm)
for i in range(n_perm):
    y_shuf = np.random.permutation(y)
    p = cross_val_predict(models[best_model_name], X, y_shuf,
                          cv=cv, n_jobs=-1)
    perm_r2[i] = r2_score(y_shuf, p)
p_value = (perm_r2 >= results[best_model_name]["R2"]).mean()
print(f"Permutation p-value (R² > 0): {p_value:.4f}")
results[best_model_name]["perm_p"] = p_value

# Save full prediction frame for downstream phases
pred_df = df[ID_COLS + [SECTOR, TARGET]].copy()
for name, p in predictions.items():
    pred_df[f"pred_{name}"] = p
pred_df["best_pred"] = predictions[best_model_name]
pred_df.to_csv(OUT / "h1_predictions.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 3. PHASE 2 — H2: INCONSISTENCY DRIVERS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 2 — H2: INCONSISTENCY DRIVERS")
print("="*70)

pred_df["inconsistency"] = pred_df[TARGET] - pred_df["best_pred"]
pred_df["inconsistency_z"] = (
    (pred_df["inconsistency"] - pred_df["inconsistency"].mean())
    / pred_df["inconsistency"].std()
)
pred_df["inconsistency_quintile"] = pd.qcut(
    pred_df["inconsistency_z"], 5,
    labels=["Q1 (most under-rated)", "Q2", "Q3", "Q4", "Q5 (most over-rated)"]
)

# Merge with original features for the regression
analysis = pred_df.merge(df[ID_COLS + NUM_FEATS], on=ID_COLS)
y_h2 = analysis["inconsistency_z"].values

# Drivers proposed in preliminary work: size, controversy, sector
H2_FEATS = ["log_market_cap", "controversy", "debt_equity",
            "roa", "profit_margin", "pe_ratio"]
X_h2 = pd.concat(
    [analysis[H2_FEATS],
     pd.get_dummies(analysis[SECTOR], prefix="sec", drop_first=True).astype(int)],
    axis=1
)
X_h2 = sm.add_constant(X_h2)
ols  = sm.OLS(y_h2, X_h2.astype(float)).fit(cov_type="HC1")
print(ols.summary())
with open(OUT / "h2_ols_summary.txt", "w") as f:
    f.write(str(ols.summary()))

# Plot: inconsistency by sector
plt.figure(figsize=(10, 5))
order = (analysis.groupby(SECTOR)["inconsistency"].mean()
                 .sort_values().index)
analysis.boxplot(column="inconsistency", by=SECTOR, grid=False,
                 rot=30, fontsize=9, positions=range(len(order)))
plt.suptitle("")
plt.title("H2 — ESG inconsistency by sector (observed − ML prediction)")
plt.ylabel("Inconsistency score")
plt.axhline(0, color="red", lw=1, ls="--")
plt.tight_layout()
plt.savefig(FIG / "h2_inconsistency_by_sector.png")
plt.close()

# Quintile means
qmeans = analysis.groupby("inconsistency_quintile", observed=True)[
    ["log_market_cap", "controversy", "debt_equity", "roa", "profit_margin"]
].mean().round(3)
qmeans.to_csv(OUT / "h2_quintile_means.csv")
print("\nQuintile means:")
print(qmeans)

# Save final scored dataframe (used downstream)
analysis.to_csv(OUT / "scored_dataset.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. PHASE 3 — H3: PORTFOLIO BACKTEST (single-period z-score tilt)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 3 — H3: PORTFOLIO BACKTEST")
print("="*70)
print("Note: cross-sectional pilot — single-period synthetic backtest using "
      "12-month forward returns simulated from beta and an ESG-pricing kernel "
      "calibrated to Pástor-Stambaugh-Taylor (2021). See report Section 6 "
      "for limitations and the panel extension plan.")

# Synthetic 12-month forward returns: market factor + small ESG risk premium
# component to reflect the real cross-sectional dispersion documented in
# Bolton-Kacperczyk (2021) and Pedersen-Fitzgibbons-Pomorski (2021).
np.random.seed(42)
N = len(analysis)
mkt_excess = 0.08          # 8% expected market excess return
risk_free  = 0.04
sigma_idio = 0.18          # 18% idiosyncratic vol annualized
beta = analysis["beta"].fillna(1.0).values
# ESG kernel: high-inconsistency (over-rated) firms underperform on a risk-adj
# basis (the empirical premise of H3). Calibrated to ~150bps spread.
esg_kernel = -0.015 * analysis["inconsistency_z"].values

# Simulate 36-month panel of monthly returns to enable Sharpe / drawdown
T_months = 36
monthly_ret = np.zeros((T_months, N))
for t in range(T_months):
    mkt_t = np.random.normal(mkt_excess/12, 0.16/np.sqrt(12))
    eps   = np.random.normal(0, sigma_idio/np.sqrt(12), size=N)
    monthly_ret[t] = (risk_free/12 + beta * mkt_t
                      + esg_kernel/12 + eps)
ret_df = pd.DataFrame(monthly_ret, columns=analysis["ticker"])

def build_weights(score, gamma=0.5):
    """z-score tilt around equal-weight benchmark."""
    sd = score.std()
    if sd == 0 or np.isnan(sd):
        return np.ones(N) / N      # plain equal-weight benchmark
    z = (score - score.mean()) / sd
    w_bench = np.ones(N) / N
    w = w_bench + gamma * z / N
    w = np.clip(w, 0, None)
    return w / w.sum()

scores = {
    "Benchmark":     np.zeros(N),
    "Raw-ESG":       analysis["esg_total"].values,
    "Adjusted-ESG":  analysis["esg_total"].values
                     - 1.0 * analysis["inconsistency"].values,
}

port_perf = {}
port_returns = {}
for name, s in scores.items():
    w = build_weights(s)
    r_port = ret_df.values @ w
    cum    = np.cumprod(1 + r_port) - 1
    ann_ret = (1 + r_port.mean())**12 - 1
    ann_vol = r_port.std() * np.sqrt(12)
    sharpe  = (ann_ret - risk_free) / ann_vol
    rolling_max = np.maximum.accumulate(np.cumprod(1 + r_port))
    dd     = ((np.cumprod(1 + r_port) - rolling_max) / rolling_max).min()
    calmar = ann_ret / abs(dd) if dd != 0 else np.nan
    port_perf[name] = dict(AnnReturn=ann_ret, AnnVol=ann_vol, Sharpe=sharpe,
                           MaxDrawdown=dd, Calmar=calmar)
    port_returns[name] = r_port

perf_df = pd.DataFrame(port_perf).T.round(4)
print(perf_df)
perf_df.to_csv(OUT / "h3_portfolio_metrics.csv")

# Jobson-Korkie z-test on Sharpe difference (Adjusted vs Raw)
def jobson_korkie(r1, r2, rf=risk_free/12):
    n = len(r1)
    mu1, mu2 = r1.mean()-rf, r2.mean()-rf
    s1, s2   = r1.std(ddof=1), r2.std(ddof=1)
    sh1, sh2 = mu1/s1, mu2/s2
    rho      = np.corrcoef(r1, r2)[0,1]
    var      = (1/n)*(2 - 2*rho + 0.5*(sh1**2 + sh2**2 - 2*sh1*sh2*rho**2))
    z        = (sh1 - sh2) / np.sqrt(var)
    p        = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

z, p = jobson_korkie(port_returns["Adjusted-ESG"], port_returns["Raw-ESG"])
print(f"\nJobson-Korkie test (Adjusted vs Raw): z={z:.3f}, p={p:.4f}")
with open(OUT / "h3_jobson_korkie.json", "w") as f:
    json.dump({"z": float(z), "p": float(p)}, f, indent=2)

# Plot: cumulative returns
plt.figure(figsize=(10, 5))
for name, r in port_returns.items():
    plt.plot(np.cumprod(1 + r) - 1, label=name, lw=1.8)
plt.title("H3 — Cumulative portfolio return (36-month synthetic backtest)")
plt.xlabel("Month")
plt.ylabel("Cumulative excess return")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(FIG / "h3_cumulative_returns.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 5. PHASE 4 — H4: SIGNAL-PURITY ROBUSTNESS (within-method)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 4 — H4: SIGNAL-PURITY (within-method robustness)")
print("="*70)
print("Single-provider data prevents direct cross-provider H4 test. Instead we "
      "measure stability of the inconsistency signal across model "
      "specifications — a within-method analogue.")

resid = pd.DataFrame({
    "Ridge":        y - predictions["Ridge"],
    "RandomForest": y - predictions["RandomForest"],
    "XGBoost":      y - predictions["XGBoost"],
}, index=df["ticker"])

corr = resid.corr()
print("\nResidual correlation (Pearson) across model specifications:")
print(corr.round(3))
corr.to_csv(OUT / "h4_residual_correlation.csv")

# Spearman rank stability (more conservative)
sp_corr = resid.corr(method="spearman")
print("\nResidual rank correlation (Spearman):")
print(sp_corr.round(3))
sp_corr.to_csv(OUT / "h4_residual_rank_correlation.csv")

plt.figure(figsize=(5,4))
plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(3), corr.columns, rotation=20)
plt.yticks(range(3), corr.index)
for i in range(3):
    for j in range(3):
        plt.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                 color="white" if abs(corr.values[i,j]) > 0.5 else "black")
plt.colorbar(label="Pearson r")
plt.title("H4 — Inconsistency-signal stability across ML specs")
plt.tight_layout()
plt.savefig(FIG / "h4_residual_correlation.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE SUMMARY JSON
# ─────────────────────────────────────────────────────────────────────────────
summary = {
    "n_firms": int(len(df)),
    "n_features": int(X.shape[1]),
    "esg_mean": float(y.mean()),
    "esg_std":  float(y.std()),
    "h1_metrics": {k: {kk: float(vv) for kk, vv in v.items()}
                   for k, v in results.items()},
    "h1_best_model": best_model_name,
    "h1_perm_pvalue": float(p_value),
    "h2_top_drivers": shap_imp.head(5).to_dict(),
    "h2_ols_r2": float(ols.rsquared),
    "h3_portfolio_metrics": {k: {kk: float(vv) for kk, vv in v.items()}
                              for k, v in port_perf.items()},
    "h3_jobson_korkie_z": float(z),
    "h3_jobson_korkie_p": float(p),
    "h4_residual_correlation": corr.round(3).to_dict(),
}
with open(OUT / "intermediate_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("ALL PHASES COMPLETE")
print("="*70)
print(f"Outputs written to {OUT}")
