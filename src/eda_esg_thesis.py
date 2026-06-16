"""
=============================================================================
  ESG THESIS — Preliminary Exploratory Data Analysis
  Author : Thomas Nguyen
  Date   : March 2026
  Topic  : Machine Learning to Identify ESG Rating Inconsistencies
           and Improve Portfolio Outcomes
=============================================================================

SETUP (run once in your terminal before executing this script):
    pip install yfinance pandas numpy matplotlib seaborn tqdm scikit-learn

HOW TO RUN:
    Option A — Jupyter Notebook: copy-paste each section into cells
    Option B — Terminal: python src/eda_esg_thesis.py
    Option C — VS Code: open and run interactively with the Python extension

NOTE ON DATA:
    This script uses yfinance (Yahoo Finance / Sustainalytics ESG scores).
    ESG scores are scraped from Yahoo Finance in real time, so results may
    vary slightly across runs. For 100 tickers expect ~5-10 minutes.
=============================================================================
"""

# =============================================================================
# 0. IMPORTS & SETTINGS
# =============================================================================
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm import tqdm
import time
import os

# Visual style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (12, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTPUT_DIR = "outputs/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ESG THESIS — Preliminary EDA")
print("=" * 70)


# =============================================================================
# 1. UNIVERSE — S&P 500 TICKERS WITH SECTORS
# =============================================================================
print("\n[1/6] Fetching S&P 500 universe from Wikipedia...")

def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = pd.read_html(url)[0]
    df.columns = df.columns.str.strip()
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "company",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
        "Date added": "date_added",
    })
    return df[["ticker", "company", "sector", "sub_industry"]]

sp500 = get_sp500()
print(f"  Universe: {len(sp500)} companies across {sp500['sector'].nunique()} sectors")
print(sp500["sector"].value_counts().to_string())


# =============================================================================
# 2. DOWNLOAD ESG DATA (Yahoo Finance / Sustainalytics)
# =============================================================================
print("\n[2/6] Downloading ESG scores (this may take several minutes)...")

# ── CONFIGURATION ────────────────────────────────────────────────────────────
# Set MAX_TICKERS to a smaller number (e.g. 50) for a quick test run,
# or to 503 to attempt the full S&P 500 universe.
MAX_TICKERS = 100          # Recommended: 100 for quick EDA, 503 for full run
SLEEP_BETWEEN = 0.3        # Seconds between requests (be polite to Yahoo)
# ─────────────────────────────────────────────────────────────────────────────

tickers_to_download = sp500["ticker"].tolist()[:MAX_TICKERS]

esg_records = []
failed = []

for ticker in tqdm(tickers_to_download, desc="ESG download"):
    try:
        t = yf.Ticker(ticker)
        sust = t.sustainability
        if sust is not None and not sust.empty:
            row = sust.T.iloc[0].to_dict()
            row["ticker"] = ticker
            esg_records.append(row)
        time.sleep(SLEEP_BETWEEN)
    except Exception as e:
        failed.append(ticker)

esg_raw = pd.DataFrame(esg_records).set_index("ticker")

# Keep core columns (rename for clarity)
core_cols = {
    "totalEsg":           "esg_total",
    "environmentScore":   "esg_env",
    "socialScore":        "esg_soc",
    "governanceScore":    "esg_gov",
    "highestControversy": "controversy",
    "esgPerformance":     "esg_performance_label",
}
available = {k: v for k, v in core_cols.items() if k in esg_raw.columns}
esg = esg_raw[list(available.keys())].rename(columns=available)
esg = esg.apply(pd.to_numeric, errors="coerce")  # coerce label columns to NaN

esg = esg.join(sp500.set_index("ticker")[["sector", "company"]])

print(f"\n  ESG data retrieved for {len(esg)} / {MAX_TICKERS} tickers")
print(f"  Download failures: {len(failed)}")
print(f"\n  ESG Score Summary (lower = better risk in Sustainalytics scale):")
print(esg[["esg_total", "esg_env", "esg_soc", "esg_gov", "controversy"]].describe().round(2))


# =============================================================================
# 3. DOWNLOAD FINANCIAL FUNDAMENTALS
# =============================================================================
print("\n[3/6] Downloading financial fundamentals...")

fund_records = []

for ticker in tqdm(esg.index.tolist(), desc="Fundamentals"):
    try:
        t = yf.Ticker(ticker)
        info = t.info

        row = {
            "ticker":       ticker,
            "market_cap":   info.get("marketCap"),
            "revenue":      info.get("totalRevenue"),
            "net_income":   info.get("netIncomeToCommon"),
            "total_assets": info.get("totalAssets"),
            "total_debt":   info.get("totalDebt"),
            "ebitda":       info.get("ebitda"),
            "roe":          info.get("returnOnEquity"),
            "roa":          info.get("returnOnAssets"),
            "profit_margin":info.get("profitMargins"),
            "debt_equity":  info.get("debtToEquity"),
            "current_ratio":info.get("currentRatio"),
            "beta":         info.get("beta"),
            "pe_ratio":     info.get("trailingPE"),
            "pb_ratio":     info.get("priceToBook"),
            "employees":    info.get("fullTimeEmployees"),
        }
        fund_records.append(row)
        time.sleep(SLEEP_BETWEEN)
    except:
        pass

fund = pd.DataFrame(fund_records).set_index("ticker")

# Derived variables
fund["log_market_cap"] = np.log(fund["market_cap"].clip(lower=1))
fund["leverage"]       = fund["total_debt"] / (fund["total_assets"].clip(lower=1))
fund["profit_margin"]  = fund["profit_margin"].clip(-1, 1)

print(f"  Fundamentals retrieved for {fund.dropna(subset=['market_cap']).shape[0]} tickers")
print(fund[["log_market_cap", "leverage", "roa", "profit_margin", "beta"]].describe().round(3))


# =============================================================================
# 4. MERGE MASTER DATASET
# =============================================================================
print("\n[4/6] Building master dataset...")

df = esg.join(fund, how="inner")
df = df.dropna(subset=["esg_total"])  # require at least the ESG total score

print(f"  Master dataset: {df.shape[0]} firms x {df.shape[1]} variables")
print(f"  Sectors: {df['sector'].nunique()}")


# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n[5/6] Running EDA — generating plots...")

# ── 5.1  MISSING DATA HEATMAP ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
miss = df.isnull().mean().sort_values(ascending=False)
bars = ax.barh(miss.index, miss.values * 100, color=["#d62728" if v > 0.3 else "#2ca02c" for v in miss.values])
ax.set_xlabel("Missing data (%)")
ax.set_title("Missing Data by Variable", fontweight="bold")
ax.axvline(30, color="red", linestyle="--", linewidth=1, label="30% threshold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_missing_data.png")
plt.show()
print("  → Saved: 01_missing_data.png")


# ── 5.2  ESG SCORE DISTRIBUTION ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
esg_cols = [("esg_total", "Total ESG Risk"), ("esg_env", "Environmental"),
            ("esg_soc", "Social"), ("esg_gov", "Governance")]

for ax, (col, label) in zip(axes, esg_cols):
    if col in df.columns:
        ax.hist(df[col].dropna(), bins=25, color="#4878CF", edgecolor="white", alpha=0.85)
        ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean: {df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="orange", linestyle=":", linewidth=1.5,
                   label=f"Median: {df[col].median():.1f}")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Score (lower = less risk)")
        ax.legend(fontsize=8)

plt.suptitle("Distribution of ESG Risk Scores (Sustainalytics)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_esg_distributions.png", bbox_inches="tight")
plt.show()
print("  → Saved: 02_esg_distributions.png")


# ── 5.3  ESG BY SECTOR ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sector_esg = df.groupby("sector")["esg_total"].agg(["mean", "std", "count"]).sort_values("mean")
sector_esg = sector_esg[sector_esg["count"] >= 3]

# Mean total ESG by sector
axes[0].barh(sector_esg.index, sector_esg["mean"],
             xerr=sector_esg["std"], capsize=4,
             color="#4878CF", alpha=0.85, edgecolor="white")
axes[0].set_xlabel("Mean Total ESG Risk Score")
axes[0].set_title("Mean ESG Risk Score by Sector\n(lower = less risk)", fontweight="bold")

# Boxplot of ESG sub-scores by sector
sector_order = df.groupby("sector")["esg_total"].median().sort_values().index.tolist()
df_plot = df[df["sector"].isin(sector_order)].copy()
sns.boxplot(data=df_plot, y="sector", x="esg_total", order=sector_order,
            palette="Blues", ax=axes[1])
axes[1].set_xlabel("Total ESG Risk Score")
axes[1].set_ylabel("")
axes[1].set_title("ESG Risk Score Distribution by Sector", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_esg_by_sector.png", bbox_inches="tight")
plt.show()
print("  → Saved: 03_esg_by_sector.png")


# ── 5.4  ESG vs. FIRM SIZE ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
scatter_vars = [
    ("log_market_cap", "Log Market Cap", "esg_total"),
    ("leverage",       "Leverage (D/A)", "esg_total"),
    ("roa",            "Return on Assets", "esg_total"),
]

for ax, (x, xlabel, y) in zip(axes, scatter_vars):
    sub = df[[x, y, "sector"]].dropna()
    ax.scatter(sub[x], sub[y], alpha=0.5, s=30, color="#4878CF")
    # Regression line
    m, b = np.polyfit(sub[x], sub[y], 1)
    x_line = np.linspace(sub[x].min(), sub[x].max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=2,
            label=f"slope = {m:.2f}")
    corr = sub[[x, y]].corr().iloc[0, 1]
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total ESG Risk Score")
    ax.set_title(f"ESG vs {xlabel}\n(r = {corr:.3f})", fontweight="bold")
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_esg_vs_fundamentals.png", bbox_inches="tight")
plt.show()
print("  → Saved: 04_esg_vs_fundamentals.png")


# ── 5.5  CORRELATION MATRIX ──────────────────────────────────────────────────
numeric_cols = ["esg_total", "esg_env", "esg_soc", "esg_gov", "controversy",
                "log_market_cap", "leverage", "roa", "profit_margin",
                "debt_equity", "beta", "pe_ratio"]
numeric_cols = [c for c in numeric_cols if c in df.columns]

corr_matrix = df[numeric_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8}, annot_kws={"size": 8})
ax.set_title("Correlation Matrix — ESG Scores & Financial Fundamentals",
             fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_correlation_matrix.png", bbox_inches="tight")
plt.show()
print("  → Saved: 05_correlation_matrix.png")


# ── 5.6  ESG SUB-SCORE SCATTER MATRIX ────────────────────────────────────────
sub_cols = [c for c in ["esg_total", "esg_env", "esg_soc", "esg_gov"] if c in df.columns]
g = sns.PairGrid(df[sub_cols + ["sector"]].dropna(subset=sub_cols), hue="sector",
                 diag_sharey=False, height=2.5)
g.map_diag(sns.histplot, kde=True, alpha=0.6)
g.map_offdiag(sns.scatterplot, alpha=0.4, s=20)
g.add_legend(title="Sector", bbox_to_anchor=(1.05, 0.5))
g.figure.suptitle("ESG Sub-Score Pair Plot", fontweight="bold", y=1.01)
plt.savefig(f"{OUTPUT_DIR}/06_esg_pairplot.png", bbox_inches="tight")
plt.show()
print("  → Saved: 06_esg_pairplot.png")


# ── 5.7  CONTROVERSY DISTRIBUTION ────────────────────────────────────────────
if "controversy" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 5))
    controversy_counts = df["controversy"].value_counts().sort_index()
    labels = {0: "No controversy", 1: "Low", 2: "Moderate", 3: "Significant", 4: "High", 5: "Severe"}
    controversy_counts.index = [labels.get(int(i), str(i)) for i in controversy_counts.index]
    controversy_counts.plot(kind="bar", ax=ax, color="#d62728", alpha=0.8, edgecolor="white")
    ax.set_title("Controversy Score Distribution", fontweight="bold")
    ax.set_xlabel("Controversy Level")
    ax.set_ylabel("Number of Firms")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_controversy.png", bbox_inches="tight")
    plt.show()
    print("  → Saved: 07_controversy.png")


# ── 5.8  PRELIMINARY INCONSISTENCY PROXY ─────────────────────────────────────
# Simple OLS residual as a first-pass inconsistency proxy
# (before the full ML model is trained)
print("\n[6/6] Computing preliminary OLS-based ESG inconsistency proxy...")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

feature_cols = [c for c in ["log_market_cap", "leverage", "roa",
                             "profit_margin", "beta"] if c in df.columns]
target_col   = "esg_total"

analysis_df = df[[target_col] + feature_cols + ["sector"]].dropna()

# One-hot encode sector
analysis_df_enc = pd.get_dummies(analysis_df, columns=["sector"], drop_first=True)

X = analysis_df_enc.drop(columns=[target_col])
y = analysis_df_enc[target_col]

pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
pipe.fit(X, y)

y_pred = pipe.predict(X)
residuals = y - y_pred

analysis_df["esg_predicted_ols"] = y_pred
analysis_df["inconsistency_proxy"] = residuals

r2 = pipe.score(X, y)
print(f"  OLS baseline R² (in-sample): {r2:.3f}")
print(f"  Inconsistency proxy — mean: {residuals.mean():.4f}, std: {residuals.std():.3f}")

# Plot inconsistency distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals, bins=30, color="#4878CF", edgecolor="white", alpha=0.85)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Inconsistency Score (OLS residual)")
axes[0].set_ylabel("Number of Firms")
axes[0].set_title(f"Distribution of Preliminary ESG Inconsistency Scores\n(OLS baseline, R² = {r2:.3f})", fontweight="bold")

# Top overrated vs underrated firms
analysis_df["company"] = df.loc[analysis_df.index, "company"]
top_over  = analysis_df.nlargest(10, "inconsistency_proxy")[["company", "inconsistency_proxy", "esg_total"]]
top_under = analysis_df.nsmallest(10, "inconsistency_proxy")[["company", "inconsistency_proxy", "esg_total"]]

colors_over  = ["#d62728"] * 10
colors_under = ["#2ca02c"] * 10

axes[1].barh(top_over["company"].values[::-1],
             top_over["inconsistency_proxy"].values[::-1],
             color=colors_over, alpha=0.85, label="Over-rated (positive residual)")
axes[1].barh(top_under["company"].values[::-1],
             top_under["inconsistency_proxy"].values[::-1],
             color=colors_under, alpha=0.85, label="Under-rated (negative residual)")
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_xlabel("Inconsistency Score")
axes[1].set_title("Top 10 Most Over- & Under-Rated Firms\n(vs. OLS prediction from fundamentals)", fontweight="bold")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_inconsistency_proxy.png", bbox_inches="tight")
plt.show()
print("  → Saved: 08_inconsistency_proxy.png")

# Inconsistency by sector
fig, ax = plt.subplots(figsize=(12, 5))
sector_incons = analysis_df.groupby("sector")["inconsistency_proxy"].mean().sort_values()
colors = ["#d62728" if v > 0 else "#2ca02c" for v in sector_incons.values]
ax.barh(sector_incons.index, sector_incons.values, color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=1)
ax.set_xlabel("Mean ESG Inconsistency Score")
ax.set_title("Mean ESG Inconsistency by Sector\n(positive = sector tends to be over-rated vs. fundamentals)",
             fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_inconsistency_by_sector.png", bbox_inches="tight")
plt.show()
print("  → Saved: 09_inconsistency_by_sector.png")


# =============================================================================
# 6. SUMMARY REPORT
# =============================================================================
print("\n" + "=" * 70)
print("EDA COMPLETE — Summary Statistics")
print("=" * 70)
print(f"\n  Sample size        : {len(df)} firms")
print(f"  Sectors covered    : {df['sector'].nunique()}")
print(f"  ESG coverage rate  : {(~df['esg_total'].isna()).mean():.1%}")
print(f"\n  ESG Total Score    : mean = {df['esg_total'].mean():.1f}, "
      f"std = {df['esg_total'].std():.1f}, "
      f"range = [{df['esg_total'].min():.1f}, {df['esg_total'].max():.1f}]")
print(f"  OLS Baseline R²    : {r2:.3f}  (target for ML model: > 0.30)")
print(f"\n  Plots saved in     : ./{OUTPUT_DIR}/")
print(f"  Files: {sorted(os.listdir(OUTPUT_DIR))}")

# Export master dataset
df.to_csv(f"{OUTPUT_DIR}/master_dataset.csv")
analysis_df[["esg_total", "esg_predicted_ols", "inconsistency_proxy",
             "sector", "company"]].to_csv(f"{OUTPUT_DIR}/inconsistency_scores.csv")
print(f"\n  Data exported to   : ./{OUTPUT_DIR}/master_dataset.csv")
print(f"                       ./{OUTPUT_DIR}/inconsistency_scores.csv")
print("\nDone! Review the plots above and the exported CSVs for your thesis.")
