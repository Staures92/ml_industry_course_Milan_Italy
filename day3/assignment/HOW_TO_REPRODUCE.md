# HOW_TO_REPRODUCE

## Day 3 Take-Home: Consumer Credit Portfolio Risk

This document lists the exact steps, in order, to regenerate every result in
`REPORT.pdf` from a clean checkout of the repository fork.

---

## Global Reproducibility Settings

- **Random seed:** `SEED = 42` everywhere. Set at the top of every notebook via
  `np.random.seed(SEED)` and passed explicitly to every stochastic function
  (`sample_one_factor_t_copula`, `bootstrap_tau`, `bootstrap_var_ci`, the
  FICO-band stratified sample draw, etc.). No result in this submission depends
  on an unseeded random draw.
- **Dependencies:** pinned via `uv.lock` at the repository root.
- **Working directory convention:** all notebooks assume they are run from
  `day3/notebooks/` and add `../src` to `sys.path` to import the pipeline
  modules (`cleaning.py`, `amortization.py`, `marginals.py`, `dependence.py`,
  `simulation.py`).

---

## Environment Setup

```bash
git clone <fork-url>
cd <fork-name>
git checkout assignment-day3-<yourname>
uv sync                       # installs pinned dependencies from uv.lock
cd day3/notebooks
jupyter lab                   # or: jupyter notebook
```

---

## Execution Order

Run the following in exact sequence. Each step's output feeds the next step's
input; skipping or reordering will produce inconsistent or missing
intermediate files.

### 1. `cleaning.ipynb` --- Part 1 (data cleaning, EDA, EAD/amortization reconstruction)

- Restart the kernel before running (**Kernel → Restart Kernel and Run All
  Cells**). Because `cleaning.py` is imported once at the top of the notebook,
  editing the module file after the kernel has started will *not* pick up the
  change without a restart or an explicit `importlib.reload(cleaning)`.
- Reads: `../generated/loan_tape.csv` (raw tape, 308,222 rows).
- Writes: `../outputs/cleaned/loan_tape_clean.csv` (277,867 rows),
  `../outputs/tables/data_quality_log.csv`, and the seven Part 1 figures to
  `../outputs/figures/`.

### 2. `marginals.ipynb` --- Part 2 (survival family comparison, FICO-band segmentation)

- Reads: `../outputs/cleaned/loan_tape_clean.csv`.
- Fits Kaplan-Meier, Aalen-Johansen, Weibull, Log-Normal, Log-Logistic on the
  pooled tape, then Weibull by FICO band, by region, by term, and by
  amortization type (four segmentation comparisons).
- Writes: `../outputs/tables/marginal_model_diagnostics.csv`,
  `../outputs/tables/marginal_model_parameters.csv`, and six figures to
  `../outputs/figures/`.
- **Expected result:** Weibull selected on AIC/BIC. FICO-band segmentation
  selected over region, term, and amortization-type segmentation (Delta AIC ≈
  14,056 vs. pooled).

### 3. `dependence.ipynb` --- Part 3 (copula selection) and Part 4 (diversification)

- Reads: `../outputs/cleaned/loan_tape_clean.csv`.
- Builds annual default-rate panels by FICO band and by region
  (`build_fico_default_panel`, `build_regional_default_panel`), computes
  Kendall tau matrices, average tau, and its bootstrap CI.
- Builds the finer region x FICO panel (`build_region_fico_default_panel`) and
  runs the within-vs-cross segment permutation test (5,000 permutations,
  seed 42).
- Fits nu by matching the Student-t copula's implied upper-tail dependence to
  the empirical estimate (75th-percentile joint-exceedance method), and
  bootstraps a 95% CI on nu (500 resamples, years resampled).
- Writes: `fico_dependence_matrix.pdf`, `regional_default_dependence.pdf` to
  `../outputs/figures/`.
- **Expected result:** average τ ≈ 0.379, ρ ≈ 0.5602, permutation test p ~ 0.403 (One-Factor structure selected, Hierarchical rejected), fitted nu~ 2.74 with 95% CI ~ [2.05, 4.79].

### 4. `simulation_concentration.ipynb` (or `.py`) --- Part 5 (Monte Carlo loss simulation) and Part 6 (concentration & attribution)

- Reads: `../outputs/cleaned/loan_tape_clean.csv`.
- Draws a FICO-band-stratified sample of 4,900 loans (`frac = 5000 /
  len(df)`, seed 42).
- Uses the FICO-band Weibull `shape_lookup`/`scale_lookup` from Part 2 (*not*
  the Log-Normal `mu_lookup`/`sigma_lookup` used in earlier drafts of this
  pipeline).
- Draws copula uniforms via `sample_one_factor_t_copula(rho=0.5602, nu=2.74,
  n_paths=10000, n_loans=4900, seed=42)`.
- Maps copula uniforms to simulated default times via
  `scipy.stats.weibull_min.ppf` per loan's FICO band.
- Imputes missing `interest_rate` with the FICO-band median *before*
  computing EAD (avoids `NaN` propagation through `outstanding_balance()`).
- Recomputes EAD at each loan's simulated default time via
  `simulated_ead_matrix()` (calls `outstanding_balance()` from
  `amortization.py`).
- Computes risk measures (`portfolio_risk_measures`), MC confidence intervals
  on VaR(99.9)(`bootstrap_var_ci`, 500 resamples), the independence-benchmark
  diversification benefit, Component VaR (`component_var`), the Euler
  additivity check, region attribution, and concentration indices (HHI,
  top-10 share, diversification ratio).
- **Expected result (sample scale, 4,900 loans):** EL ≈ €896,954; VaR(99.9)
  (Student-t) ~ €12.86M; EC(99.9)~ €11.97M; independence understates EC(99.9) by
  ~91.8%; top-10 Component-VaR share ≈0.41%; risk HHI ~0.0003.

---

## Known Non-Determinism

None. Every stochastic draw in this pipeline (deduplication order, stratified
sampling, copula simulation, and all bootstrap procedures) is seeded with
`SEED = 42`, either directly or via a fixed `np.random.default_rng(42)`
instance. Re-running the full sequence above from a clean checkout should
reproduce every figure in `REPORT.pdf` exactly.
