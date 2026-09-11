import numpy as np
import pandas as pd

from scipy.stats import norm
from scipy.stats import t
from scipy.stats import kendalltau


# ---------------------------------------------------
# KENDALL TAU -> COPULA CORRELATION
# ---------------------------------------------------

def tau_to_rho(tau):

    return np.sin(
        np.pi * tau / 2
    )


# ---------------------------------------------------
# REGIONAL DEFAULT RATE PANEL
# ---------------------------------------------------

def build_regional_default_panel(df):

    defaults = df.loc[
        df["default_event"] == 1
    ].copy()

    defaults["year"] = (
        defaults["default_date"]
        .dt.year
    )

    annual_defaults = (
        defaults
        .groupby(["year", "region"])
        .size()
        .unstack(fill_value=0)
    )

    regional_loans = (
        df.groupby("region")
        .size()
    )

    annual_rates = annual_defaults.copy()

    for c in annual_rates.columns:

        annual_rates[c] = (
            annual_rates[c]
            /
            regional_loans[c]
        )

    return annual_rates


# ---------------------------------------------------
# KENDALL TAU MATRIX
# ---------------------------------------------------

def compute_tau_matrix(
    annual_rates
):

    regions = annual_rates.columns

    tau_matrix = pd.DataFrame(
        index=regions,
        columns=regions,
        dtype=float
    )

    for r1 in regions:

        for r2 in regions:

            tau, _ = kendalltau(
                annual_rates[r1],
                annual_rates[r2]
            )

            tau_matrix.loc[
                r1,
                r2
            ] = tau

    return tau_matrix


# ---------------------------------------------------
# AVERAGE TAU
# ---------------------------------------------------

def average_tau(
    tau_matrix
):

    mask = np.triu(
        np.ones(tau_matrix.shape),
        k=1
    ).astype(bool)

    values = tau_matrix.where(
        mask
    ).stack()

    return values.mean()


# ---------------------------------------------------
# BOOTSTRAP CI
# ---------------------------------------------------

def bootstrap_tau(
    annual_rates,
    n_boot=1000,
    seed=42
):

    rng = np.random.default_rng(seed)

    regions = list(
        annual_rates.columns
    )

    results = []

    for _ in range(n_boot):

        sample_idx = rng.choice(
            annual_rates.index,
            size=len(annual_rates),
            replace=True
        )

        sample = annual_rates.loc[
            sample_idx
        ]

        tau_values = []

        for i in range(len(regions)):

            for j in range(i + 1,
                           len(regions)):

                tau, _ = kendalltau(
                    sample[regions[i]],
                    sample[regions[j]]
                )

                tau_values.append(tau)

        results.append(
            np.nanmean(tau_values)
        )

    return np.array(results)


# ---------------------------------------------------
# TAIL DEPENDENCE
# ---------------------------------------------------

def t_tail_dependence(
    rho,
    nu
):

    return (
        2
        * t.cdf(
            -np.sqrt(
                ((nu + 1)
                 * (1 - rho))
                /
                (1 + rho)
            ),
            df=nu + 1
        )
    )


# ---------------------------------------------------
# SIMULATORS
# ---------------------------------------------------

def sample_gaussian_copula(
    rho,
    n_paths,
    n_loans,
    seed=42
):

    rng = np.random.default_rng(seed)

    corr = np.full(
        (n_loans, n_loans),
        rho
    )

    np.fill_diagonal(
        corr,
        1.0
    )

    z = rng.multivariate_normal(
        mean=np.zeros(n_loans),
        cov=corr,
        size=n_paths
    )

    return norm.cdf(z)


def sample_one_factor_t_copula(rho, nu, n_paths, n_loans, seed=42):
    rng = np.random.default_rng(seed)

    global_factor = rng.standard_normal(n_paths)
    eps = rng.standard_normal((n_paths, n_loans))

    z = (
        np.sqrt(rho) * global_factor[:, None]
        + np.sqrt(1 - rho) * eps
    )

    g = rng.chisquare(nu, n_paths)
    x = z / np.sqrt(g[:, None] / nu)

    return t.cdf(x, df=nu)
# ---------------------------------------------------
# FICO DEFAULT RATE PANEL
# ---------------------------------------------------

def build_fico_default_panel(df):

    defaults = df.loc[
        df["default_event"] == 1
    ].copy()

    defaults["year"] = (
        pd.to_datetime(
            defaults["default_date"]
        ).dt.year
    )

    annual_defaults = (
        defaults
        .groupby(
            ["year", "fico_band"]
        )
        .size()
        .unstack(fill_value=0)
    )

    loan_counts = (
        df.groupby("fico_band")
        .size()
    )

    annual_rates = annual_defaults.copy()

    for c in annual_rates.columns:

        annual_rates[c] = (
            annual_rates[c]
            /
            loan_counts[c]
        )

    return annual_rates


def build_region_fico_default_panel(df):
    """
    Annual default-rate panel at the region x FICO-band cell level.
    Provides multiple sub-series per FICO band (one per region), which
    a band-only panel cannot: needed to test whether within-band
    co-movement exceeds cross-band co-movement.
    """
    defaults = df.loc[df["default_event"] == 1].copy()
    defaults["year"] = defaults["default_date"].dt.year
    defaults["cell"] = defaults["region"] + " | " + defaults["fico_band"].astype(str)

    annual_defaults = (
        defaults
        .groupby(["year", "cell"])
        .size()
        .unstack(fill_value=0)
    )

    cell_loans = (
        df.assign(cell=df["region"] + " | " + df["fico_band"].astype(str))
          .groupby("cell")
          .size()
    )

    annual_rates = annual_defaults.copy()
    for c in annual_rates.columns:
        annual_rates[c] = annual_rates[c] / cell_loans[c]

    return annual_rates

# ---------------------------------------------------
# DIVERSIFICATION BENEFIT
# ---------------------------------------------------

def diversification_benefit(
    var_independent,
    var_dependent
):

    return (
        1
        -
        var_independent
        /
        var_dependent
    )
    
