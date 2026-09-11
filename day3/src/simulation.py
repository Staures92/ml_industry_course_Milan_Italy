import numpy as np

from scipy.stats import lognorm, weibull_min


# ----------------------------------------
# DEFAULT TIMES
# ----------------------------------------

def simulate_default_times(
    uniforms, 
    shape, 
    scale
):
    
    return weibull_min.ppf(
          uniforms, 
          c=shape, 
          scale=scale
    )

# ----------------------------------------
# DEFAULT FLAGS
# ----------------------------------------

def default_within_horizon(
    default_times,
    horizon_months
):

    return (
        default_times
        <= horizon_months
    )


# ----------------------------------------
# FIXED LGD
# ----------------------------------------

def apply_lgd(
    ead_matrix,
    default_flags,
    lgd=0.45
):

    return (
        ead_matrix *
        default_flags *
        lgd
    )


# ----------------------------------------
# PORTFOLIO LOSS
# ----------------------------------------

def portfolio_loss(loss_matrix):

    return loss_matrix.sum(
        axis=1
    )


# ----------------------------------------
# RISK MEASURES
# ----------------------------------------

def portfolio_risk_measures(
    losses
):

    EL = losses.mean()

    VaR95 = np.quantile(
        losses,
        0.95
    )

    VaR99 = np.quantile(
        losses,
        0.99
    )

    VaR999 = np.quantile(
        losses,
        0.999
    )

    ES95 = losses[
        losses >= VaR95
    ].mean()

    ES99 = losses[
        losses >= VaR99
    ].mean()

    ES999 = losses[
        losses >= VaR999
    ].mean()

    return {
        "EL": EL,
        "VaR95": VaR95,
        "VaR99": VaR99,
        "VaR999": VaR999,
        "ES95": ES95,
        "ES99": ES99,
        "ES999": ES999,
        "EC999": VaR999 - EL
    }


# ----------------------------------------
# MC CONFIDENCE INTERVAL
# ----------------------------------------

def bootstrap_var_ci(
    losses,
    alpha=0.999,
    n_boot=500,
    seed=42
):

    rng = np.random.default_rng(seed)

    estimates = []

    for _ in range(n_boot):

        sample = rng.choice(
            losses,
            len(losses),
            replace=True
        )

        estimates.append(
            np.quantile(
                sample,
                alpha
            )
        )

    return np.percentile(
        estimates,
        [2.5, 97.5]
    )


# ----------------------------------------
# COMPONENT VAR
# ----------------------------------------

def component_var(
    loss_matrix,
    portfolio_losses,
    alpha=0.999
):

    var_level = np.quantile(
        portfolio_losses,
        alpha
    )

    tail = (
        portfolio_losses
        >=
        var_level
    )

    cvar = (
        loss_matrix[tail]
        .mean(axis=0)
    )

    return cvar