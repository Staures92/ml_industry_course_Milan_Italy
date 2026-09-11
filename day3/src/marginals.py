import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from lifelines import (
    KaplanMeierFitter,
    WeibullFitter,
    LogNormalFitter,
    LogLogisticFitter,
    AalenJohansenFitter
)


# Plotting style

def set_plot_style():

    plt.rcParams.update({

        "figure.figsize": (9, 6),

        "figure.dpi": 120,

        "savefig.dpi": 300,

        "font.size": 11,

        "axes.titlesize": 14,

        "axes.labelsize": 12,

        "legend.fontsize": 10,

        "xtick.labelsize": 10,

        "ytick.labelsize": 10,

        "axes.spines.top": False,

        "axes.spines.right": False,

        "axes.grid": True,

        "grid.alpha": 0.25,

        "grid.linestyle": "--"

    })


# Output directory

def ensure_output_directory():

    output_dir = Path("../outputs/figures")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return output_dir



# Survival data preparation

def prepare_survival_data(df):

    df = df.copy()

    # Loan duration in months
    df["duration_months"] = (

        (
            df["analysis_end_date"]
            -
            df["origination_date"]
        ).dt.days
        /
        30.4375

    )

    # Default event indicator
    df["default_event"] = (

        df["event_type"]
        ==
        "default"

    ).astype(int)

    # Prepayment event indicator
    df["prepayment_event"] = (

        df["event_type"]
        ==
        "prepayment"

    ).astype(int)

    # Remove invalid durations
    df = df[
        df["duration_months"] > 0
    ].copy()

    return df



# KAPLAN-MEIER estimator 

def fit_kaplan_meier(df):

    kmf = KaplanMeierFitter()

    kmf.fit(

        durations=df["duration_months"],

        event_observed=df["default_event"],

        label="Kaplan–Meier"

    )

    return kmf



# KAPLAN-MEIER plot

def plot_kaplan_meier(

    kmf,

    save=True

):

    set_plot_style()

    output_dir = ensure_output_directory()

    fig, ax = plt.subplots()

    kmf.plot_survival_function(

        ax=ax,

        ci_show=True,

        linewidth=1.5

    )

    ax.set_title(

        "Kaplan–Meier Estimate of Loan Survival",

        fontweight="bold"

    )

    ax.set_xlabel(
        "Loan Age (Months)"
    )

    ax.set_ylabel(
        "Survival Probability"
    )

    ax.set_ylim(
        0.8,
        1.02
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    fig.savefig(

            output_dir
            /
            "kaplan_meier_survival.pdf",

            bbox_inches="tight"

        )

    plt.show()

    return fig, ax



# Competing risk events

def create_competing_risk_event(df):

    event_code = np.zeros(

        len(df),

        dtype=int

    )

    # 1 = Default
    event_code[
        df["event_type"] == "default"
    ] = 1

    # 2 = Prepayment
    event_code[
        df["event_type"] == "prepayment"
    ] = 2

    return event_code


# AALEN-JOHANSEN estimator for competing risks

def fit_aalen_johansen_default(df):

    event_code = create_competing_risk_event(df)

    ajf = AalenJohansenFitter()

    ajf.fit(

        durations=df["duration_months"],

        event_observed=event_code,

        event_of_interest=1

    )

    return ajf


#  AALEN-JOHANSEN plot

def plot_aalen_johansen(

    ajf,

    save=True

):

    set_plot_style()

    output_dir = ensure_output_directory()

    fig, ax = plt.subplots()

    ajf.plot(

        ax=ax,

        linewidth=1.5

    )

    ax.set_title(

        "Aalen–Johansen Cumulative Incidence of Default",

        fontweight="bold"

    )

    ax.set_xlabel(
        "Loan Age (Months)"
    )

    ax.set_ylabel(
        "Cumulative Incidence of Default"
    )

    ax.set_ylim(
        bottom=0
    )

    ax.legend(
        ["Default"],
        frameon=True
    )

    fig.tight_layout()

  
    fig.savefig(

            output_dir
            /
            "aalen_johansen_default.pdf",

            bbox_inches="tight"

        )

    plt.show()

    return fig, ax

# Parametric survival models

def fit_weibull(df):

    model = WeibullFitter()

    model.fit(

        durations=df["duration_months"],

        event_observed=df["default_event"],

        label="Weibull"

    )

    return model


def fit_lognormal(df):

    model = LogNormalFitter()

    model.fit(

        durations=df["duration_months"],

        event_observed=df["default_event"],

        label="Log-Normal"

    )

    return model


def fit_loglogistic(df):

    model = LogLogisticFitter()

    model.fit(

        durations=df["duration_months"],

        event_observed=df["default_event"],

        label="Log-Logistic"

    )

    return model



# Model comparison plot

def plot_parametric_models(

    kmf,

    weibull,

    lognormal,

    loglogistic,

    save=True

):

    set_plot_style()

    output_dir = ensure_output_directory()

    fig, ax = plt.subplots(

        figsize=(10, 6)

    )


    # Kaplan-Meier

    kmf.plot_survival_function(

        ax=ax,

        ci_show=False,

        linewidth=1.5,

        label="Kaplan–Meier (Non-parametric)"

    )


    # Weibull

    weibull.plot_survival_function(

        ax=ax,

        linewidth=1.5,

        label="Weibull"

    )


    # Log-Normal

    lognormal.plot_survival_function(

        ax=ax,

        linewidth=1.5,

        label="Log-Normal"

    )


    # Log-Logistic

    loglogistic.plot_survival_function(

        ax=ax,

        linewidth=1.5,

        label="Log-Logistic"

    )


    ax.set_title(

        "Comparison of Marginal Survival Models",

        fontweight="bold"

    )


    ax.set_xlabel(
        "Loan Age (Months)"
    )


    ax.set_ylabel(
        "Survival Probability"
    )


    ax.set_ylim(
        0.8,
        1.02
    )


    ax.legend(

        loc="best",

        frameon=True

    )


    fig.tight_layout()


    fig.savefig(

            output_dir
            /
            "marginal_survival_comparison.pdf",

            bbox_inches="tight"

        )


    plt.show()


    return fig, ax



# AIC / BIC Model comparison table


def model_comparison_table(

    df,

    weibull,

    lognormal,

    loglogistic

):

    n = len(df)

    rows = []


    models = [

        ("Weibull", weibull, 2),

        ("Log-Normal", lognormal, 2),

        ("Log-Logistic", loglogistic, 2)

    ]


    for name, model, params in models:


        loglik = model.log_likelihood_


        aic = model.AIC_


        bic = (

            -2
            *
            loglik

            +

            params
            *
            np.log(n)

        )


        rows.append({

            "Model": name,

            "Log-Likelihood": loglik,

            "AIC": aic,

            "BIC": bic

        })


    comparison = pd.DataFrame(rows)


    comparison = comparison.sort_values(

        "AIC"

    ).reset_index(

        drop=True

    )


    # Delta AIC

    comparison["Delta AIC"] = (

        comparison["AIC"]

        -

        comparison["AIC"].min()

    )


    # Delta BIC

    comparison["Delta BIC"] = (

        comparison["BIC"]

        -

        comparison["BIC"].min()

    )


    # AIC ranking

    comparison["AIC Rank"] = (

        comparison["AIC"]

        .rank()

        .astype(int)

    )


    return comparison


# Model selection based on AIC and BIC criteria

def select_best_marginal_model(comparison):


    best_aic_model = comparison.loc[

        comparison["AIC"].idxmin(),

        "Model"

    ]


    best_bic_model = comparison.loc[

        comparison["BIC"].idxmin(),

        "Model"

    ]


    if best_aic_model == best_bic_model:


        conclusion = (

            f"{best_aic_model} is selected as the "

            "preferred marginal survival distribution because "

            "it achieves the lowest AIC and BIC."

        )


        selected_model = best_aic_model


    else:


        conclusion = (

            f"AIC selects {best_aic_model}, while BIC selects "

            f"{best_bic_model}. Visual comparison with the "

            "Kaplan–Meier estimator should be used to make the "

            "final selection."

        )


        selected_model = best_aic_model


    return {

        "selected_model": selected_model,

        "best_aic_model": best_aic_model,

        "best_bic_model": best_bic_model,

        "conclusion": conclusion

    }
    
# Improve statistical diagnose 

# Akaike weights

def add_model_selection_metrics(comparison):

    comparison = comparison.copy()

    comparison["Delta AIC"] = (
        comparison["AIC"]
        -
        comparison["AIC"].min()
    )

    comparison["Delta BIC"] = (
        comparison["BIC"]
        -
        comparison["BIC"].min()
    )

    # Relative likelihood
    comparison["Relative Likelihood"] = np.exp(
        -0.5 * comparison["Delta AIC"]
    )

    # Akaike weights
    comparison["Akaike Weight"] = (
        comparison["Relative Likelihood"]
        /
        comparison["Relative Likelihood"].sum()
    )

    comparison["AIC Rank"] = (
        comparison["AIC"]
        .rank(method="min")
        .astype(int)
    )

    comparison["BIC Rank"] = (
        comparison["BIC"]
        .rank(method="min")
        .astype(int)
    )

    return comparison.sort_values(
        "AIC"
    ).reset_index(drop=True)
    

# Kaplan–Meier vs parametric goodness-of-fit metrics
def survival_goodness_of_fit(
    kmf,
    weibull,
    lognormal,
    loglogistic
):

    models = {

        "Weibull": weibull,

        "Log-Normal": lognormal,

        "Log-Logistic": loglogistic

    }


    # Extract Kaplan-Meier survival curve
    km_data = kmf.survival_function_.copy()

    times = km_data.index.values

    km_survival = (
        km_data.iloc[:, 0]
        .values
    )


    results = []


    for name, model in models.items():

        # Predicted survival probabilities
        model_survival = (
            model
            .survival_function_at_times(times)
            .values
        )


        # Difference
        error = (
            km_survival
            -
            model_survival
        )


        # Mean Squared Error
        mse = np.mean(
            error ** 2
        )


        # Root Mean Squared Error
        rmse = np.sqrt(
            mse
        )


       


        results.append({

            "Model": name,

            "MSE_vs_KM": mse,

            "RMSE_vs_KM": rmse

        })


    return (

        pd.DataFrame(results)

        .sort_values(
             "RMSE_vs_KM"
        )

        .reset_index(
            drop=True
        )

    )
    
# goodness-of-fit comparison table
    
def comprehensive_model_diagnostics(

    df,

    kmf,

    weibull,

    lognormal,

    loglogistic

):
    # Information criteria
    comparison = model_comparison_table(

        df,

        weibull,

        lognormal,

        loglogistic

    )


    comparison = add_model_selection_metrics(
        comparison
    )


    # Goodness of fit against KM
    gof = survival_goodness_of_fit(

        kmf,

        weibull,

        lognormal,

        loglogistic

    )


    # Merge
    diagnostics = comparison.merge(

        gof,

        on="Model",

        how="left"

    )


    return diagnostics



# Hazard function diagnostics

def plot_parametric_hazards(

    weibull,

    lognormal,

    loglogistic,

    max_time=120,

    save=True

):

    set_plot_style()

    output_dir = ensure_output_directory()


    fig, ax = plt.subplots(

        figsize=(10, 6)

    )


    times = np.linspace(

        0.1,

        max_time,

        500

    )


    models = {

        "Weibull": weibull,

        "Log-Normal": lognormal,

        "Log-Logistic": loglogistic

    }


    for name, model in models.items():

        hazard = (

            model
            .hazard_at_times(times)
            .values

        )


        ax.plot(

            times,

            hazard,

            linewidth=2,

            label=name

        )


    ax.set_title(

        "Implied Default Hazard Functions",

        fontweight="bold"

    )


    ax.set_xlabel(
        "Loan Age (Months)"
    )


    ax.set_ylabel(
        "Hazard Rate"
    )


    ax.legend(

        frameon=True

    )


    fig.tight_layout()

    
    fig.savefig(

            output_dir
            /
            "parametric_hazard_comparison.pdf",

            bbox_inches="tight"

        )


    plt.show()


    return fig, ax


# Probability-scale residual diagnostic

def plot_probability_scale_residuals(

    df,

    weibull,

    lognormal,

    loglogistic,

    save=True

):

    set_plot_style()

    output_dir = ensure_output_directory()


    models = {

        "Weibull": weibull,

        "Log-Normal": lognormal,

        "Log-Logistic": loglogistic

    }


    fig, ax = plt.subplots(

        figsize=(10, 6)

    )


    durations = (
        df["duration_months"]
        .values
    )


    for name, model in models.items():

        survival_probs = (

            model
            .survival_function_at_times(
                durations
            )
            .values

        )


        # Numerical protection
        survival_probs = np.clip(

            survival_probs,

            1e-10,

            1

        )


        residuals = (

            -np.log(
                survival_probs
            )

        )


        sorted_residuals = np.sort(
            residuals
        )


        empirical_cdf = (

            np.arange(
                1,
                len(sorted_residuals) + 1
            )

            /

            len(sorted_residuals)

        )


        ax.plot(

            sorted_residuals,

            empirical_cdf,

            linewidth=2,

            label=name

        )


    ax.set_title(

        "Probability-Scale Residual Diagnostics",

        fontweight="bold"

    )


    ax.set_xlabel(
        "−log(Ŝ(t))"
    )


    ax.set_ylabel(
        "Empirical Cumulative Probability"
    )


    ax.legend(

        frameon=True

    )


    fig.tight_layout()


    fig.savefig(

            output_dir
            /
            "probability_scale_residuals.pdf",

            bbox_inches="tight"

        )


    plt.show()


    return fig, ax


# Survival fit error diagnostic

def plot_survival_fit_error(

    kmf,

    weibull,

    lognormal,

    loglogistic,

    save=True

):

    # Plotting Style

    set_plot_style()

    output_dir = ensure_output_directory()


    # Figure

    fig, ax = plt.subplots(

        figsize=(10, 6)

    )


    # Models
    
    models = {

        "Weibull": weibull,

        "Log-Normal": lognormal,

        "Log-Logistic": loglogistic

    }


    
    # KAPLAN–MEIER benchmark
    km_curve = (

        kmf
        .survival_function_
        .iloc[:, 0]

    )


    times = km_curve.index.values


    km_survival = km_curve.values


   
    # Fit error plots
    for name, model in models.items():

        fitted_survival = (

            model
            .survival_function_at_times(
                times
            )
            .values

        )


        # Kaplan–Meier minus fitted survival
        error = (

            km_survival

            -

            fitted_survival

        )


        ax.plot(

            times,

            error,

            linewidth=1.8,

            label=name

        )


 
    # Perfect fit reference line
    ax.axhline(

        y=0,

        color="black",

        linestyle="--",

        linewidth=1.2,

        alpha=0.8

    )


    # Titles and Labels
    ax.set_title(

        "Parametric Survival Fit Error Relative to Kaplan–Meier",

        fontweight="bold"

    )


    ax.set_xlabel(

        "Loan Age (Months)"

    )


    ax.set_ylabel(

        r"$\hat{S}_{KM}(t) - \hat{S}_{Model}(t)$"

    )



    ax.legend(

        title="Model",

        frameon=True,

        loc="best"

    )

    fig.tight_layout()


   
    # Save figure
    
    if save:

        fig.savefig(

            output_dir

            /

            "survival_fit_error_vs_kaplan_meier.pdf",

            bbox_inches="tight"

        )

    plt.show()


    return fig, ax