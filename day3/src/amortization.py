import numpy as np
import pandas as pd


def months_between(start_date, end_date):
    """
    Approximate elapsed loan age in months.

    Uses 30.4375 days per month.
    """
    if pd.isna(start_date) or pd.isna(end_date):
        return np.nan

    days = (end_date - start_date).days

    return max(days / 30.4375, 0.0)


def outstanding_balance(
    issue_amount,
    annual_rate,
    term_months,
    amortization_type,
    elapsed_months
):
    """
    Calculate outstanding principal at a given loan age.

    Assumptions:
    - Monthly payments
    - Nominal annual interest rate
    - Monthly rate = annual rate / 12
    - French: constant monthly annuity
    - Italian: equal principal payments
    - Bullet: interest-only with principal due at maturity

    Parameters
    ----------
    issue_amount : float
        Original principal.

    annual_rate : float
        Annual nominal interest rate.

    term_months : int
        Contractual maturity in months.

    amortization_type : str
        French, Italian, or Bullet.

    elapsed_months : float
        Months elapsed since origination.

    Returns
    -------
    float
        Outstanding principal.
    """

    if any(pd.isna(x) for x in [
        issue_amount,
        annual_rate,
        term_months,
        amortization_type,
        elapsed_months
    ]):
        return np.nan

    if issue_amount <= 0 or term_months <= 0:
        return np.nan

    if elapsed_months <= 0:
        return float(issue_amount)

    # Loan is fully mature.
    if elapsed_months >= term_months:
        return 0.0

    # Number of completed monthly payments.
    k = int(np.floor(elapsed_months))

    monthly_rate = annual_rate / 12.0

    # ---------------------------------------------------------
    # French amortisation
    # ---------------------------------------------------------

    if amortization_type == "French":

        if monthly_rate == 0:
            payment = issue_amount / term_months
        else:
            payment = (
                issue_amount
                * monthly_rate
                * (1 + monthly_rate) ** term_months
                / (
                    (1 + monthly_rate) ** term_months - 1
                )
            )

        if k == 0:
            balance = issue_amount
        else:
            if monthly_rate == 0:
                balance = issue_amount - k * payment
            else:
                balance = (
                    issue_amount * (1 + monthly_rate) ** k
                    - payment
                    * (
                        ((1 + monthly_rate) ** k - 1)
                        / monthly_rate
                    )
                )

        return max(float(balance), 0.0)

    # ---------------------------------------------------------
    # Italian / equal-principal amortisation
    # ---------------------------------------------------------

    elif amortization_type == "Italian":

        principal_payment = issue_amount / term_months

        balance = issue_amount - k * principal_payment

        return max(float(balance), 0.0)

    # ---------------------------------------------------------
    # Bullet
    # ---------------------------------------------------------

    elif amortization_type == "Bullet":

        # Interest is paid periodically but principal remains
        # outstanding until maturity.
        return float(issue_amount)

    else:
        raise ValueError(
            f"Unknown amortization type: {amortization_type}"
        )


def calculate_ead_at_date(row, date):
    """
    Calculate outstanding principal at a specified date.
    """

    elapsed_months = months_between(
        row["origination_date"],
        date
    )

    return outstanding_balance(
        issue_amount=row["issue_amount"],
        annual_rate=row["interest_rate"],
        term_months=row["term"],
        amortization_type=row["amortization_type"],
        elapsed_months=elapsed_months
    )


def add_ead_columns(df):
    """
    Add EAD-related columns to a cleaned loan tape.

    Creates:
    - EAD_at_default
    - EAD_at_prepayment
    - balance_at_obs_end
    """

    df = df.copy()

    # ---------------------------------------------------------
    # EAD at default
    # ---------------------------------------------------------

    df["EAD_at_default"] = df.apply(
        lambda row: (
            calculate_ead_at_date(
                row,
                row["default_date"]
            )
            if pd.notna(row["default_date"])
            else np.nan
        ),
        axis=1
    )

    # ---------------------------------------------------------
    # Balance at prepayment
    # ---------------------------------------------------------

    df["balance_at_prepayment"] = df.apply(
        lambda row: (
            calculate_ead_at_date(
                row,
                row["prepayment_date"]
            )
            if pd.notna(row["prepayment_date"])
            else np.nan
        ),
        axis=1
    )

    # ---------------------------------------------------------
    # Balance at observation end
    # ---------------------------------------------------------

    df["balance_at_obs_end"] = df.apply(
        lambda row: calculate_ead_at_date(
            row,
            row["obs_end_date"]
        ),
        axis=1
    )

    return df