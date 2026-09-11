import pandas as pd
import numpy as np


VALID_REGIONS = {
    "NORTHEAST": "Northeast",
    "Northe@st": "Northeast",
    "NORTHE@ST": "Northeast",

    "PACIFIC": "Pacific",
    "P@cific": "Pacific",

    "SOUTHEAST": "Southeast",
    "Southe@st": "Southeast",

    "SOUTHWEST": "Southwest",
    "Southwest_": "Southwest",
}


VALID_AMORTIZATION = {
    "French": "French",
    "Frnch": "French",
    "FRENCH": "French",

    "Italian": "Italian",
    "ITALIAN": "Italian",
    "ITALIAN ": "Italian",

    "Bullet": "Bullet",
    "bullet": "Bullet",
    "BULLET": "Bullet",
}


def clean_loan_tape(input_file, output_file=None):
    """
    Clean and validate the raw loan tape.

    Returns
    -------
    clean_df : pandas.DataFrame
        Cleaned loan-level dataset.

    quality_log : pandas.DataFrame
        Data-quality log documenting detected defects and treatments.
    """

    df = pd.read_csv(input_file)

    original_n = len(df)

    quality_records = []

    def log_issue(name, count, rule):
        quality_records.append({
            "defect": name,
            "count": int(count),
            "rule": rule
        })

    # ---------------------------------------------------------
    # 1. Standardise column names
    # ---------------------------------------------------------

    df.columns = df.columns.str.strip()

    # ---------------------------------------------------------
    # 2. Remove duplicate loan IDs
    # ---------------------------------------------------------

    duplicate_mask = df["loan_id"].duplicated(keep="first")
    duplicate_count = duplicate_mask.sum()

    log_issue(
        "Duplicate loan_id",
        duplicate_count,
        "Keep first occurrence and remove subsequent duplicates"
    )

    df = df.loc[~duplicate_mask].copy()

    # ---------------------------------------------------------
    # 3. Parse dates
    # ---------------------------------------------------------

    date_columns = [
        "origination_date",
        "maturity_date",
        "default_date",
        "prepayment_date",
        "obs_end_date",
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col],
            errors="coerce"
        )

    # ---------------------------------------------------------
    # 4. Region standardisation
    # ---------------------------------------------------------

    df["region_original"] = df["region"]

    df["region"] = (
        df["region"]
        .astype(str)
        .str.strip()
        .replace(VALID_REGIONS)
    )

    # Anything outside the four expected regions is invalid.
    invalid_region_mask = ~df["region"].isin(
        ["Northeast", "Pacific", "Southeast", "Southwest"]
    )

    # Do not silently invent a region.
    invalid_region_count = invalid_region_mask.sum()

    log_issue(
        "Invalid/unrecognised region",
        invalid_region_count,
        "Set to missing; investigate before modelling"
    )

    df.loc[invalid_region_mask, "region"] = np.nan

    # ---------------------------------------------------------
    # 5. Amortisation standardisation
    # ---------------------------------------------------------

    df["amortization_original"] = df["amortization_type"]

    df["amortization_type"] = (
        df["amortization_type"]
        .astype("string")
        .str.strip()
        .replace(VALID_AMORTIZATION)
    )

    xyz_mask = df["amortization_type"].eq("XYZ")

    log_issue(
        "Invalid amortization type (XYZ)",
        xyz_mask.sum(),
        "Remove because EAD cannot be reconstructed"
    )

    missing_amort_mask = df["amortization_type"].isna()

    log_issue(
        "Missing amortization type",
        missing_amort_mask.sum(),
        "Remove because EAD cannot be reconstructed"
    )


    invalid_amort_mask = (
        df["amortization_type"].isna()
        |
        ~df["amortization_type"].isin(
            ["French", "Italian", "Bullet"]
        )
    )
    
    log_issue(
        "Other unrecognised amortization types",
        invalid_amort_mask.sum(),
        "Remove because EAD cannot be reconstructed"
    )
    
    
    # ---------------------------------------------------------
    # 6. FICO validation
    # ---------------------------------------------------------

    invalid_fico = (
        (df["fico_score"] < 300)
        | (df["fico_score"] > 850)
    )

    log_issue(
        "Invalid FICO score",
        invalid_fico.sum(),
        "Set FICO to missing and retain loan"
    )

    df["fico_invalid_flag"] = invalid_fico.astype(int)

    df.loc[invalid_fico, "fico_score"] = np.nan

    # ---------------------------------------------------------
    # 7. Interest-rate validation
    # ---------------------------------------------------------

    invalid_rate = (
        (df["interest_rate"] <= 0)
        | (df["interest_rate"] > 1)
    )

    log_issue(
        "Invalid interest rate",
        invalid_rate.sum(),
        "Set interest rate to missing and flag"
    )

    df["interest_rate_invalid_flag"] = invalid_rate.astype(int)

    df.loc[invalid_rate, "interest_rate"] = np.nan

    # ---------------------------------------------------------
    # 8. Term validation
    # ---------------------------------------------------------

    nonpositive_term = df["term"] <= 0

    log_issue(
        "Non-positive term",
        nonpositive_term.sum(),
        "Remove because contractual maturity cannot be determined"
    )

    excessive_term = df["term"] > 120

    log_issue(
        "Term above 120 months",
        excessive_term.sum(),
        "Remove because it is outside the documented contractual range"
    )

    invalid_term = nonpositive_term | excessive_term

    df["term_invalid_flag"] = invalid_term.astype(int)

    # ---------------------------------------------------------
    # 9. Issue amount validation
    # ---------------------------------------------------------

    invalid_amount = df["issue_amount"] <= 0

    log_issue(
        "Non-positive issue amount",
        invalid_amount.sum(),
        "Remove because exposure cannot be defined"
    )

    df["issue_amount_invalid_flag"] = invalid_amount.astype(int)
    
    
    # ---------------------------------------------------------
    # 9b. Implausible issue amount (systematic unit/decimal entry error)
    # ---------------------------------------------------------

    EXTREME_AMOUNT_THRESHOLD = 12_560  # empirical break point, see Part 1 diagnostic

    implausible_amount_mask = (
        df["issue_amount"] > EXTREME_AMOUNT_THRESHOLD
    )

    log_issue(
        f"Implausible issue amount (> eur{EXTREME_AMOUNT_THRESHOLD:,.0f} — "
        "systematic unit/decimal entry defect, not isolated outliers)",
        implausible_amount_mask.sum(),
        f"Flag and cap at €{EXTREME_AMOUNT_THRESHOLD:,.0f} (winsorize); loan retained"
    )

    df["issue_amount_original"] = df["issue_amount"]
    df["issue_amount_extreme_flag"] = implausible_amount_mask.astype(int)

    df.loc[implausible_amount_mask, "issue_amount"] = EXTREME_AMOUNT_THRESHOLD

    # ---------------------------------------------------------
    # 10. Origination-date validation
    # ---------------------------------------------------------

    missing_origination = df["origination_date"].isna()

    log_issue(
        "Missing/unparseable origination date",
        missing_origination.sum(),
        "Remove because loan age and maturity cannot be determined"
    )

    # ---------------------------------------------------------
    # 11. Observation-date validation
    # ---------------------------------------------------------

    invalid_obs = (
        df["obs_end_date"].isna()
        | (
            df["obs_end_date"]
            < df["origination_date"]
        )
    )

    log_issue(
        "Invalid observation end date",
        invalid_obs.sum(),
        "Remove because observation window is undefined"
    )

    # ---------------------------------------------------------
    # 12. Event dates before origination
    # ---------------------------------------------------------

    invalid_default = (
        df["default_date"].notna()
        & (
            df["default_date"]
            < df["origination_date"]
        )
    )

    log_issue(
        "Default date before origination",
        invalid_default.sum(),
        "Set default date to missing and flag as invalid event"
    )

    df["default_date_invalid_flag"] = invalid_default.astype(int)

    df.loc[invalid_default, "default_date"] = pd.NaT

    invalid_prepay = (
        df["prepayment_date"].notna()
        & (
            df["prepayment_date"]
            < df["origination_date"]
        )
    )

    log_issue(
        "Prepayment date before origination",
        invalid_prepay.sum(),
        "Set prepayment date to missing and flag as invalid event"
    )

    df["prepayment_date_invalid_flag"] = invalid_prepay.astype(int)

    df.loc[invalid_prepay, "prepayment_date"] = pd.NaT

    # ---------------------------------------------------------
    # 13. Events after observation end
    # ---------------------------------------------------------

    default_after_obs = (
        df["default_date"].notna()
        & (
            df["default_date"]
            > df["obs_end_date"]
        )
    )

    log_issue(
        "Default after observation end",
        default_after_obs.sum(),
        "Set event to missing and flag; treat as administratively censored"
    )

    df["default_after_obs_flag"] = default_after_obs.astype(int)

    df.loc[default_after_obs, "default_date"] = pd.NaT

    prepay_after_obs = (
        df["prepayment_date"].notna()
        & (
            df["prepayment_date"]
            > df["obs_end_date"]
        )
    )

    log_issue(
        "Prepayment after observation end",
        prepay_after_obs.sum(),
        "Set event to missing and flag; treat as administratively censored"
    )

    df["prepayment_after_obs_flag"] = prepay_after_obs.astype(int)

    df.loc[prepay_after_obs, "prepayment_date"] = pd.NaT

    # ---------------------------------------------------------
    # 14. Resolve competing events
    # ---------------------------------------------------------

    both_events = (
        df["default_date"].notna()
        & df["prepayment_date"].notna()
    )

    default_first = (
        both_events
        & (df["default_date"] < df["prepayment_date"])
    )

    prepay_first = (
        both_events
        & (df["prepayment_date"] < df["default_date"])
    )

    same_day = (
        both_events
        & (df["default_date"] == df["prepayment_date"])
    )

    log_issue(
        "Both default and prepayment recorded",
        both_events.sum(),
        "Keep the first observed termination event"
    )

    log_issue(
        "Default before prepayment",
        default_first.sum(),
        "Keep default and remove later prepayment"
    )

    log_issue(
        "Prepayment before default",
        prepay_first.sum(),
        "Keep prepayment and remove later default"
    )

    log_issue(
        "Default and prepayment on same date",
        same_day.sum(),
        "Flag as ambiguous competing event"
    )

    df["competing_event_flag"] = both_events.astype(int)

    # If default happened first, remove prepayment.
    df.loc[default_first, "prepayment_date"] = pd.NaT

    # If prepayment happened first, remove default.
    df.loc[prepay_first, "default_date"] = pd.NaT

    # Same-day events are ambiguous.
    # Do not silently assign one cause.
    df["same_day_event_flag"] = same_day.astype(int)

    df.loc[same_day, "default_date"] = pd.NaT
    df.loc[same_day, "prepayment_date"] = pd.NaT

    # ---------------------------------------------------------
    # 15. Reconstruct maturity
    # ---------------------------------------------------------

    df["maturity_original"] = df["maturity_date"]

    valid = (
        df["origination_date"].notna() &
        df["term"].notna()
    )

    df["expected_maturity"] = pd.NaT

    df.loc[valid, "expected_maturity"] = [
        d + pd.DateOffset(months=int(t))
        for d, t in zip(
            df.loc[valid, "origination_date"],
            df.loc[valid, "term"]
        )
    ]
    
    maturity_errors = (
        df["maturity_original"].notna() &
        df["expected_maturity"].notna() &
        (
            df["maturity_original"]
            != df["expected_maturity"]
        )
    )
    
    log_issue(
        "Maturity inconsistencies",
        maturity_errors.sum(),
        "Replace with contractual maturity reconstructed from origination date and term"
    )
    
    df["maturity_date"] = df["expected_maturity"]
    
    # ---------------------------------------------------------
    # 16. Remove structurally unusable loans
    # ---------------------------------------------------------

    remove_mask = (
        missing_origination
        | invalid_obs
        | invalid_term
        | invalid_amount
        | invalid_amort_mask
    )

    removed_count = remove_mask.sum()

    log_issue(
        "Loans removed for structurally unusable fields",
        removed_count,
        "Remove records for which loan age, contractual schedule, or EAD cannot be reliably reconstructed"
    )

    df_clean = df.loc[~remove_mask].copy()

    # ---------------------------------------------------------
    # 17. Create event indicators
    # ---------------------------------------------------------

    df_clean["default_event"] = (
        df_clean["default_date"].notna()
    ).astype(int)

    df_clean["prepayment_event"] = (
        df_clean["prepayment_date"].notna()
    ).astype(int)

    # ---------------------------------------------------------
    # 18. Define observed termination
    # ---------------------------------------------------------

    df_clean["event_date"] = pd.NaT
    df_clean["event_type"] = "censored"

    default_mask = df_clean["default_date"].notna()
    prepay_mask = df_clean["prepayment_date"].notna()

    df_clean.loc[default_mask, "event_date"] = (
        df_clean.loc[default_mask, "default_date"]
    )

    df_clean.loc[default_mask, "event_type"] = "default"

    df_clean.loc[prepay_mask, "event_date"] = (
        df_clean.loc[prepay_mask, "prepayment_date"]
    )

    df_clean.loc[prepay_mask, "event_type"] = "prepayment"

    # ---------------------------------------------------------
    # 19. Loan age at event/censoring
    # ---------------------------------------------------------
    # A censored loan can only be at risk of default until its own contractual
    # maturity or the observation cutoff, whichever comes first — after
    # maturity, an undefaulted loan has been repaid in full and is no longer
    # at risk.
    df_clean["censoring_date"] = df_clean[
        ["obs_end_date", "maturity_date"]
    ].min(axis=1)

    df_clean["analysis_end_date"] = (
        df_clean["event_date"]
        .fillna(df_clean["censoring_date"])
    )
    df_clean["loan_age_days"] = (
        df_clean["analysis_end_date"]
        - df_clean["origination_date"]
    ).dt.days

    df_clean["loan_age_months"] = (
        df_clean["loan_age_days"] / 30.4375
    )

    # ---------------------------------------------------------
    # 20. Quality summary
    # ---------------------------------------------------------

    quality_log = pd.DataFrame(quality_records)

    quality_log["pct_original_rows"] = (
        quality_log["count"] / original_n * 100
    )

    if output_file is not None:
        df_clean.to_csv(output_file, index=False)

    return df_clean, quality_log

