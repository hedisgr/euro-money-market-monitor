#### Import of libraries

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis import (
    calculate_estr_dfr_spread,
    clean_dataset,
    convert_column_to_float,
    create_estr_dfr_rate_chart,
    create_estr_dfr_spread_chart,
    create_excess_liquidity_chart,
    create_liquidity_regression_chart,
    filter_by_date,
    get_latest_value,
    get_value_change,
    load_data,
    prepare_regression_data,
    reconstruct_excess_liquidity_history,
    run_liquidity_spread_regressions,
)

#### Dashboard application

# Set the browser title and use the full page width.
st.set_page_config(
    page_title="Euro Money Market Monitor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply the custom visual theme.
st.markdown(
    """
    <style>
        :root {
            --navy-950: #061426;
            --navy-900: #0a1d35;
            --blue-600: #2563eb;
            --blue-500: #3b82f6;
            --slate-950: #0f172a;
            --slate-700: #334155;
            --slate-500: #64748b;
            --slate-300: #cbd5e1;
            --slate-200: #e2e8f0;
            --page: #f6f8fc;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(59,130,246,.06), transparent 28rem),
                var(--page);
            color: var(--slate-950);
        }

        .block-container {
            max-width: 1480px;
            padding: 3.25rem 2rem 2.2rem;
        }

        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 20% 0%, rgba(37,99,235,.22), transparent 18rem),
                linear-gradient(180deg, var(--navy-950) 0%, #071a2f 100%);
            border-right: 1px solid rgba(148,163,184,.18);
        }

        section[data-testid="stSidebar"] * {color: #e8eef8;}
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        section[data-testid="stSidebar"] small {color: #9fb0c8 !important;}

        .brand-card {
            display: flex;
            align-items: center;
            gap: .8rem;
            padding: .2rem .25rem 1.1rem;
            margin-bottom: .7rem;
            border-bottom: 1px solid rgba(148,163,184,.16);
        }

        .brand-icon {
            display: grid;
            place-items: center;
            width: 2.3rem;
            height: 2.3rem;
            border-radius: .7rem;
            color: white;
            font-size: 1.35rem;
            font-weight: 800;
            background: linear-gradient(145deg, #3b82f6, #1d4ed8);
            box-shadow: 0 8px 24px rgba(37,99,235,.35);
        }

        .brand-title {
            margin: 0;
            color: white;
            font-size: 1.02rem;
            font-weight: 750;
            line-height: 1.15;
        }

        .brand-subtitle {
            margin: .18rem 0 0;
            color: #9fb0c8;
            font-size: .79rem;
        }

        div[role="radiogroup"] {gap: .25rem;}

        /* Hide Streamlit's native radio circle */
        div[role="radiogroup"] > label > div:first-child {
            display: none !important;
        }

        div[role="radiogroup"] input[type="radio"] {
            display: none !important;
        }
        div[role="radiogroup"] > label {
            padding: .62rem .7rem;
            margin: 0;
            border-radius: .6rem;
            border: 1px solid transparent;
            transition: all .15s ease;
        }
        div[role="radiogroup"] > label:hover {
            background: rgba(255,255,255,.07);
            border-color: rgba(148,163,184,.16);
        }
        div[role="radiogroup"] > label:has(input:checked) {
            background: linear-gradient(90deg, #183b63, #245b8f);
            border-color: rgba(255,255,255,.14);
            box-shadow: inset 3px 0 0 #7dd3fc, 0 7px 18px rgba(2,12,27,.24);
        }
        div[role="radiogroup"] > label:has(input:checked) p {
            color: white !important;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #f8fafc !important;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #cbd5e1 !important;
        }

        /* Date inputs */
        section[data-testid="stSidebar"] [data-baseweb="input"] > div {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="input"] input {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            font-weight: 650 !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="input"] input::placeholder {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
            opacity: 1 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="input"] svg {
            color: #475569 !important;
            fill: #475569 !important;
        }

        /* Multiselect */
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
            color: #0f172a !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] input {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="tag"] {
            background: #dbeafe !important;
            border: 1px solid #93c5fd !important;
            border-radius: 7px !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="tag"] span {
            color: #16324f !important;
            font-weight: 700 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="tag"] svg,
        section[data-testid="stSidebar"] [data-baseweb="select"] svg {
            color: #334155 !important;
            fill: #334155 !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        .page-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 1.5rem;
            margin-bottom: 1.25rem;
        }

        .page-eyebrow {
            color: var(--blue-600);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: .35rem;
        }

        .page-title {
            color: var(--slate-950);
            font-size: clamp(1.8rem, 2.4vw, 2.65rem);
            line-height: 1.08;
            letter-spacing: -.035em;
            margin: 0;
            font-weight: 800;
        }

        .page-subtitle {
            color: var(--slate-500);
            margin: .55rem 0 0;
            font-size: .98rem;
        }

        .update-card {
            min-width: 180px;
            margin-top: .55rem;
            padding: .9rem 1.05rem;
            border-radius: .8rem;
            background: rgba(255,255,255,.88);
            border: 1px solid var(--slate-200);
            box-shadow: 0 6px 24px rgba(15,23,42,.045);
        }

        .update-label {color: var(--slate-500); font-size: .72rem;}
        .update-value {color: var(--slate-950); font-size: .93rem; font-weight: 750;}

        [data-testid="stMetric"] {
            min-height: 132px;
            padding: 1rem 1.05rem;
            border-radius: .85rem;
            background: rgba(255,255,255,.94);
            border: 1px solid var(--slate-200);
            box-shadow: 0 7px 26px rgba(15,23,42,.045);
        }
        [data-testid="stMetricLabel"] {color: var(--slate-700); font-weight: 700;}
        [data-testid="stMetricValue"] {color: #16324f; font-weight: 800; letter-spacing: -.025em;}

        [data-testid="stPlotlyChart"] {
            padding: .45rem;
            border-radius: .9rem;
            background: rgba(255,255,255,.96);
            border: 1px solid var(--slate-200);
            box-shadow: 0 7px 25px rgba(15,23,42,.045);
        }

        .section-title {
            color: var(--slate-950);
            font-size: 1.12rem;
            font-weight: 780;
            margin: .25rem 0 .12rem;
        }
        .section-note {
            color: var(--slate-500);
            font-size: .9rem;
            line-height: 1.55;
            margin: .35rem .15rem 1rem;
        }
        .insight-card {
            padding: 1rem 1.05rem;
            margin-top: .8rem;
            border-radius: .85rem;
            border: 1px solid #bfdbfe;
            background: linear-gradient(135deg, #eff6ff, #ffffff);
            color: #1e3a5f;
            line-height: 1.55;
        }
        .footer-note {
            color: #94a3b8;
            font-size: .82rem;
            text-align: center;
            border-top: 1px solid var(--slate-200);
            padding-top: 1rem;
            margin-top: 1.6rem;
        }

        @media (max-width: 900px) {
            .page-header {flex-direction: column;}
            .update-card {width: 100%;}
            .block-container {padding-left: 1rem; padding-right: 1rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define the local paths used to load the ECB datasets.
BASE_DIR = Path(__file__).resolve().parent
ESTR_PATH = BASE_DIR / "Data/raw/market_rates/estr.csv"
DFR_PATH = BASE_DIR / "Data/raw/market_rates/dfr.csv"

EXCESS_LIQUIDITY_PATH = BASE_DIR / "Data/raw/excess_liquidity/official_excess_liquidity.csv"
EXCESS_WITHOUT_DEPOSIT_PATH = BASE_DIR / "Data/raw/excess_liquidity/excess_reserves.csv"
INSTITUTIONS_DEPOSIT_PATH = BASE_DIR / "Data/raw/excess_liquidity/deposit_facility_balances.csv"

@st.cache_data(show_spinner="Loading and preparing ECB data…")
def prepare_data():
    # Input: source file paths defined above.
    # Output: cleaned datasets, regression sample, model results and predictions.
    # Purpose: prepare all data used by the dashboard in one cached step.
    estr_data = (
        load_data(ESTR_PATH)
        .pipe(clean_dataset, "estr")
        .pipe(convert_column_to_float, "€STR Rate")
        .sort_values("Date")
    )
    dfr_data = (
        load_data(DFR_PATH)
        .pipe(clean_dataset, "dfr")
        .pipe(convert_column_to_float, "Deposit Facility Rate")
        .sort_values("Date")
    )
    excess_reserves = (
        load_data(EXCESS_WITHOUT_DEPOSIT_PATH)
        .pipe(clean_dataset, "excess_reserves")
        .pipe(convert_column_to_float, "Excess Reserves")
        .set_index("Date")
        .sort_index()
    )
    deposit_balances = (
        load_data(INSTITUTIONS_DEPOSIT_PATH)
        .pipe(clean_dataset, "deposit_balances")
        .pipe(convert_column_to_float, "Deposit Facility Balances")
        .set_index("Date")
        .sort_index()
    )
    recent_excess_liquidity = (
        load_data(EXCESS_LIQUIDITY_PATH)
        .pipe(clean_dataset, "excess_liquidity")
        .pipe(convert_column_to_float, "Excess Liquidity")
        .set_index("Date")
        .sort_index()
    )

    spread_data = calculate_estr_dfr_spread(estr_data, dfr_data)
    excess_liquidity_data = reconstruct_excess_liquidity_history(
        excess_reserves,
        deposit_balances,
        recent_excess_liquidity,
        historical_end_year=2024,
    )
    regression_data = prepare_regression_data(spread_data, excess_liquidity_data)
    regression_results, regression_predictions = run_liquidity_spread_regressions(
        regression_data
    )
    return (
        estr_data,
        dfr_data,
        spread_data,
        excess_liquidity_data,
        regression_data,
        regression_results,
        regression_predictions,
    )


# Prepare the data and stop the application if a source file is invalid.
try:
    (
        estr_data,
        dfr_data,
        spread_data,
        excess_liquidity_data,
        regression_data,
        regression_results,
        regression_predictions,
    ) = prepare_data()
except (FileNotFoundError, KeyError, ValueError) as error:
    st.error(f"Data preparation error: {error}")
    st.stop()
except Exception as error:  # pragma: no cover
    st.exception(error)
    st.stop()

# Keep only observation dates that are not in the future.
today = pd.Timestamp.today().normalize()

valid_observation_dates = pd.concat(
    [
        estr_data.loc[estr_data["Date"] <= today, "Date"],
        dfr_data.loc[dfr_data["Date"] <= today, "Date"],
        excess_liquidity_data.loc[
            excess_liquidity_data["Date"] <= today,
            "Date",
        ],
    ],
    ignore_index=True,
).dropna()

last_updated = valid_observation_dates.max()

# Build the navigation menu and user filters.
with st.sidebar:
    st.markdown(
        """
        <div class="brand-card">
            <div class="brand-icon">€</div>
            <div>
                <p class="brand-title">Euro Money Market</p>
                <p class="brand-subtitle">Liquidity monitor</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Overview", "Reserve Demand Curve", "Data", "Methodology"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Filters")
    st.caption("Rates and liquidity use separate date ranges.")

    rates_min = max(
        estr_data["Date"].min(), dfr_data["Date"].min(), spread_data["Date"].min()
    ).date()
    rates_max = min(
        estr_data["Date"].max(), dfr_data["Date"].max(), spread_data["Date"].max()
    ).date()
    rates_dates = st.date_input(
        "Rates and spread period",
        value=(rates_min, rates_max),
        min_value=rates_min,
        max_value=rates_max,
    )
    rates_start, rates_end = (
        rates_dates if len(rates_dates) == 2 else (rates_min, rates_max)
    )

    liquidity_min = excess_liquidity_data["Date"].min().date()
    liquidity_max = excess_liquidity_data["Date"].max().date()
    liquidity_dates = st.date_input(
        "Excess-liquidity period",
        value=(liquidity_min, liquidity_max),
        min_value=liquidity_min,
        max_value=liquidity_max,
    )
    liquidity_start, liquidity_end = (
        liquidity_dates
        if len(liquidity_dates) == 2
        else (liquidity_min, liquidity_max)
    )

    selected_regimes = st.multiselect(
        "Regression regimes",
        options=["Negative rates & tiering", "Normalisation"],
        default=["Negative rates & tiering", "Normalisation"],
    )

    st.divider()
    st.caption(f"ECB Data Portal · Updated {last_updated:%d %b %Y}")
    st.caption("Hedi SAGAR")

# Store the title and description shown for each page.
page_descriptions = {
    "Overview": (
        "Market overview",
        "This dashboard tracks Eurosystem excess liquidity using the €STR–DFR spread, its evolution over time, and the dynamics of excess liquidity.",
    ),
    "Reserve Demand Curve": (
        "Reserve demand curve",
        "Explore the relationship between excess liquidity and the €STR–DFR spread across monetary-policy regimes.",
    ),
    "Data": (
        "Prepared datasets",
        "Below are the cleaned and processed datasets used to generate the charts in this dashboard.",
    ),
    "Methodology": (
        "Methodology and sources",
        "Below you will find the methodology used in this dashboard, along with the data sources and references.",
    ),
}

page_eyebrow, page_subtitle = page_descriptions[page]

# Display the page header and latest observation date.
st.markdown(
    f"""
    <div class="page-header">
        <div>
            <div class="page-eyebrow">{page_eyebrow}</div>
            <h1 class="page-title">Euro Money Market Monitor</h1>
            <p class="page-subtitle">{page_subtitle}</p>
        </div>
        <div class="update-card">
            <div class="update-label">Last observation</div>
            <div class="update-value">{last_updated:%d %B %Y}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Apply the selected filters before calculating metrics and drawing charts.
estr_filtered = filter_by_date(estr_data, rates_start, rates_end)
dfr_filtered = filter_by_date(dfr_data, rates_start, rates_end)
spread_filtered = filter_by_date(spread_data, rates_start, rates_end)
liquidity_filtered = filter_by_date(
    excess_liquidity_data, liquidity_start, liquidity_end
)
regression_predictions_filtered = (
    regression_predictions.loc[
        regression_predictions["Regime"].isin(selected_regimes)
    ].copy()
    if not regression_predictions.empty
    else regression_predictions
)
regression_results_filtered = (
    regression_results.loc[regression_results["Regime"].isin(selected_regimes)].copy()
    if not regression_results.empty
    else regression_results
)

# Do not continue when the selected rate period contains no observations.
if estr_filtered.empty or dfr_filtered.empty or spread_filtered.empty:
    st.warning("No rate data is available for the selected period.")
    st.stop()

# Calculate the latest levels displayed in the four summary cards.
latest_estr = get_latest_value(estr_filtered, "€STR Rate")
latest_dfr = get_latest_value(dfr_filtered, "Deposit Facility Rate")
latest_spread = get_latest_value(spread_filtered, "Spread (bps)")
latest_liquidity = (
    get_latest_value(liquidity_filtered, "Excess Liquidity")
    if not liquidity_filtered.empty
    else float("nan")
)

# Display the main market indicators.
metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric(
    "Latest €STR",
    f"{latest_estr:.3f}%",
    f"{get_value_change(estr_filtered, '€STR Rate'):.3f} pp",
)
metric_2.metric(
    "Deposit Facility Rate",
    f"{latest_dfr:.2f}%",
    f"{get_value_change(dfr_filtered, 'Deposit Facility Rate'):.2f} pp",
)
metric_3.metric(
    "€STR–DFR Spread",
    f"{latest_spread:.2f} bps",
    f"{get_value_change(spread_filtered, 'Spread (bps)'):.2f} bps",
)
metric_4.metric(
    "Excess Liquidity",
    f"€{latest_liquidity:,.0f} bn" if pd.notna(latest_liquidity) else "No data",
    f"€{get_value_change(liquidity_filtered, 'Excess Liquidity'):,.0f} bn"
    if not liquidity_filtered.empty
    else None,
)

# Overview page with the main time-series charts.
if page == "Overview":
    st.markdown('<div class="section-title">Policy rates and €STR</div>', unsafe_allow_html=True)
    st.plotly_chart(
        create_estr_dfr_rate_chart(estr_filtered, dfr_filtered),
        use_container_width=True,
    )
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">€STR–DFR spread</div>', unsafe_allow_html=True)
        st.plotly_chart(
            create_estr_dfr_spread_chart(spread_filtered),
            use_container_width=True,
        )
        st.markdown(
            '<div class="section-note">Negative spread = €STR trades below the '
            "floor, driven by lenders without access to the deposit facility "
            "(money market funds). Downward spikes cluster on quarter-ends "
            "(regulatory balance-sheet snapshots).</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section-title">Eurosystem excess liquidity</div>', unsafe_allow_html=True)
        if liquidity_filtered.empty:
            st.info("No excess-liquidity data for this period.")
        else:
            st.plotly_chart(
                create_excess_liquidity_chart(liquidity_filtered),
                use_container_width=True,
            )
            st.markdown(
                '<div class="section-note">Reconstructed from Eurosystem '
                "accounts (excess reserves + deposit facility) until 2024, "
                "official ECB series afterwards. Near zero before 2008: the "
                "corridor era had no structural excess.</div>",
                unsafe_allow_html=True,
            )

# Reserve demand page with regime regressions and model estimates.
elif page == "Reserve Demand Curve":
    st.markdown(
        '<div class="section-note">Monthly observations. Two regimes estimated '
        "separately: the negative-rates/tiering era (2020–2022, shown muted "
        "distorted by the two-tier remuneration system) and the normalisation "
        "regime (2023 onwards). 2019 is excluded (€STR launch and tiering "
        "introduction).</div>",
        unsafe_allow_html=True,
    )

    if regression_predictions_filtered.empty:
        st.warning("Not enough common observations for the selected regimes.")
    else:
        st.plotly_chart(
            create_liquidity_regression_chart(
                regression_predictions_filtered, regression_results_filtered
            ),
            use_container_width=True,
        )

        norm_row = regression_results.loc[
            regression_results["Regime"] == "Normalisation"
        ]
        if not norm_row.empty:
            slope_bp_per_trillion = float(norm_row["Coefficient"].iloc[0]) * 1000
            implied_zero = float(norm_row["Implied zero-spread liquidity"].iloc[0])
            r_squared = float(norm_row["R²"].iloc[0])
            st.success(
            f"**Key finding**: During the normalisation regime "
            f"(R² = {r_squared:.2f}), the €STR–DFR spread narrows by "
            f"about {abs(slope_bp_per_trillion):.1f} bps for every €1,000 bn "
            f"decline in excess liquidity. A simple linear extrapolation implies "
            f"that the spread would reach zero only at an excess liquidity level "
            f"of €{implied_zero:,.0f} bn, which is not economically feasible. "
            f"This suggests that the current linear relationship cannot persist "
            f"indefinitely. At lower levels of excess liquidity, the spread is "
            f"expected to adjust more rapidly as the Eurosystem approaches the "
            f"steeper part of the reserve demand curve. The purpose of this "
            f"monitor is to detect when that transition begins."
            )

        st.subheader("Model estimates")
        st.dataframe(
            regression_results_filtered.style.format(
                {
                    "Coefficient": "{:.5f}",
                    "Intercept": "{:.3f}",
                    "R²": "{:.3f}",
                    "Observations": "{:.0f}",
                    "Implied zero-spread liquidity": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "Historical relationships provide guidance, not a fixed threshold. "
        "Changes in the ECB's operational framework and regulatory environment "
        "may shift the reserve demand curve over time."
    )
# Data page for reviewing and downloading the prepared datasets.
elif page == "Data":
    st.subheader("Prepared datasets")
    dataset_name = st.selectbox(
        "Dataset",
        ["€STR", "Deposit Facility Rate", "Spread", "Excess Liquidity", "Regression sample"],
    )
    datasets = {
        "€STR": estr_filtered,
        "Deposit Facility Rate": dfr_filtered,
        "Spread": spread_filtered,
        "Excess Liquidity": liquidity_filtered,
        "Regression sample": regression_data,
    }
    selected_data = datasets[dataset_name]

    centered_table = (
        selected_data.style
        .set_properties(**{"text-align": "center"})
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "center"),
                        ("vertical-align", "middle"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "center"),
                        ("vertical-align", "middle"),
                    ],
                },
            ]
        )
    )

    st.dataframe(
        centered_table,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download selected data as CSV",
        data=selected_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{dataset_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# Methodology page with definitions and ECB source identifiers.
elif page == "Methodology":
    st.subheader("Methodology")
    st.markdown(
        """
        **€STR–DFR spread** : daily spread computed as
        `(€STR − Deposit Facility Rate) × 100`, in basis points. The DFR (an
        administered rate, constant between Governing Council decisions) is
        forward-filled onto €STR publication dates.

        **Excess liquidity** : until 2024, reconstructed as *excess reserves +
        deposit facility balances* (EUR billions, month-end averages) from
        Eurosystem accounts; from 2025, the official ECB daily series
        (monthly averages). The reconstruction was reconciled against the
        official series on their overlap.

        **Regressions** : monthly spread and liquidity observations aligned on
        common dates; one OLS per monetary-policy regime. 2019 is excluded
        (€STR launch in October, tiering introduction the same month).
        """
    )
    st.subheader("Data sources (ECB Data Portal)")
    st.table(
        pd.DataFrame(
            {
                "Series": [
                    "€STR (rate)",
                    "Deposit facility rate",
                    "Excess liquidity (official, daily)",
                    "Excess reserves",
                    "Deposit facility balances",
                ],
                "Key": [
                    "EST.B.EU000A2X2A25.WT",
                    "FM.B.U2.EUR.4F.KR.DFR.LEV",
                    "ILM.D.U2.C.EXLIQ.U2.EUR",
                    "BSI.M.U2.N.R.LRE.X.1.A1.3000.Z01.E",
                    "ILM.M.U2.C.L020200.U2.EUR",
                ],
            }
        )
    )

# Add the project footer.
st.markdown(
    '<div class="footer-note">Euro Money Market Monitor · Hedi Sagar · '
    "Data: ECB Data Portal (public) · Built with Python, pandas, Plotly and "
    "Streamlit</div>",
    unsafe_allow_html=True,
)
