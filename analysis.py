#### Import of libraries

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#### Definition of the functions and chart settings

COLUMN_RENAME_MAPS = {
    "estr": {
        "Euro short-term rate (EST.B.EU000A2X2A25.WT)": "€STR Rate",
    },
    "dfr": {
        "Deposit facility - date of changes (raw data) - Level (FM.B.U2.EUR.4F.KR.DFR.LEV)": "Deposit Facility Rate",
    },
    "unsecured": {
        "Unsecured - Wholesale - Borrowing - Average daily turnover - Overnight  (MMSR.B.U2._X._Z.S1ZV._Z.U.BO.AT._X.MA._Z._Z.EUR._Z)": "Unsecured Turnover",
    },
    "secured": {
        "Secured - Wholesale - Borrowing - Average daily turnover - Overnight  (MMSR.B.U2._X.BE.S1ZV._Z.T.BO.AT._X.MA._Z._Z.EUR._Z)": "Secured Turnover",
    },
    "excess_reserves": {
        "Total excess reserves of credit institutions subject to minimum reserve requirements (BSI.M.U2.N.R.LRE.X.1.A1.3000.Z01.E)": "Excess Reserves",
    },
    "deposit_balances": {
        "Deposit facility - Eurosystem (ILM.M.U2.C.L020200.U2.EUR)": "Deposit Facility Balances",
    },
    "excess_liquidity": {
        "Excess liquidity - Eurosystem (ILM.D.U2.C.EXLIQ.U2.EUR)": "Excess Liquidity",
    },
}


def load_data(path: str | Path) -> pd.DataFrame:
    # Input: path to a CSV file.
    # Output: pandas DataFrame containing the raw data.
    # Purpose: load one source dataset from disk.
    """Load a CSV file."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame, type_of_data: str) -> pd.DataFrame:
    # Inputs: raw DataFrame and dataset type.
    # Output: cleaned DataFrame with standard date and value columns.
    # Purpose: standardise the ECB datasets before analysis.
    """Standardise the ECB date and value-column names."""
    if type_of_data not in COLUMN_RENAME_MAPS:
        valid = ", ".join(COLUMN_RENAME_MAPS)
        raise KeyError(f"Unknown dataset type '{type_of_data}'. Valid values: {valid}")
    if "DATE" not in df.columns:
        raise KeyError("The source dataset does not contain a 'DATE' column.")

    cleaned = (
        df.copy()
        .assign(Date=lambda x: pd.to_datetime(x["DATE"], errors="coerce"))
        .drop(columns=["TIME PERIOD", "DATE"], errors="ignore")
        .rename(columns=COLUMN_RENAME_MAPS[type_of_data])
        .dropna(subset=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    expected_columns = list(COLUMN_RENAME_MAPS[type_of_data].values())
    missing = [column for column in expected_columns if column not in cleaned.columns]
    if missing:
        raise KeyError(f"Expected column(s) missing after cleaning: {missing}")
    return cleaned


def convert_column_to_float(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # Inputs: DataFrame and name of the column to convert.
    # Output: DataFrame with a numeric version of the selected column.
    # Purpose: make ECB values ready for calculations.
    """Convert one column to numeric without mutating the input DataFrame."""
    if column not in df.columns:
        raise KeyError(f"Column '{column}' is missing.")
    values = pd.to_numeric(
        df[column].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    return df.assign(**{column: values}).dropna(subset=[column])


def calculate_estr_dfr_spread(
    estr_df: pd.DataFrame,
    dfr_df: pd.DataFrame,
    ) -> pd.DataFrame:
    # Inputs: cleaned €STR and deposit facility rate DataFrames.
    # Output: daily €STR minus DFR spread in basis points.
    # Purpose: measure how far the overnight rate trades below the policy floor.
    """Calculate the daily €STR minus DFR spread in basis points."""
    estr_series = estr_df.set_index("Date")["€STR Rate"].sort_index()
    dfr_series = (
        dfr_df.set_index("Date")["Deposit Facility Rate"]
        .sort_index()
        .reindex(estr_series.index, method="ffill")
    )
    spread = (estr_series - dfr_series) * 100
    return spread.rename("Spread (bps)").dropna().reset_index()


def reconstruct_excess_liquidity_history(
    excess_reserves_df: pd.DataFrame,
    deposit_facility_balance_df: pd.DataFrame,
    excess_liquidity_ecb_df: pd.DataFrame,
    historical_end_year: int = 2024,
) -> pd.DataFrame:
    # Inputs: excess reserves, deposit balances, official excess liquidity and cut-off year.
    # Output: one monthly excess-liquidity history.
    # Purpose: combine the reconstructed historical series with the recent official ECB series.
    """Build a monthly excess-liquidity history and append the recent ECB series."""
    historical = (
        excess_reserves_df[["Excess Reserves"]]
        .join(
            deposit_facility_balance_df[["Deposit Facility Balances"]],
            how="outer",
        )
        .assign(
            **{
                "Excess Liquidity": lambda x: (
                    x["Excess Reserves"] + x["Deposit Facility Balances"]
                ) / 1000
            }
        )
        [["Excess Liquidity"]]
        .dropna()
        .sort_index()
        .resample("ME")
        .mean()
    )
    historical = historical.loc[historical.index.year <= historical_end_year]

    recent = (
        excess_liquidity_ecb_df[["Excess Liquidity"]]
        .sort_index()
        .resample("ME")
        .mean()
        .assign(**{"Excess Liquidity": lambda x: x["Excess Liquidity"] / 1000})
    )
    recent = recent.loc[recent.index.year > historical_end_year]

    return (
        pd.concat([historical, recent])
        .sort_index()
        .loc[lambda x: ~x.index.duplicated(keep="last")]
        .reset_index()
    )


def filter_by_date(
    df: pd.DataFrame,
    start_date: Any,
    end_date: Any,
) -> pd.DataFrame:
    # Inputs: DataFrame, start date and end date.
    # Output: copy of the observations within the selected period.
    # Purpose: apply the dashboard date filters.
    """Filter a DataFrame containing a Date column."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df.loc[df["Date"].between(start, end)].copy()


def get_latest_value(df: pd.DataFrame, column: str) -> float:
    # Inputs: DataFrame and value column.
    # Output: latest available value as a float.
    # Purpose: feed the current-value cards in the dashboard.
    values = df.sort_values("Date")[column].dropna()
    return float(values.iloc[-1]) if not values.empty else float("nan")


def get_value_change(df: pd.DataFrame, column: str) -> float:
    # Inputs: DataFrame and value column.
    # Output: change between the last two available observations.
    # Purpose: show the latest movement in each dashboard metric.
    values = df.sort_values("Date")[column].dropna()
    return float(values.iloc[-1] - values.iloc[-2]) if len(values) >= 2 else 0.0


def prepare_regression_data(
    spread_data: pd.DataFrame,
    excess_liquidity_data: pd.DataFrame,
) -> pd.DataFrame:
    # Inputs: daily spread data and monthly excess-liquidity data.
    # Output: aligned monthly sample with a monetary-policy regime label.
    # Purpose: prepare a consistent dataset for the regime regressions.
    """Align monthly spread and liquidity data and assign monetary-policy regimes."""
    monthly_spread = (
        spread_data.set_index("Date")[["Spread (bps)"]]
        .sort_index()
        .resample("ME")
        .mean()
    )
    monthly_liquidity = (
        excess_liquidity_data.set_index("Date")[["Excess Liquidity"]]
        .sort_index()
        .resample("ME")
        .mean()
    )

    regression_data = (
        monthly_spread.join(monthly_liquidity, how="inner")
        .dropna()
        .reset_index()
    )

    year = regression_data["Date"].dt.year
    regression_data["Regime"] = pd.NA
    regression_data.loc[year.between(2020, 2022), "Regime"] = "Negative rates & tiering"
    regression_data.loc[year >= 2023, "Regime"] = "Normalisation"
    return regression_data.dropna(subset=["Regime"]).reset_index(drop=True)


def run_liquidity_spread_regressions(
    regression_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Input: prepared monthly regression sample.
    # Outputs: model estimates and observation-level predictions.
    # Purpose: estimate the spread and liquidity relationship separately for each regime.
    """Estimate one linear regression per monetary-policy regime."""
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for regime in ["Negative rates & tiering", "Normalisation"]:
        regime_data = regression_data.loc[regression_data["Regime"] == regime].copy()
        if len(regime_data) < 2 or regime_data["Excess Liquidity"].nunique() < 2:
            continue

        x = regime_data[["Excess Liquidity"]]
        y = regime_data["Spread (bps)"]
        model = LinearRegression().fit(x, y)
        regime_data["Predicted Spread (bps)"] = model.predict(x)

        coefficient = float(model.coef_[0])
        intercept = float(model.intercept_)
        results.append(
            {
                "Regime": regime,
                "Coefficient": coefficient,
                "Intercept": intercept,
                "R²": float(r2_score(y, regime_data["Predicted Spread (bps)"])),
                "Observations": int(len(regime_data)),
                "Implied zero-spread liquidity": (
                    -intercept / coefficient if coefficient != 0 else float("nan")
                ),
            }
        )
        predictions.append(regime_data)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return results_df, predictions_df


# ---------------------------------------------------------------------------
# Charts (Plotly)
# ---------------------------------------------------------------------------

COLOR_ESTR = "#1f4e79"
COLOR_DFR = "#9ca3af"
COLOR_SPREAD = "#0ea5e9"
COLOR_LIQUIDITY = "#1f4e79"
COLOR_TIERING = "#f59e0b"
COLOR_NORMALISATION = "#1f4e79"


def _apply_base_layout(figure: go.Figure, title: str, y_title: str) -> go.Figure:
    # Inputs: Plotly figure, chart title and y-axis title.
    # Output: Plotly figure with the common dashboard layout.
    # Purpose: keep the main charts visually consistent.
    figure.update_layout(
        template="plotly_white",
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=16)),
        font=dict(family="Helvetica, Arial, sans-serif", size=13),
        margin=dict(l=10, r=10, t=70, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        yaxis=dict(title=y_title, zeroline=False),
        xaxis=dict(title=None),
        height=430,
    )
    return figure


def create_estr_dfr_rate_chart(
    estr_data: pd.DataFrame,
    dfr_data: pd.DataFrame,
) -> go.Figure:
    # Inputs: €STR and deposit facility rate DataFrames.
    # Output: Plotly chart comparing the two rates.
    # Purpose: show how €STR follows the ECB policy floor over time.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dfr_data["Date"],
            y=dfr_data["Deposit Facility Rate"],
            name="Deposit Facility Rate",
            line=dict(color=COLOR_DFR, width=2, dash="dash", shape="hv"),
            hovertemplate="DFR: %{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=estr_data["Date"],
            y=estr_data["\u20acSTR Rate"],
            name="\u20acSTR",
            line=dict(color=COLOR_ESTR, width=2.2),
            hovertemplate="\u20acSTR: %{y:.3f}%<extra></extra>",
        )
    )
    return _apply_base_layout(
        figure, "\u20acSTR and ECB Deposit Facility Rate", "Rate (%)"
    )


def create_estr_dfr_spread_chart(spread_data: pd.DataFrame) -> go.Figure:
    # Input: DataFrame containing the daily €STR–DFR spread.
    # Output: Plotly chart with daily values and a 30-day average.
    # Purpose: track the evolution and short-term volatility of the spread.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=spread_data["Date"],
            y=spread_data["Spread (bps)"],
            name="Daily spread",
            line=dict(color="rgba(14,165,233,0.45)", width=1.2),
            hovertemplate="Spread: %{y:.2f} bps<extra></extra>",
        )
    )
    moving_average = (
        spread_data.set_index("Date")["Spread (bps)"]
        .rolling(30, min_periods=10)
        .mean()
    )
    figure.add_trace(
        go.Scatter(
            x=moving_average.index,
            y=moving_average.values,
            name="30-day average",
            line=dict(color=COLOR_ESTR, width=2.4),
            hovertemplate="30d avg: %{y:.2f} bps<extra></extra>",
        )
    )
    figure.add_hline(y=0, line_width=1, line_color="#6b7280", opacity=0.7)
    last = spread_data.dropna(subset=["Spread (bps)"]).iloc[-1]
    figure.add_annotation(
        x=last["Date"],
        y=float(last["Spread (bps)"]),
        text=f"{float(last['Spread (bps)']):.1f} bps",
        showarrow=True,
        arrowhead=0,
        ax=40,
        ay=-25,
        font=dict(size=12, color=COLOR_ESTR),
    )
    return _apply_base_layout(figure, "\u20acSTR\u2013DFR Spread", "Basis points")


def create_excess_liquidity_chart(excess_liquidity_data: pd.DataFrame) -> go.Figure:
    # Input: DataFrame containing the excess-liquidity history.
    # Output: Plotly area chart of Eurosystem excess liquidity.
    # Purpose: show the build-up and subsequent decline in excess liquidity.
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=excess_liquidity_data["Date"],
            y=excess_liquidity_data["Excess Liquidity"],
            name="Excess liquidity",
            line=dict(color=COLOR_LIQUIDITY, width=2.2),
            fill="tozeroy",
            fillcolor="rgba(31,78,121,0.10)",
            connectgaps=True,
            hovertemplate="Excess liquidity: \u20ac%{y:,.0f} bn<extra></extra>",
        )
    )
    figure.update_yaxes(rangemode="tozero")
    peak = excess_liquidity_data.loc[
        excess_liquidity_data["Excess Liquidity"].idxmax()
    ]
    figure.add_annotation(
        x=peak["Date"],
        y=float(peak["Excess Liquidity"]),
        text=f"Peak: \u20ac{float(peak['Excess Liquidity']):,.0f} bn",
        showarrow=True,
        arrowhead=0,
        ay=-28,
        font=dict(size=12),
    )
    return _apply_base_layout(
        figure, "Eurosystem Excess Liquidity", "EUR billions"
    )


def create_liquidity_regression_chart(
    predictions_data: pd.DataFrame,
    regression_results: pd.DataFrame,
):
    # Inputs: regression observations with predictions and the model results.
    # Output: Plotly scatter chart with one fitted line per regime.
    # Purpose: compare the reserve-demand relationship across monetary-policy regimes.
    import plotly.express as px
    import plotly.graph_objects as go

    data = predictions_data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year.astype(str)

    years = sorted(data["Year"].unique())

    colors_theme = [
        "#003F5C",
        "#58508D", 
        "#BC5090",  
        "#FF6361",  
        "#FFA600",  
        "#2F4B7C",  
        "#1B998B", 
        "#8C564B",  
    ]

    year_colors = {
        year: colors_theme[index % len(colors_theme)]
        for index, year in enumerate(years)
    }

    figure = px.scatter(
        data,
        x="Excess Liquidity",
        y="Spread (bps)",
        color="Year",
        category_orders={"Year": years},
        color_discrete_map=year_colors,
        hover_data={
            "Date": "|%b %Y",
            "Regime": True,
            "Excess Liquidity": ":,.0f",
            "Spread (bps)": ":.2f",
            "Year": False,
        },
        labels={
            "Excess Liquidity": "Excess liquidity (€bn)",
            "Spread (bps)": "€STR–DFR spread (bps)",
            "Year": "Year",
        },
    )

    regression_styles = {
        "Negative rates & tiering": {
            "color": "#94A3B8",
            "dash": "dot",
        },
        "Normalisation": {
            "color": "#B45309",
            "dash": "solid",
        },
    }

    for regime, regime_data in data.groupby("Regime", observed=True):
        sorted_data = regime_data.sort_values("Excess Liquidity")
        result = regression_results.loc[
            regression_results["Regime"] == regime
        ]

        r_squared = (
            float(result["R²"].iloc[0])
            if not result.empty
            else float("nan")
        )

        style = regression_styles.get(
            regime,
            {"color": "#475569", "dash": "dash"},
        )

        figure.add_trace(
            go.Scatter(
                x=sorted_data["Excess Liquidity"],
                y=sorted_data["Predicted Spread (bps)"],
                mode="lines",
                name=f"{regime} · R²={r_squared:.2f}",
                line={
                    "color": style["color"],
                    "width": 3,
                    "dash": style["dash"],
                },
                hoverinfo="skip",
            )
        )

    figure.update_traces(
        marker={
            "size": 9,
            "opacity": 0.82,
            "line": {
                "width": 0.8,
                "color": "white",
            },
        },
        selector={"mode": "markers"},
    )

    figure.update_layout(
        title={
            "text": "€STR–DFR Spread vs Excess Liquidity",
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        height=650,
        hovermode="closest",
        legend={
            "title": {"text": "Year / Regression"},
            "orientation": "v",
            "x": 1.02,
            "y": 1,
            "bgcolor": "rgba(255,255,255,0)",
        },
        margin={"l": 70, "r": 180, "t": 80, "b": 60},
        font={
            "family": "Arial, sans-serif",
            "size": 13,
            "color": "#1F2937",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    figure.update_xaxes(
        title="Excess liquidity (€bn)",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=False,
        linecolor="#CBD5E1",
    )

    figure.update_yaxes(
        title="€STR–DFR spread (bps)",
        showgrid=True,
        gridcolor="rgba(148,163,184,0.18)",
        zeroline=True,
        zerolinecolor="#94A3B8",
        linecolor="#CBD5E1",
    )

    return figure