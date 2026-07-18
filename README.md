# Euro Money Market Monitor

Link to the dashboard : https://euro-money-market-monitor-h5yzaw4koipwcv3thmjqua.streamlit.app/

## Overview

This project is an interactive dashboard built with **Python**,
**Streamlit** and **Plotly** to monitor the evolution of the euro money
market.

The dashboard focuses on the relationship between **Eurosystem excess
liquidity** and the **€STR--Deposit Facility Rate spread**, with a
particular emphasis on the transition from an abundant-reserves
environment towards a scarcer-reserves regime.

The project was developed using publicly available data from the **ECB
Data Portal**.

------------------------------------------------------------------------

## Main features

-   Interactive overview of €STR and the ECB Deposit Facility Rate
-   Daily €STR--DFR spread monitoring
-   Historical reconstruction of Eurosystem excess liquidity
-   Reserve demand curve analysis across different monetary policy
    regimes
-   Downloadable cleaned datasets
-   Methodology and data-source documentation

------------------------------------------------------------------------

## Project structure

``` text
.
├── app.py                  # Streamlit application
├── analysis.py             # Data processing and chart functions
├── data/
│   ├── raw/
│   │   ├── market_rates/
│   │   └── excess_liquidity/
│   └── ...
└── README.md
```

------------------------------------------------------------------------

## Data

The dashboard uses the following ECB datasets:

-   €STR
-   Deposit Facility Rate
-   Excess Liquidity
-   Excess Reserves
-   Deposit Facility Balances

Historical excess liquidity is reconstructed from excess reserves and
deposit facility balances until the official ECB series becomes
available. The two series are then combined into a single monthly
history.

------------------------------------------------------------------------

## Methodology

The dashboard computes the daily **€STR--DFR spread** in basis points
and aligns it with monthly excess liquidity observations.

Two monetary policy regimes are analysed separately:

-   Negative rates and tiering (2020--2022)
-   Normalisation (2023 onwards)

A simple linear regression is estimated for each regime to study how the
spread evolves as excess liquidity declines.

The objective is not to predict the exact level at which reserves become
scarce, but to monitor whether the historical relationship begins to
change as liquidity is gradually withdrawn.

------------------------------------------------------------------------

## Technologies

-   Python
-   pandas
-   Plotly
-   Streamlit
-   scikit-learn

------------------------------------------------------------------------

## Running the dashboard

Install the required packages and start the application:

``` bash
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## Data source

All data used in this project comes from the **ECB Data Portal**.

https://data.ecb.europa.eu/

------------------------------------------------------------------------

## Author

**Hedi SAGAR**
