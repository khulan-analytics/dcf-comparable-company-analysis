
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MRK Equity Valuation Lab",
    page_icon="📈",
    layout="wide"
)

st.title("Merck & Co. (MRK) Equity Valuation Lab")

st.caption(
    "Created by Khulan Ganzorig | "
    "MS in Business Analytics, University at Buffalo"
)

st.markdown("""
An equity research project covering historical financial analysis,
free cash flow valuation, WACC, comparable company analysis,
sensitivity analysis, and scenario modeling.
""")

st.divider()


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def number(value, default=np.nan):
    try:
        if value is None:
            return default

        result = float(value)

        return result if np.isfinite(result) else default

    except (TypeError, ValueError, OverflowError):
        return default


def first_valid(*values):
    for value in values:
        value = number(value)

        if np.isfinite(value):
            return value

    return np.nan


def money(value):
    value = number(value)

    if not np.isfinite(value):
        return "N/A"

    return f"${value:,.2f}"


def percent(value):
    value = number(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.2%}"


def multiple(value):
    value = number(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.2f}x"


def pick(df, *labels):
    if not isinstance(df, pd.DataFrame):
        return None

    for label in labels:
        if label in df.columns:
            return df[label]

    return None


def normalize_statement(df):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    result = df.T.copy()

    result.index = pd.to_datetime(
        result.index,
        errors="coerce"
    )

    result = result.loc[~result.index.isna()]

    return result.sort_index()


def get_series(df, *labels):
    result = pick(df, *labels)

    if not isinstance(result, pd.Series):
        return pd.Series(dtype=float)

    result = pd.to_numeric(
        result,
        errors="coerce"
    )

    result.index = pd.to_datetime(
        result.index,
        errors="coerce"
    )

    return (
        result.loc[~result.index.isna()]
        .sort_index()
        .dropna()
    )


def latest_value(df, *labels):
    values = get_series(df, *labels)

    if values.empty:
        return np.nan

    return number(values.iloc[-1])


# ============================================================
# 3. DOWNLOAD FINANCIAL DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner=False)
def download_data(ticker):

    stock = yf.Ticker(ticker)

    def get_statement(attribute):
        try:
            return normalize_statement(
                getattr(stock, attribute)
            )
        except Exception:
            return pd.DataFrame()

    income = get_statement("income_stmt")
    balance = get_statement("balance_sheet")
    cashflow = get_statement("cashflow")

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    price = first_valid(
        info.get("currentPrice"),
        info.get("regularMarketPrice")
    )

    if not np.isfinite(price):
        try:
            history = stock.history(period="5d")

            if not history.empty:
                price = number(
                    history["Close"].dropna().iloc[-1]
                )

        except Exception:
            pass

    return {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "info": info,
        "price": price
    }


# ============================================================
# 4. PROFESSOR'S FCF FORMULA
# ============================================================

def calculate_professor_fcf(cashflow):

    operating_cf = get_series(
        cashflow,
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities"
    )

    capex = get_series(
        cashflow,
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of Property Plant And Equipment"
    ).abs()

    dates = sorted(
        set(operating_cf.index) | set(capex.index)
    )

    result = pd.DataFrame(
        index=pd.DatetimeIndex(dates, name="Date")
   
