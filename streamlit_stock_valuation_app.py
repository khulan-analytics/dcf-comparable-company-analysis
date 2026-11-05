import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# SETUP
# ============================================================
st.set_page_config(
    page_title="Merck & Co., Inc. (MRK) Equity Valuation Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Merck & Co., Inc. (MRK) Equity Valuation Lab")
st.caption(
    "Created by Khulan Ganzorig | "
    "MS in Business Analytics, University at Buffalo"
)

st.markdown(
    """
    An equity research project covering financial statement analysis,
    DCF valuation, comparable company analysis, sensitivity analysis,
    and scenario modeling.
    """
)

st.divider()

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# DISPLAY HELPERS (work on old and new Streamlit versions)
# ============================================================
def show_df(obj):
    try:
        st.dataframe(obj, width="stretch")
    except Exception:
        st.dataframe(obj, use_container_width=True)


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# SAFE HELPERS
# ============================================================
def pick(df, *labels, default=np.nan):
    """Return the first matching column, else a default."""
    if not isinstance(df, pd.DataFrame):
        return default
    for label in labels:
        if label in df.columns:
            return df[label]
    return default


def number(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, (pd.Series, pd.DataFrame, list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            value = np.asarray(value).reshape(-1)[0]
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError, OverflowError):
        return default


def divide(a, b):
    a, b = number(a), number(b)
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def money(value):
    value = number(value)
    return "N/A" if not np.isfinite(value) else f"${value:,.2f}"


def percent(value):
    value = number(value)
    return "N/A" if not np.isfinite(value) else f"{value:.1%}"


def normalize_statement(df):
    """Transpose yfinance statements into date rows, sorted by date."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    result = df.T.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()]
    return result.sort_index()


def get_series(df, *labels):
    """Safely retrieve one statement line item as a dated numeric Series."""
    result = pick(df, *labels, default=None)
    if not isinstance(result, pd.Series):
        return pd.Series(dtype=float)
    result = pd.to_numeric(result, errors="coerce")
    result.index = pd.to_datetime(result.index, errors="coerce")
    return result.loc[~result.index.isna()].sort_index().dropna()


def latest_value(df, *labels):
    values = get_series(df, *labels)
    return number(values.iloc[-1]) if not values.empty else np.nan


def first_valid(*values):
    for value in values:
        value = number(value)
        if np.isfinite(value):
            return value
    return np.nan


# ============================================================
# CSV CACHE
# ============================================================
def safe_ticker(ticker):
    return "".join(c for c in ticker.upper() if c.isalnum() or c in "._-")


def save_cache(ticker, data):
    symbol = safe_ticker(ticker)

    for name in ("income", "balance", "cashflow"):
        try:
            if not data[name].empty:
                data[name].to_csv(
                    CACHE_DIR / f"{symbol}_{name}.csv",
                    index_label="Date",
                )
        except Exception:
            pass

    try:
        if data.get("info"):
            pd.DataFrame([data["info"]]).to_csv(
                CACHE_DIR / f"{symbol}_info.csv",
                index=False,
            )
    except Exception:
        pass


def load_cache(ticker):
    symbol = safe_ticker(ticker)
    data = {}
    found = False

    for name in ("income", "balance", "cashflow"):
        path = CACHE_DIR / f"{symbol}_{name}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df.loc[~df.index.isna()].sort_index()
                data[name] = df
                found = found or not df.empty
            except Exception:
                data[name] = pd.DataFrame()
        else:
            data[name] = pd.DataFrame()

    info = {}
    info_path = CACHE_DIR / f"{symbol}_info.csv"
    if info_path.exists():
        try:
            df = pd.read_csv(info_path)
            if not df.empty:
                info = df.iloc[0].to_dict()
        except Exception:
            pass

    if not found:
        return None

    price = first_valid(info.get("currentPrice"), info.get("regularMarketPrice"))
    return {**data, "info": info, "price": price}


# ============================================================
# DATA DOWNLOAD
# Raises on failure so Streamlit does NOT cache an empty result.
# ============================================================
@st.cache_data(ttl=900, show_spinner=False)
def download_data(ticker):
    stock = yf.Ticker(ticker)

    def statement(attribute):
        try:
            return normalize_statement(getattr(stock, attribute))
        except Exception:
            return pd.DataFrame()

    income = statement("income_stmt")
    balance = statement("balance_sheet")
    cashflow = statement("cashflow")

    if income.empty and balance.empty and cashflow.empty:
        raise RuntimeError("Yahoo Finance returned no financial statements.")

    info = {}
    try:
        info = stock.info or {}
    except Exception:
        pass

    price = first_valid(info.get("currentPrice"), info.get("regularMarketPrice"))

    if not np.isfinite(price):
        try:
            history = stock.history(period="5d")
            if not history.empty:
                price = number(history["Close"].dropna().iloc[-1])
        except Exception:
            pass

    return {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "info": info,
        "price": price,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_peer_multiples(peer):
    """Cached per peer. Raises on failure so failures are not cached."""
    info = yf.Ticker(peer).info or {}
    return {
        "Ticker": peer,
        "P/E": number(info.get("trailingPE")),
        "EV/EBITDA": number(info.get("enterpriseToEbitda")),
        "P/S": number(info.get("priceToSalesTrailing12Months")),
    }


def load_data(ticker, csv_only=False):
    if csv_only:
        cached = load_cache(ticker)
        if cached:
            return cached, "CSV cache"
        return None, "No usable CSV cache found in data_cache/."

    try:
        data = download_data(ticker)
        save_cache(ticker, data)
        return data, "Yahoo Finance"
    except Exception as error:
        cached = load_cache(ticker)
        if cached:
            return cached, f"CSV fallback used (Yahoo error: {error})"
        return None, (
            f"Unable to retrieve financial data: {error}. "
            "No cached CSV files were found."
        )


# ============================================================
# FUNDAMENTALS AND FCF
# ============================================================
def calculate_fundamentals(data):
    income = data["income"]
    balance = data["balance"]
    cashflow = data["cashflow"]

    revenue = get_series(income, "Total Revenue", "Operating Revenue", "Revenue")
    net_income = get_series(
        income,
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income From Continuing Operation Net Minority Interest",
    )
    operating_cf = get_series(
        cashflow,
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    )
    capex = get_series(
        cashflow,
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of Property Plant And Equipment",
    )

    dates = sorted(
        set(revenue.index)
        | set(net_income.index)
        | set(operating_cf.index)
        | set(capex.index)
    )

    annual = pd.DataFrame(index=pd.DatetimeIndex(dates, name="Date"))
    annual["Revenue"] = revenue
    annual["Net Income"] = net_income
    annual["Operating Cash Flow"] = operating_cf
    annual["CapEx"] = capex

    # FCF = Operating Cash Flow - |CapEx|
    annual["FCF"] = annual["Operating Cash Flow"] - annual["CapEx"].abs()
    annual["Net Margin"] = annual["Net Income"] / annual["Revenue"].replace(0, np.nan)
    annual["FCF Margin"] = annual["FCF"] / annual["Revenue"].replace(0, np.nan)
    annual = annual.sort_index()

    # Average of up to the latest three annual FCF observations.
    fcf_values = annual["FCF"].dropna()
    fcf_count = min(len(fcf_values), 3)
    base_fcf = number(fcf_values.tail(3).mean()) if fcf_count >= 1 else np.nan

    current_assets = latest_value(balance, "Current Assets", "Total Current Assets")
    current_liabilities = latest_value(
        balance, "Current Liabilities", "Total Current Liabilities"
    )
    debt = latest_value(
        balance,
        "Total Debt",
        "Long Term Debt And Capital Lease Obligation",
        "Long Term Debt",
    )
    equity = latest_value(
        balance,
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Common Stock Equity",
        "Total Equity Gross Minority Interest",
    )

    return annual, {
        "current_ratio": divide(current_assets, current_liabilities),
        "debt_equity": divide(debt, equity),
        "debt": debt,
        "base_fcf": base_fcf,
        "fcf_count": fcf_count,
    }


# ============================================================
# DCF MODEL
# ============================================================
def calculate_dcf(base_fcf, growth, terminal_growth, wacc, years, cash, debt, shares):
    values = [base_fcf, growth, terminal_growth, wacc, cash, debt, shares]
    if not all(np.isfinite(number(x)) for x in values):
        return None
    if shares <= 0 or wacc <= terminal_growth:
        return None

    forecast_years = np.arange(1, years + 1)
    projected = base_fcf * (1 + growth) ** forecast_years
    pv = projected / (1 + wacc) ** forecast_years

    terminal = projected[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal / (1 + wacc) ** years
    enterprise = pv.sum() + pv_terminal
    equity_value = enterprise + cash - debt
    intrinsic = equity_value / shares

    return {
        "projected": projected,
        "pv": pv,
        "terminal": terminal,
        "pv_terminal": pv_terminal,
        "enterprise": enterprise,
        "equity": equity_value,
        "intrinsic": intrinsic,
        "terminal_pct": pv_terminal / enterprise if enterprise != 0 else np.nan,
    }


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Company and data")

ticker = st.sidebar.text_input("Stock ticker", value="MRK").strip().upper()

data_mode = st.sidebar.radio(
    "Data source",
    ["Yahoo Finance with CSV fallback", "CSV only"],
)

if st.sidebar.button("🔄 Clear cache and reload"):
    st.cache_data.clear()
    st.rerun()

show_debug = st.sidebar.checkbox("Show debug panel", value=False)

st.sidebar.caption(
    "CSV files are saved in data_cache as TICKER_income.csv, "
    "TICKER_balance.csv, TICKER_cashflow.csv, and TICKER_info.csv."
)

st.sidebar.header("Part 2 — DCF assumptions")

years = st.sidebar.slider("Forecast years", min_value=3, max_value=10, value=5)

growth = st.sidebar.slider(
    "Forecast FCF growth",
    min_value=-0.20, max_value=0.40, value=0.06, step=0.01, format="%.2f",
)

terminal_growth = st.sidebar.slider(
    "Terminal growth",
    min_value=-0.01, max_value=0.06, value=0.025, step=0.0025, format="%.3f",
)

risk_free = st.sidebar.slider("Risk-free rate", 0.0, 0.10, 0.04, 0.005)

beta = st.sidebar.number_input(
    "Beta", min_value=0.0, max_value=4.0, value=1.0, step=0.1
)

equity_risk_premium = st.sidebar.slider("Equity risk premium", 0.0, 0.12, 0.05, 0.005)
cost_of_debt = st.sidebar.slider("Pre-tax cost of debt", 0.0, 0.20, 0.05, 0.005)
tax_rate = st.sidebar.slider("Tax rate", 0.0, 0.50, 0.21, 0.01)


# ============================================================
# LOAD COMPANY DATA
# ============================================================
if not ticker:
    st.info("Enter a stock ticker in the sidebar.")
    st.stop()

with st.spinner(f"Loading data for {ticker}..."):
    data, data_message = load_data(ticker, csv_only=(data_mode == "CSV only"))

if data is None:
    st.error(data_message)
    st.info(
        "Tip: run the app locally once so data_cache/*.csv files are created, "
        "then commit them to your repo so the fallback works when deployed."
    )
    st.stop()

st.caption(f"Data source: {data_message}")

annual, fundamentals = calculate_fundamentals(data)
info = data.get("info", {}) or {}

price = number(data.get("price"))
shares = number(info.get("sharesOutstanding"))
market_cap = number(info.get("marketCap"))

if not np.isfinite(market_cap) and price > 0 and np.isfinite(shares) and shares > 0:
    market_cap = price * shares

if not np.isfinite(shares) and np.isfinite(market_cap) and market_cap > 0 and price > 0:
    shares = market_cap / price

cash = first_valid(
    info.get("totalCash"),
    latest_value(
        data["balance"],
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents And Short Term Investments",
    ),
    0.0,
)

debt = first_valid(fundamentals["debt"], info.get("totalDebt"), 0.0)
base_fcf = fundamentals["base_fcf"]

if show_debug:
    with st.expander("🐞 Debug panel", expanded=True):
        st.write("Data source:", data_message)
        st.write(
            "Statement shapes:",
            {k: data[k].shape for k in ("income", "balance", "cashflow")},
        )
        st.write(
            {
                "price": price,
                "shares": shares,
                "market_cap": market_cap,
                "cash": cash,
                "debt": debt,
                "base_fcf": base_fcf,
                "fcf_observations_used": fundamentals["fcf_count"],
            }
        )
        if not annual.empty:
            show_df(annual[["Operating Cash Flow", "CapEx", "FCF"]])


# ============================================================
# PART 1 — FUNDAMENTALS & FREE CASH FLOW
# ============================================================
st.header("Part 1 — Fundamentals & Free Cash Flow")

st.write(
    "Annual income statement, balance sheet, and cash-flow data are "
    "transposed into date rows and sorted chronologically. "
    "Missing statement labels are handled safely."
)

col1, col2, col3 = st.columns(3)
col1.metric("Current market price", money(price))
col2.metric(
    "Current ratio",
    f"{fundamentals['current_ratio']:.2f}"
    if np.isfinite(fundamentals["current_ratio"]) else "N/A",
)
col3.metric(
    "Debt / equity",
    f"{fundamentals['debt_equity']:.2f}"
    if np.isfinite(fundamentals["debt_equity"]) else "N/A",
)

st.subheader("Last four annual periods")

if annual.empty:
    st.warning("No annual fundamentals are available.")
else:
    display = annual.tail(4).copy()

    for col in ["Revenue", "Net Income", "Operating Cash Flow", "CapEx", "FCF"]:
        display[col] = display[col].map(
            lambda x: "N/A" if pd.isna(x) else f"${x:,.0f}"
        )

    for col in ["Net Margin", "FCF Margin"]:
        display[col] = display[col].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.1%}"
        )

    display.index = display.index.strftime("%Y")
    show_df(display)

if np.isfinite(base_fcf):
    label = (
        "Three-year average FCF"
        if fundamentals["fcf_count"] >= 3
        else f"Average FCF ({fundamentals['fcf_count']} year(s) available)"
    )
    st.metric(label, money(base_fcf))
    if fundamentals["fcf_count"] < 3:
        st.warning(
            "Fewer than three annual FCF observations were available, "
            "so the base FCF uses fewer years than intended."
        )
else:
    st.warning("No annual FCF observations are available for the DCF.")

st.caption(
    "FCF = Operating Cash Flow − |CapEx|. Missing data is shown as N/A where possible."
)

with st.expander("View the three financial statements"):
    st.subheader("Income statement")
    show_df(data["income"])
    st.subheader("Balance sheet")
    show_df(data["balance"])
    st.subheader("Cash-flow statement")
    show_df(data["cashflow"])


# ============================================================
# PART 2 — DCF VALUATION
# ============================================================
st.header("Part 2 — DCF Valuation")

cost_of_equity = risk_free + beta * equity_risk_premium

if np.isfinite(market_cap) and market_cap > 0:
    equity_weight = market_cap / (market_cap + max(debt, 0))
    debt_weight = 1 - equity_weight
    capm_wacc = (
        equity_weight * cost_of_equity
        + debt_weight * cost_of_debt * (1 - tax_rate)
    )
else:
    capm_wacc = cost_of_equity
    st.warning(
        "Market capitalization is unavailable. Using CAPM cost of equity "
        "as the starting discount rate."
    )

st.sidebar.header("Adjustable discount rate")

wacc = st.sidebar.slider(
    "WACC",
    min_value=0.01,
    max_value=0.30,
    value=float(np.clip(capm_wacc, 0.01, 0.30)),
    step=0.005,
    format="%.3f",
)

col1, col2, col3 = st.columns(3)
col1.metric("CAPM cost of equity", percent(cost_of_equity))
col2.metric("CAPM-based starting WACC", percent(capm_wacc))
col3.metric("Selected WACC", percent(wacc))

dcf = None

if not np.isfinite(base_fcf):
    st.error("DCF cannot be calculated: no usable annual FCF data.")
elif not np.isfinite(shares) or shares <= 0:
    st.error("DCF cannot be calculated: shares outstanding are unavailable.")
elif terminal_growth >= wacc:
    st.error(
        "Terminal growth must be lower than WACC. "
        "Reduce terminal growth or increase WACC."
    )
else:
    dcf = calculate_dcf(
        base_fcf, growth, terminal_growth, wacc, years, cash, debt, shares
    )

if dcf is not None:
    intrinsic = dcf["intrinsic"]
    dcf_value = intrinsic

    col1, col2, col3 = st.columns(3)
    col1.metric("DCF intrinsic value / share", money(intrinsic))
    col2.metric("Market price", money(price))
    col3.metric(
        "Implied upside / downside",
        percent(intrinsic / price - 1)
        if np.isfinite(price) and price > 0 else "N/A",
    )

    col1, col2 = st.columns(2)
    col1.metric("Enterprise value", money(dcf["enterprise"]))
    col2.metric(
        "PV of terminal value / enterprise value",
        percent(dcf["terminal_pct"]),
    )

    forecast = pd.DataFrame({
        "Year": np.arange(1, years + 1),
        "Projected FCF": dcf["projected"],
        "PV of FCF": dcf["pv"],
    })

    st.subheader("DCF forecast")
    show_df(
        forecast.style.format({
            "Projected FCF": "${:,.0f}",
            "PV of FCF": "${:,.0f}",
        })
    )
else:
    intrinsic = np.nan
    dcf_value = np.nan


# ============================================================
# PART 3 — COMPARABLE COMPANY ANALYSIS
# ============================================================
st.header("Part 3 — Comparable Company Analysis")

peer_input = st.text_input(
    "Enter peer tickers separated by commas",
    value="PFE, ABBV, BMY, AMGN, GILD",
)

peer_tickers = list(dict.fromkeys(
    peer.strip().upper()
    for peer in peer_input.split(",")
    if peer.strip() and peer.strip().upper() != ticker
))

peer_rows = []

with st.spinner("Retrieving peer valuation multiples..."):
    for peer in peer_tickers:
        try:
            peer_rows.append(get_peer_multiples(peer))
        except Exception:
            peer_rows.append({
                "Ticker": peer,
                "P/E": np.nan,
                "EV/EBITDA": np.nan,
                "P/S": np.nan,
            })

peers = pd.DataFrame(peer_rows, columns=["Ticker", "P/E", "EV/EBITDA", "P/S"])

comps_value = np.nan

if peers.empty:
    st.info("Enter peer tickers to calculate comparable-company valuations.")
else:
    median_row = {"Ticker": "PEER MEDIAN"}
    for col in ["P/E", "EV/EBITDA", "P/S"]:
        values = peers[col].dropna()
        median_row[col] = values.median() if not values.empty else np.nan

    peers_display = pd.concat(
        [peers, pd.DataFrame([median_row])], ignore_index=True
    )

    def highlight_median(row):
        if row["Ticker"] == "PEER MEDIAN":
            return ["background-color: #d9ead3; font-weight: bold;"] * len(row)
        return [""] * len(row)

    st.subheader("Peer multiples and median")
    show_df(
        peers_display.style
        .format(
            {"P/E": "{:.2f}x", "EV/EBITDA": "{:.2f}x", "P/S": "{:.2f}x"},
            na_rep="N/A",
        )
        .apply(highlight_median, axis=1)
    )

    # Company metrics
    eps = number(info.get("trailingEps"))

    sales = first_valid(
        info.get("totalRevenue"),
        latest_value(
            data["income"], "Total Revenue", "Operating Revenue", "Revenue"
        ),
    )

    ebitda = number(info.get("ebitda"))

    if not np.isfinite(ebitda):
        operating_income = latest_value(data["income"], "Operating Income", "EBIT")
        depreciation = latest_value(
            data["cashflow"],
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
        )
        if np.isfinite(operating_income) and np.isfinite(depreciation):
            ebitda = operating_income + depreciation

    pe_median = number(median_row["P/E"])
    ev_ebitda_median = number(median_row["EV/EBITDA"])
    ps_median = number(median_row["P/S"])

    pe_value = (
        eps * pe_median
        if np.isfinite(eps) and eps > 0
        and np.isfinite(pe_median) and pe_median > 0
        else np.nan
    )

    ev_ebitda_value = np.nan
    if (
        np.isfinite(ebitda) and ebitda > 0
        and np.isfinite(ev_ebitda_median) and ev_ebitda_median > 0
        and np.isfinite(shares) and shares > 0
    ):
        ev = ebitda * ev_ebitda_median
        ev_ebitda_value = (ev + cash - debt) / shares

    ps_value = (
        sales * ps_median / shares
        if np.isfinite(sales) and sales > 0
        and np.isfinite(ps_median) and ps_median > 0
        and np.isfinite(shares) and shares > 0
        else np.nan
    )

    implied_values = pd.DataFrame({
        "Method": ["P/E", "EV/EBITDA", "P/S"],
        "Peer Median Multiple": [pe_median, ev_ebitda_median, ps_median],
        "Company Metric": [eps, ebitda, sales],
        "Implied Value / Share": [pe_value, ev_ebitda_value, ps_value],
    })

    st.subheader(f"{ticker} Implied Share Price Using Peer Market Benchmarks")
    show_df(
        implied_values.style.format(
            {
                "Peer Median Multiple": "{:.2f}x",
                "Company Metric": "{:,.2f}",
                "Implied Value / Share": "${:,.2f}",
            },
            na_rep="N/A",
        )
    )

    valid_comp_values = implied_values["Implied Value / Share"].dropna()
    comps_value = (
        valid_comp_values.median() if not valid_comp_values.empty else np.nan
    )

    st.subheader("Valuation comparison")

    comparison = pd.DataFrame({
        "Method": [
            "Comparable-company value",
            "DCF intrinsic value",
            "Current market price",
        ],
        "Value per Share": [comps_value, dcf_value, price],
    })

    show_df(
        comparison.style.format({"Value per Share": "${:,.2f}"}, na_rep="N/A")
    )

    if (
        np.isfinite(comps_value)
        and np.isfinite(dcf_value)
        and np.isfinite(price)
    ):
        comparison_chart = go.Figure()
        comparison_chart.add_trace(go.Bar(
            x=["Comparable Companies", "DCF", "Market Price"],
            y=[comps_value, dcf_value, price],
            text=[money(comps_value), money(dcf_value), money(price)],
            textposition="outside",
        ))
        comparison_chart.update_layout(
            title="Comps vs. DCF vs. Market Price",
            yaxis_title="Value per share ($)",
            showlegend=False,
        )
        show_chart(comparison_chart)

    st.markdown(
        """
        **Why the valuations may differ**

        - **Comparable-company valuation:** reflects how the market prices
          peer earnings, EBITDA, or sales.
        - **DCF valuation:** depends on projected cash flows, forecast growth,
          WACC, and terminal growth.
        - **Market price:** reflects investors' current expectations,
          sentiment, risks, and other information.

        Peer selection, accounting differences, margins, growth, debt,
        and missing multiples can also cause differences.
        """
    )


# ============================================================
# PART 4 — TWO-WAY SENSITIVITY ANALYSIS
# ============================================================
st.header("Part 4 — Sensitivity Analysis")

if dcf is None:
    st.warning(
        "Sensitivity analysis requires a valid base-case DCF. "
        "Check FCF data and ensure terminal growth is below WACC."
    )
else:
    wacc_values = sorted(set(
        round(wacc + delta, 6)
        for delta in [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]
        if 0.01 <= wacc + delta <= 0.30
    ))

    terminal_values = sorted(set(
        round(terminal_growth + delta, 6)
        for delta in [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]
        if -0.05 <= terminal_growth + delta <= 0.15
    ))

    sensitivity = pd.DataFrame(
        index=wacc_values, columns=terminal_values, dtype=float
    )

    for rate in wacc_values:
        for terminal_g in terminal_values:
            result = calculate_dcf(
                base_fcf, growth, terminal_g, rate, years, cash, debt, shares
            )
            sensitivity.loc[rate, terminal_g] = (
                result["intrinsic"] if result is not None else np.nan
            )

    # Locate the base-case cell by nearest grid value (robust to rounding)
    base_wacc_key = min(sensitivity.index, key=lambda x: abs(float(x) - wacc))
    base_terminal_key = min(
        sensitivity.columns, key=lambda x: abs(float(x) - terminal_growth)
    )

    x_labels = [f"{g:.2%}" for g in terminal_values]
    y_labels = [f"{r:.2%}" for r in wacc_values]

    base_x = terminal_values.index(base_terminal_key)
    base_y = wacc_values.index(base_wacc_key)

    cell_text = []
    for i, rate in enumerate(wacc_values):
        row = []
        for j, terminal_g in enumerate(terminal_values):
            value = sensitivity.loc[rate, terminal_g]
            text = money(value) if np.isfinite(value) else "N/A"
            if i == base_y and j == base_x:
                text = f"BASE<br>{text}"
            row.append(text)
        cell_text.append(row)

    heatmap = go.Figure(
        data=go.Heatmap(
            z=sensitivity.to_numpy(dtype=float),
            x=x_labels,
            y=y_labels,
            text=cell_text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            colorbar={"title": "Value / share"},
            hovertemplate=(
                "WACC: %{y}<br>Terminal growth: %{x}<br>"
                "Intrinsic value: $%{z:,.2f}<extra></extra>"
            ),
        )
    )

    heatmap.add_shape(
        type="rect",
        x0=base_x - 0.5, x1=base_x + 0.5,
        y0=base_y - 0.5, y1=base_y + 0.5,
        line={"color": "black", "width": 4},
    )

    heatmap.update_layout(
        title=f"{ticker} DCF Sensitivity: WACC vs. Terminal Growth",
        xaxis_title="Terminal Growth",
        yaxis_title="WACC",
    )

    show_chart(heatmap)

    st.caption(
        "Rows represent WACC and columns represent terminal growth. "
        "The outlined BASE cell uses the selected Part 2 assumptions."
    )

    wacc_range = sensitivity[base_terminal_key].dropna().astype(float)
    terminal_range = sensitivity.loc[base_wacc_key].dropna().astype(float)

    wacc_impact = (
        wacc_range.max() - wacc_range.min() if len(wacc_range) >= 2 else np.nan
    )
    terminal_impact = (
        terminal_range.max() - terminal_range.min()
        if len(terminal_range) >= 2 else np.nan
    )

    st.subheader("Sensitivity Interpretation")

    st.write(
        f"At the base-case assumptions, {ticker}'s estimated intrinsic "
        f"value is **{money(dcf_value)} per share**."
    )

    if np.isfinite(wacc_impact) and np.isfinite(terminal_impact):
        if wacc_impact > terminal_impact:
            st.write(
                f"**WACC has the greater impact on valuation.** Changing WACC "
                f"produces a valuation range of {money(wacc_impact)}, compared "
                f"with {money(terminal_impact)} for terminal growth."
            )
        elif terminal_impact > wacc_impact:
            st.write(
                f"**Terminal growth has the greater impact on valuation.** "
                f"Changing terminal growth produces a valuation range of "
                f"{money(terminal_impact)}, compared with {money(wacc_impact)} "
                f"for WACC."
            )
        else:
            st.write(
                "WACC and terminal growth produce equal valuation ranges "
                "across the tested grid."
            )

    if np.isfinite(price):
        above = int((sensitivity > price).sum().sum())
        below = int((sensitivity < price).sum().sum())

        st.write(
            f"Compared with the current market price of **{money(price)}**, "
            f"{above} grid combinations produce an intrinsic value above "
            f"market price, suggesting potential undervaluation. {below} "
            f"combinations produce a value below market price, suggesting "
            f"potential overvaluation."
        )

        st.caption(
            "These comparisons reflect the model assumptions and are not "
            "predictions of future stock performance."
        )


# ============================================================
# PART 5 — EXTENSION: BULL / BASE / BEAR SCENARIOS
# ============================================================
st.header("Part 5 — Extension: Bull / Base / Bear Scenarios")

st.write(
    "These are alternative assumption sets. They use the same base FCF, "
    "cash, debt, shares, and forecast horizon, but change growth, "
    "terminal growth, and WACC."
)

scenario_presets = {
    "Bear": {"growth": -0.02, "terminal_growth": 0.015, "wacc_adjustment": 0.02},
    "Base": {"growth": 0.06, "terminal_growth": 0.025, "wacc_adjustment": 0.00},
    "Bull": {"growth": 0.12, "terminal_growth": 0.03, "wacc_adjustment": -0.01},
}

selected_scenario = st.selectbox(
    "Choose a scenario to inspect", ["Base", "Bull", "Bear"], index=0
)
selected = scenario_presets[selected_scenario]

st.write(f"**Selected scenario: {selected_scenario}**")

scenario_cols = st.columns(3)
scenario_cols[0].metric("Scenario FCF growth", percent(selected["growth"]))
scenario_cols[1].metric(
    "Scenario terminal growth", percent(selected["terminal_growth"])
)
scenario_cols[2].metric("Scenario WACC", percent(wacc + selected["wacc_adjustment"]))

scenario_results = []

for scenario_name, assumptions in scenario_presets.items():
    scenario_rate = wacc + assumptions["wacc_adjustment"]

    result = calculate_dcf(
        base_fcf,
        assumptions["growth"],
        assumptions["terminal_growth"],
        scenario_rate,
        years,
        cash,
        debt,
        shares,
    )

    scenario_value = result["intrinsic"] if result is not None else np.nan

    scenario_upside = (
        scenario_value / price - 1
        if np.isfinite(scenario_value) and np.isfinite(price) and price > 0
        else np.nan
    )

    scenario_results.append({
        "Scenario": scenario_name,
        "FCF Growth": assumptions["growth"],
        "Terminal Growth": assumptions["terminal_growth"],
        "WACC": scenario_rate,
        "Intrinsic Value / Share": scenario_value,
        "Upside / Downside": scenario_upside,
    })

scenario_table = pd.DataFrame(scenario_results)
scenario_display = scenario_table.copy()

for col in ["FCF Growth", "Terminal Growth", "WACC", "Upside / Downside"]:
    scenario_display[col] = scenario_display[col].map(percent)

scenario_display["Intrinsic Value / Share"] = (
    scenario_display["Intrinsic Value / Share"].map(money)
)

st.subheader("Scenario comparison")


def highlight_selected(row):
    if row["Scenario"] == selected_scenario:
        return ["background-color: #d9ead3; font-weight: bold;"] * len(row)
    return [""] * len(row)


show_df(scenario_display.style.apply(highlight_selected, axis=1))

scenario_chart = go.Figure()
scenario_chart.add_trace(go.Bar(
    x=scenario_table["Scenario"],
    y=scenario_table["Intrinsic Value / Share"],
    text=scenario_table["Intrinsic Value / Share"].map(money),
    textposition="outside",
    name="Scenario value",
))

if np.isfinite(price):
    scenario_chart.add_hline(
        y=price,
        line_dash="dash",
        annotation_text=f"Market price: {money(price)}",
    )

scenario_chart.update_layout(
    title="Bull / Base / Bear valuation",
    yaxis_title="Intrinsic value per share ($)",
    showlegend=False,
)

show_chart(scenario_chart)

st.caption(
    "The scenario presets are illustrative assumptions, not forecasts. "
    "The Base scenario is a preset and may differ from your custom Part 2 "
    "assumptions. You can compare both in the scenario table and DCF section."
)


# ============================================================
# DISCLAIMER
# ============================================================
st.divider()

st.caption(
    "Educational use only, not investment advice. "
    "Yahoo Finance data may be incomplete or delayed. "
    "Check company filings and verify assumptions before relying "
    "on any valuation."
)
