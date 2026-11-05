
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# STREAMLIT SETUP
# ============================================================
st.set_page_config(
    page_title="Stock Valuation Lab",
    page_icon="📈",
    layout="wide",
)

st.title("Stock Valuation Lab")
st.caption(
    "Fundamentals & FCF | DCF | Bull/Base/Bear | "
    "Comparable Companies | Sensitivity Analysis"
)

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# SAFE HELPERS — WEEK 6 PICK FUNCTION
# ============================================================
def pick(df, *labels, default=np.nan):
    """Return the first matching column label, else a default."""
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
    """Transpose yfinance statements and sort by date."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    result = df.T.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()]
    return result.sort_index()


def get_series(df, *labels):
    """Return a numeric line-item series, or an empty series."""
    result = pick(df, *labels, default=None)

    if not isinstance(result, pd.Series):
        return pd.Series(dtype=float)

    result = pd.to_numeric(result, errors="coerce")
    result.index = pd.to_datetime(result.index, errors="coerce")
    return result.loc[~result.index.isna()].sort_index().dropna()


def latest_value(df, *labels):
    values = get_series(df, *labels)
    return number(values.iloc[-1]) if not values.empty else np.nan


# ============================================================
# CSV CACHE
# ============================================================
def safe_ticker(ticker):
    return "".join(
        c for c in ticker.upper() if c.isalnum() or c in "._-"
    )


def save_cache(ticker, data):
    symbol = safe_ticker(ticker)

    for name in ("income", "balance", "cashflow"):
        try:
            data[name].to_csv(
                CACHE_DIR / f"{symbol}_{name}.csv",
                index_label="Date",
            )
        except Exception:
            pass

    try:
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
                data[name] = df.loc[~df.index.isna()].sort_index()
                found = True
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
                found = True
        except Exception:
            pass

    if not found:
        return None

    return {
        **data,
        "info": info,
        "price": number(info.get("currentPrice")),
    }


@st.cache_data(ttl=900, show_spinner=False)
def download_data(ticker):
    """Download the three statements independently."""
    stock = yf.Ticker(ticker)

    def safe_statement(attribute):
        try:
            return normalize_statement(getattr(stock, attribute))
        except Exception:
            return pd.DataFrame()

    income = safe_statement("income_stmt")
    balance = safe_statement("balance_sheet")
    cashflow = safe_statement("cashflow")

    info = {}
    try:
        info = stock.info or {}
    except Exception:
        pass

    price = number(
        info.get("currentPrice", info.get("regularMarketPrice"))
    )

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


def load_data(ticker, csv_only=False):
    if csv_only:
        cached = load_cache(ticker)
        return cached, "CSV cache" if cached else "No CSV cache found."

    try:
        data = download_data(ticker)

        available = [
            not data[key].empty
            for key in ("income", "balance", "cashflow")
        ]

        if any(available):
            save_cache(ticker, data)
            return data, "Yahoo Finance"

        raise RuntimeError("No usable financial statements returned.")

    except Exception as error:
        cached = load_cache(ticker)

        if cached:
            return cached, f"CSV fallback: {error}"

        return None, (
            f"Could not retrieve data: {error}. "
            "No cached CSV files were found."
        )


# ============================================================
# PART 1 — FUNDAMENTALS & FREE CASH FLOW
# ============================================================
def calculate_fundamentals(data):
    income = data["income"]
    balance = data["balance"]
    cashflow = data["cashflow"]

    revenue = get_series(
        income, "Total Revenue", "Operating Revenue", "Revenue"
    )

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

    # Combine dates from all four line items.
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

    # Required formula: FCF = Operating Cash Flow - |CapEx|
    annual["FCF"] = (
        annual["Operating Cash Flow"] - annual["CapEx"].abs()
    )

    annual["Net Margin"] = (
        annual["Net Income"] / annual["Revenue"].replace(0, np.nan)
    )

    annual["FCF Margin"] = (
        annual["FCF"] / annual["Revenue"].replace(0, np.nan)
    )

    annual = annual.sort_index().tail(4)

    current_assets = latest_value(
        balance, "Current Assets", "Total Current Assets"
    )

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

    current_ratio = divide(current_assets, current_liabilities)
    debt_equity = divide(debt, equity)

    # Average of the three most recent available annual FCF values.
    fcf_values = annual["FCF"].dropna()
    base_fcf = number(fcf_values.tail(3).mean())

    return annual, {
        "current_ratio": current_ratio,
        "debt_equity": debt_equity,
        "debt": debt,
        "base_fcf": base_fcf,
        "latest_revenue": number(
            annual["Revenue"].dropna().iloc[-1]
        ) if not annual["Revenue"].dropna().empty else np.nan,
    }


# ============================================================
# DCF CALCULATION
# ============================================================
def calculate_dcf(base_fcf, growth, terminal_growth, wacc,
                  years, cash, debt, shares):

    inputs = [
        base_fcf, growth, terminal_growth,
        wacc, cash, debt, shares
    ]

    if not all(np.isfinite(number(x)) for x in inputs):
        return np.nan

    if shares <= 0 or wacc <= terminal_growth:
        return np.nan

    projected_fcf = np.array([
        base_fcf * (1 + growth) ** year
        for year in range(1, years + 1)
    ])

    present_values = projected_fcf / (
        (1 + wacc) ** np.arange(1, years + 1)
    )

    terminal_value = (
        projected_fcf[-1] * (1 + terminal_growth)
        / (wacc - terminal_growth)
    )

    terminal_pv = terminal_value / (1 + wacc) ** years

    enterprise_value = present_values.sum() + terminal_pv
    equity_value = enterprise_value + cash - debt

    return equity_value / shares


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Company & data")

ticker = st.sidebar.text_input("Stock ticker", "MRK").strip().upper()

data_mode = st.sidebar.radio(
    "Data source",
    ["Yahoo Finance with CSV fallback", "CSV only"],
)

st.sidebar.caption(
    "CSV files belong in data_cache/: "
    "TICKER_income.csv, TICKER_balance.csv, "
    "TICKER_cashflow.csv, and optionally TICKER_info.csv."
)

st.sidebar.header("DCF assumptions")

years = st.sidebar.slider("Forecast years", 3, 10, 5)
risk_free = st.sidebar.slider(
    "Risk-free rate", 0.0, 0.10, 0.04, 0.005
)
beta = st.sidebar.number_input(
    "Beta", min_value=0.0, max_value=4.0, value=1.0
)
equity_risk_premium = st.sidebar.slider(
    "Equity risk premium", 0.0, 0.12, 0.05, 0.005
)
cost_of_debt = st.sidebar.slider(
    "Pre-tax cost of debt", 0.0, 0.20, 0.05, 0.005
)
tax_rate = st.sidebar.slider("Tax rate", 0.0, 0.50, 0.21, 0.01)


# ============================================================
# LOAD DATA
# ============================================================
if not ticker:
    st.info("Enter a ticker to begin.")
    st.stop()

with st.spinner(f"Loading {ticker} financial data..."):
    data, message = load_data(
        ticker,
        csv_only=(data_mode == "CSV only"),
    )

if data is None:
    st.error(message)
    st.stop()

st.caption(f"Data source: {message}")

annual, fundamentals = calculate_fundamentals(data)
info = data.get("info", {}) or {}

price = number(data.get("price"))
shares = number(info.get("sharesOutstanding"))
market_cap = number(info.get("marketCap"))

if not np.isfinite(shares) and np.isfinite(market_cap) and price > 0:
    shares = market_cap / price

if not np.isfinite(market_cap) and price > 0 and shares > 0:
    market_cap = price * shares

cash = number(info.get("totalCash"), default=0.0)
debt = number(fundamentals["debt"])

if not np.isfinite(debt):
    debt = number(info.get("totalDebt"), default=0.0)

base_fcf = fundamentals["base_fcf"]


# ============================================================
# PART 1 — REQUIRED FUNDAMENTALS
# ============================================================
st.header("Part 1 — Fundamentals & Free Cash Flow")

st.write(
    "The three annual financial statements are retrieved with "
    "yfinance, transposed to date rows, and sorted chronologically. "
    "Missing line items are displayed as N/A."
)

c1, c2, c3 = st.columns(3)
c1.metric("Current share price", money(price))
c2.metric("Current ratio", (
    f"{fundamentals['current_ratio']:.2f}"
    if np.isfinite(fundamentals["current_ratio"]) else "N/A"
))
c3.metric("Debt / equity", (
    f"{fundamentals['debt_equity']:.2f}"
    if np.isfinite(fundamentals["debt_equity"]) else "N/A"
))

st.subheader("Last four annual periods")

if annual.empty:
    st.warning("No annual fundamentals are available for this ticker.")
else:
    display = annual.copy()

    for column in [
        "Revenue", "Net Income", "Operating Cash Flow", "CapEx", "FCF"
    ]:
        display[column] = display[column].map(
            lambda x: "N/A" if pd.isna(x) else f"${x:,.0f}"
        )

    for column in ["Net Margin", "FCF Margin"]:
        display[column] = display[column].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.1%}"
        )

    display.index = display.index.strftime("%Y")
    st.dataframe(display, use_container_width=True)

with st.expander("View all three downloaded financial statements"):
    st.subheader("Income statement")
    st.dataframe(data["income"], use_container_width=True)

    st.subheader("Balance sheet")
    st.dataframe(data["balance"], use_container_width=True)

    st.subheader("Cash-flow statement")
    st.dataframe(data["cashflow"], use_container_width=True)

st.caption(
    "FCF = Operating Cash Flow − |CapEx|. "
    "Financial statement labels may differ between companies."
)


# ============================================================
# PART 2 — DCF AND WACC
# ============================================================
st.header("Part 2 — DCF Valuation")

cost_of_equity = risk_free + beta * equity_risk_premium

if not np.isfinite(market_cap) or market_cap <= 0:
    st.warning("Market capitalization is unavailable; WACC uses cost of equity.")
    wacc = cost_of_equity
else:
    total_capital = market_cap + max(debt, 0)
    wacc = (
        market_cap / total_capital * cost_of_equity
        + max(debt, 0) / total_capital
        * cost_of_debt * (1 - tax_rate)
    )

a, b = st.columns(2)
a.metric("CAPM cost of equity", percent(cost_of_equity))
b.metric("Estimated WACC", percent(wacc))

growth = st.slider(
    "Custom forecast FCF growth", -0.20, 0.40, 0.06, 0.01
)
terminal_growth = st.slider(
    "Custom terminal growth", -0.01, 0.06, 0.025, 0.0025
)

dcf_value = calculate_dcf(
    base_fcf, growth, terminal_growth, wacc,
    years, cash, debt, shares
)

st.metric("Custom DCF value per share", money(dcf_value))

if np.isfinite(price) and price > 0 and np.isfinite(dcf_value):
    st.metric(
        "DCF implied upside / downside",
        percent(dcf_value / price - 1),
    )


# ============================================================
# BULL / BASE / BEAR SCENARIOS
# ============================================================
st.subheader("Bull / Base / Bear Scenarios")

scenarios = {
    "Bull": {
        "growth": 0.12,
        "terminal_growth": 0.03,
        "wacc_adjustment": -0.01,
    },
    "Base": {
        "growth": 0.06,
        "terminal_growth": 0.025,
        "wacc_adjustment": 0.00,
    },
    "Bear": {
        "growth": -0.02,
        "terminal_growth": 0.015,
        "wacc_adjustment": 0.02,
    },
}

scenario_rows = []

for name, assumptions in scenarios.items():
    scenario_wacc = wacc + assumptions["wacc_adjustment"]

    value = calculate_dcf(
        base_fcf,
        assumptions["growth"],
        assumptions["terminal_growth"],
        scenario_wacc,
        years,
        cash,
        debt,
        shares,
    )

    upside = (
        value / price - 1
        if np.isfinite(value) and np.isfinite(price) and price > 0
        else np.nan
    )

    scenario_rows.append({
        "Scenario": name,
        "FCF Growth": assumptions["growth"],
        "Terminal Growth": assumptions["terminal_growth"],
        "WACC": scenario_wacc,
        "Intrinsic Value / Share": value,
        "Upside / Downside": upside,
    })

scenario_df = pd.DataFrame(scenario_rows)

scenario_display = scenario_df.copy()
scenario_display["FCF Growth"] = scenario_display["FCF Growth"].map(percent)
scenario_display["Terminal Growth"] = scenario_display["Terminal Growth"].map(percent)
scenario_display["WACC"] = scenario_display["WACC"].map(percent)
scenario_display["Intrinsic Value / Share"] = scenario_display["Intrinsic Value / Share"].map(money)
scenario_display["Upside / Downside"] = scenario_display["Upside / Downside"].map(percent)

st.dataframe(scenario_display, use_container_width=True)

chart = go.Figure()

chart.add_trace(go.Bar(
    x=scenario_df["Scenario"],
    y=scenario_df["Intrinsic Value / Share"],
    text=scenario_df["Intrinsic Value / Share"].map(money),
    textposition="outside",
    name="Intrinsic value",
))

if np.isfinite(price):
    chart.add_hline(
        y=price,
        line_dash="dash",
        annotation_text=f"Market price: {money(price)}",
    )

chart.update_layout(
    title="Bull / Base / Bear intrinsic value per share",
    yaxis_title="Value per share ($)",
    showlegend=False,
)

st.plotly_chart(chart, use_container_width=True)


# ============================================================
# PART 3 — COMPARABLE COMPANY ANALYSIS
# ============================================================
st.header("Part 3 — Comparable Company Analysis")

peer_input = st.text_input(
    "Peer tickers, separated by commas",
    "PFE, ABBV, BMY, AMGN, GILD",
)

peer_rows = []

for peer in dict.fromkeys(
    p.strip().upper()
    for p in peer_input.split(",")
    if p.strip() and p.strip().upper() != ticker
):
    try:
        peer_info = yf.Ticker(peer).info or {}

        peer_rows.append({
            "Ticker": peer,
            "P/E": number(peer_info.get("trailingPE")),
            "EV/EBITDA": number(peer_info.get("enterpriseToEbitda")),
            "P/S": number(peer_info.get("priceToSalesTrailing12Months")),
        })
    except Exception:
        peer_rows.append({
            "Ticker": peer,
            "P/E": np.nan,
            "EV/EBITDA": np.nan,
            "P/S": np.nan,
        })

peers = pd.DataFrame(peer_rows)

if not peers.empty:
    st.subheader("Peer valuation multiples")
    st.dataframe(
        peers.style.format({
            "P/E": "{:.2f}x",
            "EV/EBITDA": "{:.2f}x",
            "P/S": "{:.2f}x",
        }, na_rep="N/A"),
        use_container_width=True,
    )

    st.subheader("Median peer multiples")
    medians = peers[["P/E", "EV/EBITDA", "P/S"]].median()
    st.dataframe(
        medians.to_frame("Peer Median").style.format(
            "{:.2f}x", na_rep="N/A"
        ),
        use_container_width=True,
    )
else:
    st.info("Enter peer tickers to compare valuation multiples.")


# ============================================================
# PART 4 — SENSITIVITY ANALYSIS
# ============================================================
st.header("Part 4 — DCF Sensitivity Analysis")

if np.isfinite(base_fcf) and np.isfinite(shares) and shares > 0:
    wacc_values = np.unique(
        np.round(np.clip(
            wacc + np.array([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]),
            0.01, 0.50,
        ), 4)
    )

    terminal_values = np.unique(
        np.round(np.clip(
            terminal_growth + np.array(
                [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]
            ),
            -0.05, 0.15,
        ), 4)
    )

    sensitivity = pd.DataFrame(
        index=wacc_values,
        columns=terminal_values,
        dtype=float,
    )

    for rate in wacc_values:
        for terminal in terminal_values:
            if terminal < rate:
                sensitivity.loc[rate, terminal] = calculate_dcf(
                    base_fcf, growth, terminal, rate,
                    years, cash, debt, shares
                )
            else:
                sensitivity.loc[rate, terminal] = np.nan

    sensitivity.index = [f"{x:.2%}" for x in sensitivity.index]
    sensitivity.columns = [f"{x:.2%}" for x in sensitivity.columns]

    def cell_style(value):
        if pd.isna(value):
            return "background-color: #eeeeee; color: #666666;"

        values = sensitivity.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]

        if finite.size == 0 or finite.max() == finite.min():
            return ""

        ratio = (value - finite.min()) / (finite.max() - finite.min())

        if ratio < 0.5:
            t = ratio * 2
            red, green, blue = 220, int(80 + 140 * t), 80
        else:
            t = (ratio - 0.5) * 2
            red, green, blue = int(220 - 140 * t), 220, 100

        return (
            f"background-color: rgb({red}, {green}, {blue}); "
            "color: #111111;"
        )

    styled = sensitivity.style.map(cell_style).format(
        "${:,.2f}", na_rep="N/A"
    )

    st.dataframe(styled, use_container_width=True)

    st.caption(
        "Higher modeled values are green; lower values are red. "
        "N/A means terminal growth is not below WACC."
    )
else:
    st.warning(
        "Sensitivity analysis requires usable FCF and shares outstanding."
    )


# ============================================================
# DISCLAIMER
# ============================================================
st.divider()
st.caption(
    "Educational use only, not investment advice. "
    "Yahoo Finance data may be incomplete or delayed. "
    "Verify financial statements and assumptions against company filings."
)
