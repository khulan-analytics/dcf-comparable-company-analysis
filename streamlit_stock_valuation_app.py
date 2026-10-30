
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
    page_title="Stock Valuation Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title("Stock Valuation Dashboard")
st.caption(
    "Part 1: Fundamentals | Part 2: DCF | "
    "Part 3: Comparable Companies | Part 4: Sensitivity | "
    "Part 5: Extensions"
)

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ============================================================
# SAFE HELPERS — WEEK 6 PICK()
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


def multiple(value):
    value = number(value)
    return "N/A" if not np.isfinite(value) else f"{value:.2f}x"


def normalize_statement(df):
    """Transpose yfinance statements and sort date rows."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    result = df.T.copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[~result.index.isna()]
    return result.sort_index()


def get_series(df, *labels):
    """Use pick() to safely retrieve a statement line item."""
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
    return "".join(
        c for c in ticker.upper()
        if c.isalnum() or c in "._-"
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
        pd.DataFrame([data.get("info", {})]).to_csv(
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
                found = True
        except Exception:
            pass

    if not found:
        return None

    price = first_valid(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
    )

    return {
        **data,
        "info": info,
        "price": price,
    }


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

    info = {}

    try:
        info = stock.info or {}
    except Exception:
        pass

    price = first_valid(
        info.get("currentPrice"),
        info.get("regularMarketPrice"),
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
        return cached, (
            "CSV cache loaded"
            if cached else
            "No usable CSV cache found."
        )

    try:
        data = download_data(ticker)

        if any(
            not data[key].empty
            for key in ("income", "balance", "cashflow")
        ):
            save_cache(ticker, data)
            return data, "Yahoo Finance"

        raise RuntimeError("Yahoo Finance returned no usable statements.")

    except Exception as error:
        cached = load_cache(ticker)

        if cached:
            return cached, f"CSV fallback used ({error})"

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

    revenue = get_series(
        income,
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
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
        annual["Net Income"]
        / annual["Revenue"].replace(0, np.nan)
    )

    annual["FCF Margin"] = (
        annual["FCF"]
        / annual["Revenue"].replace(0, np.nan)
    )

    annual = annual.sort_index()

    # Use the latest three available annual FCF observations.
    fcf_values = annual["FCF"].dropna()
    base_fcf = (
        number(fcf_values.tail(3).mean())
        if len(fcf_values) >= 3 else np.nan
    )

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

    return annual, {
        "current_ratio": divide(current_assets, current_liabilities),
        "debt_equity": divide(debt, equity),
        "debt": debt,
        "base_fcf": base_fcf,
    }


# ============================================================
# DCF MODEL
# ============================================================
def calculate_dcf(
    base_fcf,
    growth,
    terminal_growth,
    wacc,
    years,
    cash,
    debt,
    shares,
):
    values = [
        base_fcf, growth, terminal_growth,
        wacc, cash, debt, shares,
    ]

    if not all(np.isfinite(number(x)) for x in values):
        return None

    if shares <= 0 or wacc <= terminal_growth:
        return None

    forecast_years = np.arange(1, years + 1)
    projected = base_fcf * (1 + growth) ** forecast_years
    pv = projected / (1 + wacc) ** forecast_years

    terminal = (
        projected[-1] * (1 + terminal_growth)
        / (wacc - terminal_growth)
    )

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
        "terminal_pct": (
            pv_terminal / enterprise
            if enterprise != 0 else np.nan
        ),
    }


# ============================================================
# SIDEBAR — COMPANY, DATA, AND PART 2 ASSUMPTIONS
# ============================================================
st.sidebar.header("Company and data")

ticker = st.sidebar.text_input(
    "Stock ticker",
    value="MRK",
).strip().upper()

data_mode = st.sidebar.radio(
    "Data source",
    [
        "Yahoo Finance with CSV fallback",
        "CSV only",
    ],
)

st.sidebar.caption(
    "CSV files are saved in data_cache using the names "
    "TICKER_income.csv, TICKER_balance.csv, "
    "TICKER_cashflow.csv, and TICKER_info.csv."
)

st.sidebar.header("Part 2 — DCF assumptions")

years = st.sidebar.slider(
    "Forecast years",
    min_value=3,
    max_value=10,
    value=5,
)

growth = st.sidebar.slider(
    "Forecast FCF growth",
    min_value=-0.20,
    max_value=0.40,
    value=0.06,
    step=0.01,
    format="%.2f",
)

terminal_growth = st.sidebar.slider(
    "Terminal growth",
    min_value=-0.01,
    max_value=0.06,
    value=0.025,
    step=0.0025,
    format="%.3f",
)

risk_free = st.sidebar.slider(
    "Risk-free rate",
    0.0, 0.10, 0.04, 0.005,
)

beta = st.sidebar.number_input(
    "Beta",
    min_value=0.0,
    max_value=4.0,
    value=1.0,
    step=0.1,
)

equity_risk_premium = st.sidebar.slider(
    "Equity risk premium",
    0.0, 0.12, 0.05, 0.005,
)

cost_of_debt = st.sidebar.slider(
    "Pre-tax cost of debt",
    0.0, 0.20, 0.05, 0.005,
)

tax_rate = st.sidebar.slider(
    "Tax rate",
    0.0, 0.50, 0.21, 0.01,
)


# ============================================================
# LOAD COMPANY DATA
# ============================================================
if not ticker:
    st.info("Enter a stock ticker in the sidebar.")
    st.stop()

with st.spinner(f"Loading data for {ticker}..."):
    data, data_message = load_data(
        ticker,
        csv_only=(data_mode == "CSV only"),
    )

if data is None:
    st.error(data_message)
    st.stop()

st.caption(f"Data source: {data_message}")

annual, fundamentals = calculate_fundamentals(data)
info = data.get("info", {}) or {}

price = number(data.get("price"))

shares = number(info.get("sharesOutstanding"))
market_cap = number(info.get("marketCap"))

if not np.isfinite(shares) and np.isfinite(market_cap) and price > 0:
    shares = market_cap / price

if not np.isfinite(market_cap) and price > 0 and shares > 0:
    market_cap = price * shares

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

debt = first_valid(
    fundamentals["debt"],
    info.get("totalDebt"),
    0.0,
)

base_fcf = fundamentals["base_fcf"]


# ============================================================
# PART 1 — FUNDAMENTALS & FREE CASH FLOW
# ============================================================
st.header("Part 1 — Fundamentals & Free Cash Flow")

st.write(
    "Annual income statement, balance sheet, and cash-flow data "
    "are transposed into date rows and sorted chronologically. "
    "Missing statement labels are handled safely."
)

col1, col2, col3 = st.columns(3)

col1.metric("Current market price", money(price))
col2.metric(
    "Current ratio",
    (
        f"{fundamentals['current_ratio']:.2f}"
        if np.isfinite(fundamentals["current_ratio"])
        else "N/A"
    ),
)
col3.metric(
    "Debt / equity",
    (
        f"{fundamentals['debt_equity']:.2f}"
        if np.isfinite(fundamentals["debt_equity"])
        else "N/A"
    ),
)

st.subheader("Last four annual periods")

if annual.empty:
    st.warning("No annual fundamentals are available.")
else:
    display = annual.tail(4).copy()

    for col in [
        "Revenue",
        "Net Income",
        "Operating Cash Flow",
        "CapEx",
        "FCF",
    ]:
        display[col] = display[col].map(
            lambda x: "N/A" if pd.isna(x) else f"${x:,.0f}"
        )

    for col in ["Net Margin", "FCF Margin"]:
        display[col] = display[col].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.1%}"
        )

    display.index = display.index.strftime("%Y")
    st.dataframe(display, use_container_width=True)

if np.isfinite(base_fcf):
    st.metric("Three-year average FCF", money(base_fcf))
else:
    st.warning(
        "At least three annual FCF observations are required "
        "for the three-year average used in the DCF."
    )

st.caption(
    "FCF = Operating Cash Flow − |CapEx|. "
    "Missing data is shown as N/A where possible."
)

with st.expander("View the three financial statements"):
    st.subheader("Income statement")
    st.dataframe(data["income"], use_container_width=True)

    st.subheader("Balance sheet")
    st.dataframe(data["balance"], use_container_width=True)

    st.subheader("Cash-flow statement")
    st.dataframe(data["cashflow"], use_container_width=True)


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
        "Market capitalization is unavailable. Using CAPM cost "
        "of equity as the starting discount rate."
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
    st.error(
        "DCF cannot be calculated: at least three annual FCF "
        "observations are required."
    )
elif terminal_growth >= wacc:
    st.error(
        "Terminal growth must be lower than WACC. "
        "Reduce terminal growth or increase WACC."
    )
else:
    dcf = calculate_dcf(
        base_fcf,
        growth,
        terminal_growth,
        wacc,
        years,
        cash,
        debt,
        shares,
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
    st.dataframe(
        forecast.style.format({
            "Projected FCF": "${:,.0f}",
            "PV of FCF": "${:,.0f}",
        }),
        use_container_width=True,
    )

else:
    intrinsic = np.nan
    dcf_value = np.nan


# ============================================================
# PART 3 — COMPARABLE COMPANY ANALYSIS
# ============================================================
i wanna add MRK in the box and tell me how to fix the code: # ============================================================
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
            peer_info = yf.Ticker(peer).info or {}

            peer_rows.append({
                "Ticker": peer,
                "P/E": number(peer_info.get("trailingPE")),
                "EV/EBITDA": number(peer_info.get("enterpriseToEbitda")),
                "P/S": number(
                    peer_info.get("priceToSalesTrailing12Months")
                ),
            })
        except Exception:
            peer_rows.append({
                "Ticker": peer,
                "P/E": np.nan,
                "EV/EBITDA": np.nan,
                "P/S": np.nan,
            })

peers = pd.DataFrame(
    peer_rows,
    columns=["Ticker", "P/E", "EV/EBITDA", "P/S"],
)

peer_medians = peers[["P/E", "EV/EBITDA", "P/S"]].median(
    numeric_only=True
) if not peers.empty else pd.Series(dtype=float)

if not peers.empty:
    median_row = {"Ticker": "PEER MEDIAN"}

    for col in ["P/E", "EV/EBITDA", "P/S"]:
        values = peers[col].dropna()
        median_row[col] = values.median() if not values.empty else np.nan

    peers_display = pd.concat(
        [peers, pd.DataFrame([median_row])],
        ignore_index=True,
    )

    def highlight_median(row):
        if row["Ticker"] == "PEER MEDIAN":
            return [
                "background-color: #d9ead3; font-weight: bold;"
            ] * len(row)
        return [""] * len(row)

    st.subheader("Peer multiples and median")

    st.dataframe(
        peers_display.style
        .format({
            "P/E": "{:.2f}x",
            "EV/EBITDA": "{:.2f}x",
            "P/S": "{:.2f}x",
        }, na_rep="N/A")
        .apply(highlight_median, axis=1),
        use_container_width=True,
    )

    # Company's corresponding metrics.
    eps = number(info.get("trailingEps"))

    sales = first_valid(
        info.get("totalRevenue"),
        latest_value(
            data["income"],
            "Total Revenue", "Operating Revenue", "Revenue",
        ),
    )

    ebitda = number(info.get("ebitda"))

    if not np.isfinite(ebitda):
        operating_income = latest_value(
            data["income"], "Operating Income", "EBIT"
        )
        depreciation = latest_value(
            data["cashflow"],
            "Depreciation And Amortization",
            "Depreciation Amortization Depletion",
        )

        if np.isfinite(operating_income) and np.isfinite(depreciation):
            ebitda = operating_income + depreciation

    pe_median = number(peer_medians.get("P/E"))
    ev_ebitda_median = number(peer_medians.get("EV/EBITDA"))
    ps_median = number(peer_medians.get("P/S"))

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
        "Peer Median Multiple": [
            pe_median, ev_ebitda_median, ps_median
        ],
        "Company Metric": [eps, ebitda, sales],
        "Implied Value / Share": [
            pe_value, ev_ebitda_value, ps_value
        ],
    })

    st.subheader("Implied value from each peer multiple")

    st.dataframe(
        implied_values.style.format({
            "Peer Median Multiple": "{:.2f}x",
            "Company Metric": "{:,.2f}",
            "Implied Value / Share": "${:,.2f}",
        }, na_rep="N/A"),
        use_container_width=True,
    )

    valid_comp_values = implied_values[
        "Implied Value / Share"
    ].dropna()

    comps_value = (
        valid_comp_values.median()
        if not valid_comp_values.empty else np.nan
    )

    st.subheader("Valuation comparison")

    comparison = pd.DataFrame({
        "Method": [
            "Comparable-company value",
            "DCF intrinsic value",
            "Current market price",
        ],
        "Value per Share": [
            comps_value,
            dcf_value,
            price,
        ],
    })

    st.dataframe(
        comparison.style.format(
            {"Value per Share": "${:,.2f}"},
            na_rep="N/A",
        ),
        use_container_width=True,
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
            text=[
                money(comps_value),
                money(dcf_value),
                money(price),
            ],
            textposition="outside",
        ))
        comparison_chart.update_layout(
            title="Comps vs. DCF vs. Market Price",
            yaxis_title="Value per share ($)",
            showlegend=False,
        )
        st.plotly_chart(comparison_chart, use_container_width=True)

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

else:
    comps_value = np.nan
    st.info("Enter peer tickers to calculate comparable-company valuations.")
# ============================================================
# PART 4 — TWO-WAY SENSITIVITY ANALYSIS
# ============================================================
st.header("Part 4 — Sensitivity Analysis")

if dcf is None:
    st.warning(
        "Sensitivity analysis requires a valid base-case DCF. "
        "Check the FCF data and ensure terminal growth is below WACC."
    )
else:
    # The selected assumptions are explicitly included in both axes.
    wacc_values = sorted(set(
        [round(float(wacc), 6)]
        + [
            round(float(wacc + delta), 6)
            for delta in [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03]
            if 0.01 <= wacc + delta <= 0.30
        ]
    ))

    terminal_values = sorted(set(
        [round(float(terminal_growth), 6)]
        + [
            round(float(terminal_growth + delta), 6)
            for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]
            if -0.05 <= terminal_growth + delta <= 0.15
        ]
    ))

    sensitivity = pd.DataFrame(
        index=wacc_values,
        columns=terminal_values,
        dtype=float,
    )

    for rate in wacc_values:
        for terminal_g in terminal_values:
            result = calculate_dcf(
                base_fcf,
                growth,
                terminal_g,
                rate,
                years,
                cash,
                debt,
                shares,
            )

            sensitivity.loc[rate, terminal_g] = (
                result["intrinsic"]
                if result is not None else np.nan
            )

    # Build the heatmap with the exact base-case cell marked.
    x_labels = [f"{g:.2%}" for g in terminal_values]
    y_labels = [f"{r:.2%}" for r in wacc_values]

    cell_text = []

    for rate in wacc_values:
        row = []

        for terminal_g in terminal_values:
            value = sensitivity.loc[rate, terminal_g]

            if np.isclose(rate, wacc) and np.isclose(
                terminal_g, terminal_growth
            ):
                row.append(
                    f"BASE<br>{money(value)}"
                    if np.isfinite(value) else "BASE<br>N/A"
                )
            else:
                row.append(
                    money(value) if np.isfinite(value) else "N/A"
                )

        cell_text.append(row)

    heatmap = go.Figure(data=go.Heatmap(
        z=sensitivity.to_numpy(dtype=float),
        x=x_labels,
        y=y_labels,
        text=cell_text,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        colorbar={"title": "Value / share"},
        hovertemplate=(
            "WACC: %{y}<br>"
            "Terminal growth: %{x}<br>"
            "Intrinsic value: $%{z:,.2f}"
            "<extra></extra>"
        ),
    ))

    base_x = x_labels.index(f"{terminal_growth:.2%}")
    base_y = y_labels.index(f"{wacc:.2%}")

    heatmap.add_shape(
        type="rect",
        x0=base_x - 0.5,
        x1=base_x + 0.5,
        y0=base_y - 0.5,
        y1=base_y + 0.5,
        line={"color": "black", "width": 4},
    )

    heatmap.update_layout(
        title="Intrinsic Value Sensitivity — BASE cell outlined",
        xaxis_title="Terminal Growth",
        yaxis_title="WACC",
    )

    st.plotly_chart(heatmap, use_container_width=True)

    st.caption(
        "Rows are WACC; columns are terminal growth. "
        "The outlined BASE cell uses your selected assumptions."
    )

    
    # Explain which input moves the value most, holding the other
    # input at its base-case level.
    base_wacc_key = min(
        sensitivity.index,
        key=lambda value: abs(float(value) - float(wacc)),
    )

    base_terminal_key = min(
        sensitivity.columns,
        key=lambda value: abs(float(value) - float(terminal_growth)),
    )

    # WACC impact while holding terminal growth at its base case.
    wacc_range_values = [
        float(sensitivity.loc[rate, base_terminal_key])
        for rate in sensitivity.index
        if base_terminal_key < rate
        and np.isfinite(sensitivity.loc[rate, base_terminal_key])
    ]

    # Terminal-growth impact while holding WACC at its base case.
    growth_range_values = [
        float(sensitivity.loc[base_wacc_key, terminal_g])
        for terminal_g in sensitivity.columns
        if terminal_g < base_wacc_key
        and np.isfinite(sensitivity.loc[base_wacc_key, terminal_g])
    ]

    wacc_impact = (
        max(wacc_range_values) - min(wacc_range_values)
        if len(wacc_range_values) >= 2
        else np.nan
    )

    terminal_impact = (
        max(growth_range_values) - min(growth_range_values)
        if len(growth_range_values) >= 2
        else np.nan
    )


    st.subheader("Sensitivity interpretation")

    st.write(
        f"At the base-case assumptions, intrinsic value is "
        f"**{money(dcf_value)} per share**."
    )

    if np.isfinite(wacc_impact) and np.isfinite(terminal_impact):
        if wacc_impact > terminal_impact:
            st.write(
                f"Across the tested grid, **WACC** moves valuation more "
                f"({money(wacc_impact)} range versus "
                f"{money(terminal_impact)} for terminal growth), "
                "holding the other assumption at its base value."
            )
        elif terminal_impact > wacc_impact:
            st.write(
                f"Across the tested grid, **terminal growth** moves "
                f"valuation more ({money(terminal_impact)} range versus "
                f"{money(wacc_impact)} for WACC), holding the other "
                "assumption at its base value."
            )
        else:
            st.write(
                "WACC and terminal growth have equal valuation ranges "
                "across the tested grid."
            )

    if np.isfinite(price):
        st.write(
            f"Cells with intrinsic value above the market price "
            f"({money(price)}) suggest potential undervaluation under "
            "those assumptions. Cells below the market price suggest "
            "potential overvaluation. This is a model comparison, "
            "not a prediction."
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
    "Bear": {
        "growth": -0.02,
        "terminal_growth": 0.015,
        "wacc_adjustment": 0.02,
    },
    "Base": {
        "growth": 0.06,
        "terminal_growth": 0.025,
        "wacc_adjustment": 0.00,
    },
    "Bull": {
        "growth": 0.12,
        "terminal_growth": 0.03,
        "wacc_adjustment": -0.01,
    },
}

selected_scenario = st.selectbox(
    "Choose a scenario to inspect",
    ["Base", "Bull", "Bear"],
    index=0,
)

selected = scenario_presets[selected_scenario]

st.write(f"**Selected scenario: {selected_scenario}**")

scenario_cols = st.columns(3)
scenario_cols[0].metric(
    "Scenario FCF growth",
    percent(selected["growth"]),
)
scenario_cols[1].metric(
    "Scenario terminal growth",
    percent(selected["terminal_growth"]),
)
scenario_wacc = wacc + selected["wacc_adjustment"]
scenario_cols[2].metric(
    "Scenario WACC",
    percent(scenario_wacc),
)

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

    scenario_value = (
        result["intrinsic"] if result is not None else np.nan
    )

    scenario_upside = (
        scenario_value / price - 1
        if np.isfinite(scenario_value)
        and np.isfinite(price)
        and price > 0
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
        return [
            "background-color: #d9ead3; font-weight: bold;"
        ] * len(row)
    return [""] * len(row)

st.dataframe(
    scenario_display.style.apply(highlight_selected, axis=1),
    use_container_width=True,
)

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

st.plotly_chart(scenario_chart, use_container_width=True)

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
