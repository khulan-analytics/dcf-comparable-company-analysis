
# ============================================================
# MRK EQUITY VALUATION DASHBOARD
# Part 1: Fundamentals and Free Cash Flow
# Part 2: DCF Valuation and CAPM / WACC
# Part 3: Comparable Company Analysis
# Part 4: Sensitivity Analysis
# Part 5: Bull / Base / Bear Scenarios
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MRK Equity Valuation",
    page_icon="📊",
    layout="wide"
)

st.title("MRK | Equity Valuation Dashboard")

st.caption(
    "Fundamental Analysis | DCF | CAPM / WACC | Comparable Companies"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.08);
        padding: 12px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def pick(df, *labels, default=np.nan):
    """Return the first matching column label, otherwise N/A."""

    for label in labels:
        if label in df.columns:
            return df[label]

    return pd.Series(default, index=df.index)


def number(value, default=np.nan):
    """Safely convert values to float."""

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return default


def latest_value(df, *labels):
    """Return the latest available value from a statement."""

    if df.empty:
        return np.nan

    result = pick(df, *labels)

    if isinstance(result, pd.Series):
        result = result.dropna()

        if not result.empty:
            return number(result.iloc[-1])

    return np.nan


def ratio(a, b):
    a = number(a)
    b = number(b)

    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan

    return a / b


def money(value):
    if pd.isna(value):
        return "N/A"

    return f"${value:,.2f}"


def billions(value):
    if pd.isna(value):
        return "N/A"

    return f"${value / 1e9:,.2f}B"


def percent(value):
    if pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def read_uploaded_csv(uploaded_file):
    """Read a cached statement CSV."""

    if uploaded_file is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(
            BytesIO(uploaded_file.getvalue()),
            index_col=0
        )

        df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        return df

    except Exception:
        return pd.DataFrame()


def load_statement(stock, statement_type, uploaded_file=None):
    """Try Yahoo Finance first, then the uploaded CSV."""

    try:

        if statement_type == "income":
            df = stock.income_stmt

        elif statement_type == "balance":
            df = stock.balance_sheet

        elif statement_type == "cashflow":
            df = stock.cashflow

        else:
            df = pd.DataFrame()

        if df is not None and not df.empty:

            df = df.T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            return df, "Yahoo Finance"

    except Exception:
        pass

    df = read_uploaded_csv(uploaded_file)

    if not df.empty:
        return df, "Uploaded CSV"

    return pd.DataFrame(), "Unavailable"


@st.cache_data(ttl=3600, show_spinner=False)
def load_yahoo_data(ticker):
    """Download company information and financial statements."""

    stock = yf.Ticker(ticker)

    result = {
        "income": pd.DataFrame(),
        "balance": pd.DataFrame(),
        "cashflow": pd.DataFrame(),
        "info": {}
    }

    try:
        result["income"] = stock.income_stmt

    except Exception:
        pass

    try:
        result["balance"] = stock.balance_sheet

    except Exception:
        pass

    try:
        result["cashflow"] = stock.cashflow

    except Exception:
        pass

    try:
        result["info"] = stock.info

    except Exception:
        pass

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_price(ticker):

    try:
        history = yf.Ticker(ticker).history(
            period="1mo",
            auto_adjust=True
        )

        if not history.empty:
            return number(history["Close"].dropna().iloc[-1])

    except Exception:
        pass

    return np.nan


def calculate_dcf(
    base_fcf,
    growth,
    terminal_growth,
    years,
    wacc,
    cash,
    debt,
    shares
):
    """Calculate FCFF discounted cash flow valuation."""

    if wacc <= terminal_growth:
        raise ValueError("Terminal growth must be below WACC.")

    if wacc <= -1:
        raise ValueError("WACC must be greater than -100%.")

    if years < 1:
        raise ValueError("Forecast years must be positive.")

    if pd.isna(base_fcf) or base_fcf <= 0:
        raise ValueError("A positive base FCF is required.")

    if pd.isna(shares) or shares <= 0:
        raise ValueError("Valid shares outstanding are required.")

    forecast_years = np.arange(1, years + 1)

    projected = base_fcf * (1 + growth) ** forecast_years

    pv = projected / (1 + wacc) ** forecast_years

    terminal = (
        projected[-1] * (1 + terminal_growth)
        / (wacc - terminal_growth)
    )

    pv_terminal = terminal / (1 + wacc) ** years

    enterprise = pv.sum() + pv_terminal

    equity = enterprise + cash - debt

    intrinsic = equity / shares

    terminal_pct = (
        pv_terminal / enterprise
        if enterprise != 0 else np.nan
    )

    return {
        "projected": projected,
        "pv": pv,
        "terminal": terminal,
        "pv_terminal": pv_terminal,
        "enterprise": enterprise,
        "equity": equity,
        "intrinsic": intrinsic,
        "terminal_pct": terminal_pct
    }


def peer_median(peer_df, column):

    values = peer_df[column].dropna()

    values = values[
        np.isfinite(values) & (values > 0)
    ]

    if values.empty:
        return np.nan

    return values.median()


# ============================================================
# 2. SIDEBAR
# ============================================================

st.sidebar.title("Valuation Assumptions")

ticker = st.sidebar.text_input(
    "Company Ticker",
    value="MRK"
).strip().upper()

if not ticker:
    st.error("Please enter a ticker.")
    st.stop()


# ------------------------------------------------------------
# CSV FALLBACK
# ------------------------------------------------------------

st.sidebar.subheader("Financial Data Backup")

income_csv = st.sidebar.file_uploader(
    "Income Statement CSV",
    type="csv"
)

balance_csv = st.sidebar.file_uploader(
    "Balance Sheet CSV",
    type="csv"
)

cashflow_csv = st.sidebar.file_uploader(
    "Cash Flow CSV",
    type="csv"
)


# ------------------------------------------------------------
# SCENARIOS
# ------------------------------------------------------------

st.sidebar.subheader("Scenario")

scenario = st.sidebar.selectbox(
    "Choose Scenario",
    ["Base", "Bull", "Bear"]
)

scenario_settings = {
    "Bull": {
        "growth": 0.08,
        "terminal": 0.035
    },
    "Base": {
        "growth": 0.04,
        "terminal": 0.025
    },
    "Bear": {
        "growth": 0.00,
        "terminal": 0.015
    }
}

default_growth = scenario_settings[scenario]["growth"]
default_terminal = scenario_settings[scenario]["terminal"]


# ------------------------------------------------------------
# DCF ASSUMPTIONS
# ------------------------------------------------------------

st.sidebar.subheader("DCF Assumptions")

growth = st.sidebar.slider(
    "Forecast FCF Growth",
    -0.10,
    0.20,
    default_growth,
    0.005,
    format="%.2%"
)

terminal_growth = st.sidebar.slider(
    "Terminal Growth",
    0.000,
    0.050,
    default_terminal,
    0.005,
    format="%.2%"
)

years = st.sidebar.slider(
    "Forecast Years",
    3,
    10,
    5
)


# ------------------------------------------------------------
# CAPM / WACC ASSUMPTIONS
# ------------------------------------------------------------

st.sidebar.subheader("CAPM / WACC")

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate",
    min_value=0.0,
    max_value=0.15,
    value=0.04,
    step=0.005
)

market_return = st.sidebar.number_input(
    "Expected Market Return",
    min_value=0.0,
    max_value=0.25,
    value=0.10,
    step=0.005
)

manual_beta = st.sidebar.number_input(
    "Beta Override (0 = Yahoo Beta)",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.05
)

manual_debt_cost = st.sidebar.number_input(
    "Pre-Tax Cost of Debt",
    min_value=0.0,
    max_value=0.20,
    value=0.055,
    step=0.005
)

tax_rate = st.sidebar.number_input(
    "Corporate Tax Rate",
    min_value=0.0,
    max_value=0.50,
    value=0.21,
    step=0.01
)

wacc_adjustment = st.sidebar.slider(
    "WACC Adjustment",
    -0.03,
    0.03,
    0.0,
    0.0025,
    format="%.2%"
)


# ------------------------------------------------------------
# PEER INPUT
# ------------------------------------------------------------

st.sidebar.subheader("Comparable Companies")

peer_input = st.sidebar.text_input(
    "Peer Tickers",
    value="ABBV, BMY, PFE, JNJ, GILD"
)

peers = list(dict.fromkeys([
    p.strip().upper()
    for p in peer_input.split(",")
    if p.strip()
]))

peers = [p for p in peers if p != ticker]


# ============================================================
# 3. LOAD COMPANY DATA
# ============================================================

with st.spinner(f"Loading financial data for {ticker}..."):

    yahoo_data = load_yahoo_data(ticker)

    stock = yf.Ticker(ticker)

    income_yahoo = yahoo_data["income"]
    balance_yahoo = yahoo_data["balance"]
    cashflow_yahoo = yahoo_data["cashflow"]

    info = yahoo_data["info"]

    # Yahoo Finance statements are transposed
    # and sorted by fiscal date.

    if not income_yahoo.empty:
        income_yahoo = income_yahoo.T
        income_yahoo.index = pd.to_datetime(income_yahoo.index)
        income_yahoo = income_yahoo.sort_index()

    if not balance_yahoo.empty:
        balance_yahoo = balance_yahoo.T
        balance_yahoo.index = pd.to_datetime(balance_yahoo.index)
        balance_yahoo = balance_yahoo.sort_index()

    if not cashflow_yahoo.empty:
        cashflow_yahoo = cashflow_yahoo.T
        cashflow_yahoo.index = pd.to_datetime(cashflow_yahoo.index)
        cashflow_yahoo = cashflow_yahoo.sort_index()


# Use uploaded files if Yahoo data is unavailable

if income_yahoo.empty:
    income, income_source = load_statement(
        stock, "income", income_csv
    )
else:
    income, income_source = income_yahoo, "Yahoo Finance"

if balance_yahoo.empty:
    balance, balance_source = load_statement(
        stock, "balance", balance_csv
    )
else:
    balance, balance_source = balance_yahoo, "Yahoo Finance"

if cashflow_yahoo.empty:
    cashflow, cashflow_source = load_statement(
        stock, "cashflow", cashflow_csv
    )
else:
    cashflow, cashflow_source = cashflow_yahoo, "Yahoo Finance"


# Current price

market_price = number(info.get("currentPrice"))

if pd.isna(market_price):
    market_price = number(info.get("regularMarketPrice"))

if pd.isna(market_price):
    market_price = load_market_price(ticker)


# Company beta

beta = number(info.get("beta"))

if manual_beta > 0:
    beta = manual_beta


# ============================================================
# PART 1 — FUNDAMENTALS & FREE CASH FLOW
# ============================================================

st.header("Part 1 | Fundamentals & Free Cash Flow")

st.caption(
    f"Income: {income_source} | "
    f"Balance Sheet: {balance_source} | "
    f"Cash Flow: {cashflow_source}"
)

# ------------------------------------------------------------
# INCOME STATEMENT
# ------------------------------------------------------------

revenue = pick(
    income,
    "Total Revenue",
    "Operating Revenue"
)

net_income = pick(
    income,
    "Net Income",
    "Net Income Common Stockholders"
)

operating_income = pick(
    income,
    "Operating Income"
)

# ------------------------------------------------------------
# CASH FLOW STATEMENT
# ------------------------------------------------------------

operating_cash_flow = pick(
    cashflow,
    "Operating Cash Flow",
    "Cash Flow From Continuing Operating Activities"
)

capex = pick(
    cashflow,
    "Capital Expenditure",
    "Capital Expenditures"
)

# FCF = Operating Cash Flow - Absolute CapEx

fcf = operating_cash_flow - capex.abs()


# ------------------------------------------------------------
# BALANCE SHEET
# ------------------------------------------------------------

current_assets = latest_value(
    balance,
    "Current Assets"
)

current_liabilities = latest_value(
    balance,
    "Current Liabilities"
)

cash = latest_value(
    balance,
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments"
)

debt = latest_value(
    balance,
    "Total Debt"
)

# If total debt is unavailable, calculate from components.

if pd.isna(debt):

    current_debt = latest_value(
        balance,
        "Current Debt",
        "Current Debt And Capital Lease Obligation"
    )

    long_term_debt = latest_value(
        balance,
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation"
    )

    components = [
        x for x in [current_debt, long_term_debt]
        if pd.notna(x)
    ]

    if components:
        debt = sum(components)

equity_book = latest_value(
    balance,
    "Stockholders Equity",
    "Total Stockholder Equity",
    "Total Equity Gross Minority Interest"
)

# Avoid silently treating missing balance sheet values as zero.

if pd.isna(cash):
    st.warning("Cash unavailable. DCF cash bridge cannot be verified.")
    cash = 0.0

if pd.isna(debt):
    st.warning("Debt unavailable. Zero debt is used provisionally.")
    debt = 0.0

current_ratio = ratio(
    current_assets,
    current_liabilities
)

debt_equity = ratio(
    debt,
    equity_book
)


# ------------------------------------------------------------
# FOUR-YEAR FINANCIAL TABLE
# ------------------------------------------------------------

fundamentals = pd.concat(
    [
        revenue.rename("Revenue"),
        net_income.rename("Net Income"),
        operating_cash_flow.rename("Operating Cash Flow"),
        capex.rename("Capital Expenditure"),
        fcf.rename("Free Cash Flow")
    ],
    axis=1
)

fundamentals = fundamentals.sort_index(
    ascending=False
).head(4)

fundamentals["Revenue Growth"] = (
    fundamentals["Revenue"].pct_change(-1)
)

fundamentals["Net Margin"] = (
    fundamentals["Net Income"]
    / fundamentals["Revenue"]
)

fundamentals["FCF Margin"] = (
    fundamentals["Free Cash Flow"]
    / fundamentals["Revenue"]
)

st.subheader("Historical Financial Performance")

display_fundamentals = fundamentals.copy()

money_columns = [
    "Revenue",
    "Net Income",
    "Operating Cash Flow",
    "Capital Expenditure",
    "Free Cash Flow"
]

for col in money_columns:
    display_fundamentals[col] = (
        display_fundamentals[col] / 1e9
    )

st.dataframe(
    display_fundamentals.style.format({
        "Revenue": "${:,.2f}B",
        "Net Income": "${:,.2f}B",
        "Operating Cash Flow": "${:,.2f}B",
        "Capital Expenditure": "${:,.2f}B",
        "Free Cash Flow": "${:,.2f}B",
        "Revenue Growth": "{:.2%}",
        "Net Margin": "{:.2%}",
        "FCF Margin": "{:.2%}"
    }, na_rep="N/A"),
    width="stretch"
)

st.caption(
    "Financial statement values are in USD billions. "
    "Margins and growth are percentages."
)


# ------------------------------------------------------------
# FUNDAMENTAL METRICS
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Market Price",
    money(market_price)
)

col2.metric(
    "Current Ratio",
    f"{current_ratio:.2f}" if pd.notna(current_ratio) else "N/A"
)

col3.metric(
    "Debt / Equity",
    f"{debt_equity:.2f}" if pd.notna(debt_equity) else "N/A"
)

col4.metric(
    "Company Beta",
    f"{beta:.2f}" if pd.notna(beta) else "N/A"
)


# ------------------------------------------------------------
# FUNDAMENTAL CHART
# ------------------------------------------------------------

st.subheader("Financial Performance Trend")

chart_data = fundamentals[
    ["Revenue", "Net Income", "Free Cash Flow"]
].sort_index()

fig_fundamentals = go.Figure()

for column in chart_data.columns:

    fig_fundamentals.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data[column] / 1e9,
            mode="lines+markers",
            name=column
        )
    )

fig_fundamentals.update_layout(
    template="plotly_dark",
    height=430,
    xaxis_title="Fiscal Year",
    yaxis_title="USD Billions",
    hovermode="x unified"
)

st.plotly_chart(
    fig_fundamentals,
    width="stretch"
)


# ------------------------------------------------------------
# THREE-YEAR AVERAGE FCF
# ------------------------------------------------------------

st.subheader("Three-Year Average Free Cash Flow")

historical_fcf = fcf.dropna().sort_index()

recent_fcf = historical_fcf.tail(3)

base_fcf = (
    recent_fcf.mean()
    if not recent_fcf.empty
    else np.nan
)

col1, col2 = st.columns(2)

col1.metric(
    "3-Year Average FCF",
    billions(base_fcf)
)

col2.metric(
    "Latest FCF",
    billions(
        historical_fcf.iloc[-1]
        if not historical_fcf.empty
        else np.nan
    )
)

if len(recent_fcf) < 3:

    st.warning(
        f"Only {len(recent_fcf)} usable FCF observations found. "
        "The average uses the available years."
    )


# ============================================================
# PART 2 — DCF VALUATION
# ============================================================

st.header("Part 2 | DCF Valuation")

st.subheader("CAPM — Cost of Equity")

# ------------------------------------------------------------
# MARKET CAPITALIZATION AND SHARES
# ------------------------------------------------------------

shares = number(
    info.get("sharesOutstanding")
)

market_cap = number(
    info.get("marketCap")
)

if pd.isna(market_cap):

    if pd.notna(shares) and pd.notna(market_price):
        market_cap = shares * market_price

# Fallback shares calculation

if pd.isna(shares):

    if (
        pd.notna(market_cap)
        and pd.notna(market_price)
        and market_price > 0
    ):
        shares = market_cap / market_price


# ------------------------------------------------------------
# CAPM AND WACC
# ------------------------------------------------------------

wacc_available = False

if pd.notna(beta):

    cost_of_equity = (
        risk_free_rate
        + beta * (market_return - risk_free_rate)
    )

    after_tax_cost_debt = (
        manual_debt_cost * (1 - tax_rate)
    )

    if (
        pd.notna(market_cap)
        and market_cap > 0
    ):

        total_capital = market_cap + debt

        if total_capital > 0:

            equity_weight = market_cap / total_capital

            debt_weight = debt / total_capital

            base_wacc = (
                equity_weight * cost_of_equity
                + debt_weight * after_tax_cost_debt
            )

            adjusted_wacc = (
                base_wacc + wacc_adjustment
            )

            wacc_available = True

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Cost of Equity",
                percent(cost_of_equity)
            )

            c2.metric(
                "After-Tax Cost of Debt",
                percent(after_tax_cost_debt)
            )

            c3.metric(
                "Baseline WACC",
                percent(base_wacc)
            )

            c4.metric(
                "Adjusted WACC",
                percent(adjusted_wacc)
            )

            st.subheader("Capital Structure")

            capital_structure = pd.DataFrame({
                "Component": ["Equity", "Debt"],
                "Market Value": [market_cap, debt],
                "Weight": [equity_weight, debt_weight]
            })

            st.dataframe(
                capital_structure.style.format({
                    "Market Value": "${:,.0f}",
                    "Weight": "{:.2%}"
                }),
                width="stretch"
            )

            st.caption(
                "Debt is represented by its balance sheet value, "
                "used as a proxy for market value. Equity uses market capitalization."
            )

if not wacc_available:

    st.warning(
        "CAPM/WACC is unavailable. Check the company beta, "
        "market capitalization and debt data. "
        "A beta override is available in the sidebar."
    )


# ------------------------------------------------------------
# DCF CALCULATION
# ------------------------------------------------------------

dcf = None
intrinsic_value = np.nan
upside = np.nan

if not wacc_available:

    st.info(
        "DCF is waiting for valid CAPM/WACC inputs. "
        "The fundamentals and comparable-company sections can still run."
    )

elif adjusted_wacc <= terminal_growth:

    st.error(
        "Invalid DCF assumptions: terminal growth must be strictly "
        "below WACC. Change the assumptions in the sidebar."
    )

elif pd.isna(base_fcf) or base_fcf <= 0:

    st.error(
        "DCF requires positive historical average FCF. "
        "Review MRK's cash flow statements before proceeding."
    )

elif pd.isna(shares) or shares <= 0:

    st.error(
        "Shares outstanding are unavailable. DCF per-share valuation "
        "cannot be calculated."
    )

else:

    try:

        dcf = calculate_dcf(
            base_fcf=base_fcf,
            growth=growth,
            terminal_growth=terminal_growth,
            years=years,
            wacc=adjusted_wacc,
            cash=cash,
            debt=debt,
            shares=shares
        )

        intrinsic_value = dcf["intrinsic"]

        if pd.notna(market_price) and market_price > 0:

            upside = (
                intrinsic_value / market_price - 1
            )

        st.subheader("DCF Forecast")

        forecast_table = pd.DataFrame({
            "Year": np.arange(1, years + 1),
            "Projected FCF": dcf["projected"],
            "Discount Factor": (
                1 / (1 + adjusted_wacc)
                ** np.arange(1, years + 1)
            ),
            "PV of FCF": dcf["pv"]
        })

        st.dataframe(
            forecast_table.style.format({
                "Projected FCF": "${:,.0f}",
                "Discount Factor": "{:.4f}",
                "PV of FCF": "${:,.0f}"
            }),
            width="stretch"
        )

        # Enterprise value bridge

        st.subheader("Enterprise Value to Equity Value")

        dcf_bridge = pd.DataFrame({
            "Component": [
                "PV of Forecast FCF",
                "PV of Terminal Value",
                "Enterprise Value",
                "Add: Cash",
                "Less: Debt",
                "Equity Value",
                "Shares Outstanding"
            ],
            "Value": [
                dcf["pv"].sum(),
                dcf["pv_terminal"],
                dcf["enterprise"],
                cash,
                -debt,
                dcf["equity"],
                shares
            ]
        })

        st.dataframe(
            dcf_bridge.style.format({
                "Value": "${:,.0f}"
            }),
            width="stretch"
        )

        # Valuation metrics

        st.subheader("DCF Valuation Results")

        v1, v2, v3, v4 = st.columns(4)

        v1.metric(
            "Intrinsic Value",
            money(intrinsic_value)
        )

        v2.metric(
            "Market Price",
            money(market_price)
        )

        v3.metric(
            "Upside / Downside",
            percent(upside)
        )

        v4.metric(
            "Terminal Value / EV",
            percent(dcf["terminal_pct"])
        )

        # DCF chart

        valuation_chart = pd.DataFrame({
            "Method": [
                "DCF Intrinsic Value",
                "Market Price"
            ],
            "Share Price": [
                intrinsic_value,
                market_price
            ]
        }).dropna()

        fig_dcf = px.bar(
            valuation_chart,
            x="Method",
            y="Share Price",
            text="Share Price",
            title="DCF Intrinsic Value vs. Market Price",
            template="plotly_dark"
        )

        fig_dcf.update_traces(
            texttemplate="$%{text:.2f}",
            textposition="outside"
        )

        st.plotly_chart(
            fig_dcf,
            width="stretch"
        )

    except ValueError as error:

        st.error(f"DCF calculation error: {error}")


# ============================================================
# PART 3 — COMPARABLE COMPANY ANALYSIS
# ============================================================

st.header("Part 3 | Comparable Company Analysis")

st.write(
    "Compare MRK with selected pharmaceutical peers using "
    "P/E, EV/EBITDA and P/S multiples."
)

peer_rows = []

with st.spinner("Loading comparable-company data..."):

    for peer in peers:

        try:

            peer_info = load_yahoo_data(peer)["info"]

            peer_rows.append({
                "Ticker": peer,
                "P/E": number(
                    peer_info.get("trailingPE")
                ),
                "EV/EBITDA": number(
                    peer_info.get("enterpriseToEbitda")
                ),
                "P/S": number(
                    peer_info.get("priceToSalesTrailing12Months")
                )
            })

        except Exception:

            peer_rows.append({
                "Ticker": peer,
                "P/E": np.nan,
                "EV/EBITDA": np.nan,
                "P/S": np.nan
            })


peer_df = pd.DataFrame(peer_rows)

median_pe = np.nan
median_ev_ebitda = np.nan
median_ps = np.nan

implied_pe = np.nan
implied_ev_ebitda = np.nan
implied_ps = np.nan

if not peer_df.empty:

    st.subheader("Peer Multiples")

    st.dataframe(
        peer_df.style.format({
            "P/E": "{:.2f}x",
            "EV/EBITDA": "{:.2f}x",
            "P/S": "{:.2f}x"
        }, na_rep="N/A"),
        width="stretch"
    )

    # Median calculations

    median_pe = peer_median(peer_df, "P/E")

    median_ev_ebitda = peer_median(
        peer_df, "EV/EBITDA"
    )

    median_ps = peer_median(peer_df, "P/S")

    median_row = pd.DataFrame({
        "Ticker": ["PEER MEDIAN"],
        "P/E": [median_pe],
        "EV/EBITDA": [median_ev_ebitda],
        "P/S": [median_ps]
    })

    peer_display = pd.concat(
        [peer_df, median_row],
        ignore_index=True
    )

    st.subheader("Peer Median")

    def highlight_median(row):

        if row["Ticker"] == "PEER MEDIAN":

            return [
                "background-color: #355E3B; color: white; "
                "font-weight: bold"
            ] * len(row)

        return [""] * len(row)

    st.dataframe(
        peer_display.style.format({
            "P/E": "{:.2f}x",
            "EV/EBITDA": "{:.2f}x",
            "P/S": "{:.2f}x"
        }, na_rep="N/A").apply(
            highlight_median,
            axis=1
        ),
        width="stretch"
    )

else:

    st.warning(
        "No peer data available. Check the peer tickers."
    )


# ------------------------------------------------------------
# COMPANY METRICS FOR COMPS
# ------------------------------------------------------------

company_eps = number(
    info.get("trailingEps")
)

company_ebitda = number(
    info.get("ebitda")
)

company_revenue = latest_value(
    income,
    "Total Revenue",
    "Operating Revenue"
)

# Fallback EBITDA calculation:
# EBITDA = Operating Income + Depreciation & Amortization

if pd.isna(company_ebitda):

    operating_income_latest = latest_value(
        income,
        "Operating Income"
    )

    depreciation = latest_value(
        cashflow,
        "Depreciation And Amortization",
        "Depreciation Amortization Depletion"
    )

    if (
        pd.notna(operating_income_latest)
        and pd.notna(depreciation)
    ):

        company_ebitda = (
            operating_income_latest + depreciation
        )


# ------------------------------------------------------------
# IMPLIED VALUATION
# ------------------------------------------------------------

# P/E: Equity multiple

if (
    pd.notna(median_pe)
    and pd.notna(company_eps)
    and median_pe > 0
    and company_eps > 0
):

    implied_pe = median_pe * company_eps


# EV/EBITDA: Enterprise value multiple

if (
    pd.notna(median_ev_ebitda)
    and pd.notna(company_ebitda)
    and pd.notna(shares)
    and median_ev_ebitda > 0
    and company_ebitda > 0
    and shares > 0
):

    implied_enterprise = (
        median_ev_ebitda * company_ebitda
    )

    implied_equity = (
        implied_enterprise + cash - debt
    )

    implied_ev_ebitda = (
        implied_equity / shares
    )


# P/S: Equity multiple

if (
    pd.notna(median_ps)
    and pd.notna(company_revenue)
    and pd.notna(shares)
    and median_ps > 0
    and shares > 0
):

    implied_ps = (
        median_ps * company_revenue / shares
    )


# ------------------------------------------------------------
# COMPS RESULTS
# ------------------------------------------------------------

st.subheader("Implied Share Price by Multiple")

comps_values = pd.DataFrame({
    "Valuation Method": [
        "P/E",
        "EV/EBITDA",
        "P/S"
    ],
    "Peer Median Multiple": [
        median_pe,
        median_ev_ebitda,
        median_ps
    ],
    "Implied Share Price": [
        implied_pe,
        implied_ev_ebitda,
        implied_ps
    ]
})

st.dataframe(
    comps_values.style.format({
        "Peer Median Multiple": "{:.2f}x",
        "Implied Share Price": "${:.2f}"
    }, na_rep="N/A"),
    width="stretch"
)


# ------------------------------------------------------------
# THREE-WAY VALUATION COMPARISON
# ------------------------------------------------------------

st.subheader("DCF vs. Comparable Valuation vs. Market Price")

valuation_comparison = pd.DataFrame({
    "Valuation Method": [
        "DCF",
        "P/E",
        "EV/EBITDA",
        "P/S",
        "Market Price"
    ],
    "Share Price": [
        intrinsic_value,
        implied_pe,
        implied_ev_ebitda,
        implied_ps,
        market_price
    ]
})

st.dataframe(
    valuation_comparison.style.format({
        "Share Price": "${:.2f}"
    }, na_rep="N/A"),
    width="stretch"
)

fig_comps = px.bar(
    valuation_comparison.dropna(),
    x="Valuation Method",
    y="Share Price",
    text="Share Price",
    title="MRK Valuation Comparison",
    template="plotly_dark"
)

fig_comps.update_traces(
    texttemplate="$%{text:.2f}",
    textposition="outside"
)

fig_comps.update_layout(
    xaxis_tickangle=-20,
    height=450
)

st.plotly_chart(
    fig_comps,
    width="stretch"
)

st.markdown("""
**Why the values differ:**

- **DCF:** Based on projected future free cash flows and discount rates.
- **P/E:** Reflects the market's valuation of earnings.
- **EV/EBITDA:** Values operating earnings before financing and taxes.
- **P/S:** Compares market capitalization with company sales.
- **Market Price:** Represents the current market's assessment of MRK.

Differences can reflect growth expectations, profitability, capital structure,
risk, accounting policies and market sentiment.
""")


# ============================================================
# PART 4 — SENSITIVITY ANALYSIS
# ============================================================

st.header("Part 4 | Sensitivity Analysis")

st.write(
    "The sensitivity analysis measures how DCF intrinsic value changes "
    "when WACC, terminal growth and forecast FCF growth assumptions change."
)


if dcf is not None:

    # --------------------------------------------------------
    # WACC x TERMINAL GROWTH
    # --------------------------------------------------------

    st.subheader("WACC vs. Terminal Growth")

    # Center the grid exactly on the base case.

    wacc_values = (
        adjusted_wacc
        + np.linspace(-0.02, 0.02, 9)
    )

    growth_values = (
        terminal_growth
        + np.linspace(-0.02, 0.02, 9)
    )

    sensitivity = pd.DataFrame(
        index=np.round(wacc_values * 100, 3),
        columns=np.round(growth_values * 100, 3),
        dtype=float
    )

    for w in wacc_values:

        for g in growth_values:

            if w <= g:
                continue

            result = calculate_dcf(
                base_fcf,
                growth,
                g,
                years,
                w,
                cash,
                debt,
                shares
            )

            sensitivity.loc[
                round(w * 100, 3),
                round(g * 100, 3)
            ] = result["intrinsic"]


    # Base case is the center of the grid.

    base_row = 4
    base_col = 4

    base_case_value = sensitivity.iloc[
        base_row,
        base_col
    ]

    st.subheader("Base Case Verification")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Base WACC",
        percent(adjusted_wacc)
    )

    c2.metric(
        "Terminal Growth",
        percent(terminal_growth)
    )

    c3.metric(
        "Base Intrinsic Value",
        money(base_case_value)
    )


    # --------------------------------------------------------
    # HEATMAP
    # --------------------------------------------------------

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=sensitivity.values,
            x=sensitivity.columns,
            y=sensitivity.index,
            colorscale="RdYlGn",
            colorbar={
                "title": "Value / Share"
            },
            hovertemplate=(
                "Terminal Growth: %{x}%<br>"
                "WACC: %{y}%<br>"
                "Intrinsic Value: $%{z:.2f}"
                "<extra></extra>"
            )
        )
    )

    # Mark base case

    fig_heatmap.add_trace(
        go.Scatter(
            x=[round(terminal_growth * 100, 3)],
            y=[round(adjusted_wacc * 100, 3)],
            mode="markers+text",
            marker={
                "symbol": "star",
                "size": 20,
                "color": "white",
                "line": {
                    "color": "black",
                    "width": 1
                }
            },
            text=["BASE CASE"],
            textposition="top center",
            name="Base Case"
        )
    )

    fig_heatmap.update_layout(
        title="Intrinsic Value: WACC vs. Terminal Growth",
        xaxis_title="Terminal Growth (%)",
        yaxis_title="WACC (%)",
        template="plotly_dark",
        height=550
    )

    st.plotly_chart(
        fig_heatmap,
        width="stretch"
    )


    # --------------------------------------------------------
    # COLOR-GRADED TABLE
    # --------------------------------------------------------

    st.subheader("Sensitivity Table — Value per Share")

    def highlight_base(data):

        styles = pd.DataFrame(
            "",
            index=data.index,
            columns=data.columns
        )

        styles.iloc[base_row, base_col] = (
            "background-color: yellow; "
            "color: black; "
            "font-weight: bold; "
            "border: 2px solid black;"
        )

        return styles

    st.dataframe(
        sensitivity.style.format(
            "${:.2f}",
            na_rep="N/A"
        ).background_gradient(
            cmap="RdYlGn",
            axis=None
        ).apply(
            highlight_base,
            axis=None
        ),
        width="stretch"
    )


    # --------------------------------------------------------
    # WHICH ASSUMPTION MOVES VALUATION MOST?
    # --------------------------------------------------------

    st.subheader("Sensitivity Interpretation")

    # Average valuation across terminal growth
    # for each WACC.

    wacc_means = sensitivity.mean(axis=1)

    # Average valuation across WACC
    # for each terminal growth.

    growth_means = sensitivity.mean(axis=0)

    wacc_impact = (
        wacc_means.max() - wacc_means.min()
    )

    terminal_impact = (
        growth_means.max() - growth_means.min()
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "WACC Valuation Impact",
        money(wacc_impact)
    )

    col2.metric(
        "Terminal Growth Impact",
        money(terminal_impact)
    )

    if wacc_impact > terminal_impact:

        st.info(
            "WACC has the greater valuation impact across the tested grid."
        )

    elif terminal_impact > wacc_impact:

        st.info(
            "Terminal growth has the greater valuation impact across the tested grid."
        )

    else:

        st.info(
            "Both assumptions have similar valuation impacts across the tested grid."
        )


    # --------------------------------------------------------
    # CHEAP VS. EXPENSIVE
    # --------------------------------------------------------

    if pd.notna(market_price) and market_price > 0:

        valid_values = sensitivity.stack().dropna()

        cheaper = (valid_values > market_price).sum()

        more_expensive = (valid_values < market_price).sum()

        total = len(valid_values)

        st.subheader("Market Price Comparison")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Scenarios Above Market",
            f"{cheaper} / {total}"
        )

        c2.metric(
            "Scenarios Below Market",
            f"{more_expensive} / {total}"
        )

        c3.metric(
            "Current Market Price",
            money(market_price)
        )

        st.caption(
            "Above market means the model estimates intrinsic value "
            "higher than the current market price."
        )


    # --------------------------------------------------------
    # SECOND GRID: FORECAST GROWTH x WACC
    # --------------------------------------------------------

    st.subheader("Forecast FCF Growth vs. WACC")

    forecast_growth_values = (
        growth + np.linspace(-0.03, 0.03, 7)
    )

    forecast_wacc_values = (
        adjusted_wacc + np.linspace(-0.02, 0.02, 7)
    )

    growth_sensitivity = pd.DataFrame(
        index=np.round(forecast_wacc_values * 100, 3),
        columns=np.round(forecast_growth_values * 100, 3),
        dtype=float
    )

    for w in forecast_wacc_values:

        for g in forecast_growth_values:

            if w <= terminal_growth:
                continue

            result = calculate_dcf(
                base_fcf,
                g,
                terminal_growth,
                years,
                w,
                cash,
                debt,
                shares
            )

            growth_sensitivity.loc[
                round(w * 100, 3),
                round(g * 100, 3)
            ] = result["intrinsic"]


    fig_growth_sensitivity = px.imshow(
        growth_sensitivity,
        labels={
            "x": "Forecast FCF Growth (%)",
            "y": "WACC (%)",
            "color": "Intrinsic Value"
        },
        color_continuous_scale="RdYlGn",
        aspect="auto",
        title="Forecast Growth vs. WACC"
    )

    st.plotly_chart(
        fig_growth_sensitivity,
        width="stretch"
    )

    st.dataframe(
        growth_sensitivity.style.format(
            "${:.2f}",
            na_rep="N/A"
        ).background_gradient(
            cmap="RdYlGn",
            axis=None
        ),
        width="stretch"
    )

else:

    st.warning(
        "Sensitivity analysis requires a valid DCF valuation. "
        "Resolve the WACC, FCF or shares outstanding issue above."
    )


# ============================================================
# PART 5 — BULL / BASE / BEAR SCENARIOS
# ============================================================

st.header("Part 5 | Bull / Base / Bear Scenarios")

scenario_results = []

if wacc_available and pd.notna(base_fcf) and base_fcf > 0 and pd.notna(shares) and shares > 0:

    for name, assumptions in scenario_settings.items():

        scenario_g = assumptions["growth"]

        scenario_tg = assumptions["terminal"]

        if adjusted_wacc <= scenario_tg:

            scenario_value = np.nan

        else:

            scenario_dcf = calculate_dcf(
                base_fcf,
                scenario_g,
                scenario_tg,
                years,
                adjusted_wacc,
                cash,
                debt,
                shares
            )

            scenario_value = scenario_dcf["intrinsic"]

        scenario_results.append({
            "Scenario": name,
            "Forecast Growth": scenario_g,
            "Terminal Growth": scenario_tg,
            "WACC": adjusted_wacc,
            "Intrinsic Value": scenario_value
        })


    scenario_df = pd.DataFrame(scenario_results)

    if pd.notna(market_price) and market_price > 0:

        scenario_df["Upside / Downside"] = (
            scenario_df["Intrinsic Value"] / market_price - 1
        )

    st.subheader("Scenario Valuation Table")

    st.dataframe(
        scenario_df.style.format({
            "Forecast Growth": "{:.2%}",
            "Terminal Growth": "{:.2%}",
            "WACC": "{:.2%}",
            "Intrinsic Value": "${:.2f}",
            "Upside / Downside": "{:.2%}"
        }, na_rep="N/A"),
        width="stretch"
    )

    fig_scenarios = px.bar(
        scenario_df,
        x="Scenario",
        y="Intrinsic Value",
        color="Scenario",
        text="Intrinsic Value",
        color_discrete_map={
            "Bull": "green",
            "Base": "royalblue",
            "Bear": "red"
        },
        title="MRK Scenario Valuation",
        template="plotly_dark"
    )

    fig_scenarios.update_traces(
        texttemplate="$%{text:.2f}",
        textposition="outside"
    )

    if pd.notna(market_price):

        fig_scenarios.add_hline(
            y=market_price,
            line_dash="dash",
            annotation_text="Current Market Price"
        )

    st.plotly_chart(
        fig_scenarios,
        width="stretch"
    )

else:

    st.warning(
        "Scenario valuation requires valid FCF, WACC and shares outstanding."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

st.header("Investment Valuation Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "DCF Intrinsic Value",
    money(intrinsic_value)
)

col2.metric(
    "Market Price",
    money(market_price)
)

col3.metric(
    "Upside / Downside",
    percent(upside)
)

st.divider()

st.subheader("Key Investment Considerations")

st.markdown("""
1. **Fundamentals:** Evaluate historical revenue, net income and free cash flow.

2. **Capital Structure:** CAPM estimates cost of equity, while WACC incorporates
   the cost of debt and market-value capital weights.

3. **DCF:** Intrinsic value depends on forecast free cash flow, WACC and
   terminal growth.

4. **Comparable Companies:** Peer multiples provide market-based valuation
   references.

5. **Sensitivity:** Evaluate how changes in WACC and growth assumptions
   affect intrinsic value.

6. **Scenarios:** Bull, base and bear cases illustrate different
   potential valuation outcomes.
""")

st.caption(
    "Data source: Yahoo Finance via yfinance. "
    "CSV fallback available. Educational valuation model."
)

st.caption(
    f"Equity Research Dashboard | {ticker} | "
    "Python • Streamlit • Yahoo Finance • DCF • CAPM"
)
