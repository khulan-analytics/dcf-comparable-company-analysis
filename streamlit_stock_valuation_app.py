
# ============================================================
# MRK EQUITY VALUATION DASHBOARD
# Fundamentals | DCF | CAPM/WACC | Comparable Companies
# Sensitivity Analysis | Bull/Base/Bear Scenarios
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="MRK Equity Valuation",
    page_icon="📊",
    layout="wide"
)

st.title("MRK | Equity Valuation Dashboard")
st.caption(
    "Fundamental Analysis • Discounted Cash Flow • Comparable Companies"
)

# ------------------------------------------------------------
# CUSTOM STYLE
# ------------------------------------------------------------

st.markdown("""
<style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128,128,128,0.08);
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# 1. HELPER FUNCTIONS
# ------------------------------------------------------------

def pick(df, *labels, default=np.nan):
    """Return the first matching column label, else a default."""

    for label in labels:
        if label in df.columns:
            return df[label]

    return default


def scalar(value, default=np.nan):
    """Safely convert a value into a float."""

    try:
        if value is None:
            return default

        value = float(value)

        if np.isfinite(value):
            return value

        return default

    except (TypeError, ValueError):
        return default


def format_money(value):
    if pd.isna(value):
        return "N/A"

    return f"${value:,.2f}"


def format_billions(value):
    if pd.isna(value):
        return "N/A"

    return f"${value / 1e9:,.2f}B"


def get_statement(ticker, statement_type, uploaded_file=None):
    """
    Download financial statements using yfinance.
    If unavailable, use an uploaded cached CSV.
    """

    try:
        stock = yf.Ticker(ticker)

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

    # CSV FALLBACK

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file, index_col=0)

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            return df, "Uploaded CSV"

        except Exception:
            pass

    return pd.DataFrame(), "Unavailable"


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_info(ticker):
    stock = yf.Ticker(ticker)

    info = stock.info

    return info


@st.cache_data(ttl=3600, show_spinner=False)
def get_price_history(ticker):
    stock = yf.Ticker(ticker)

    history = stock.history(period="1mo")

    if history.empty:
        return np.nan

    return scalar(history["Close"].dropna().iloc[-1])


def get_value(df, *labels, default=np.nan):
    """Safely retrieve the latest available statement value."""

    if df.empty:
        return default

    result = pick(df, *labels, default=default)

    if isinstance(result, pd.Series):

        result = result.dropna()

        if not result.empty:
            return scalar(result.iloc[-1], default)

    return default


def safe_ratio(numerator, denominator):
    numerator = scalar(numerator)
    denominator = scalar(denominator)

    if pd.isna(numerator) or pd.isna(denominator):
        return np.nan

    if denominator == 0:
        return np.nan

    return numerator / denominator


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
    """Calculate FCFF DCF valuation."""

    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")

    if wacc <= -1:
        raise ValueError("WACC must be greater than -100%.")

    if years < 1:
        raise ValueError("Forecast years must be at least 1.")

    if pd.isna(base_fcf) or base_fcf <= 0:
        raise ValueError("Base FCF must be positive.")

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

    terminal_percentage = (
        pv_terminal / enterprise * 100
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
        "terminal_percentage": terminal_percentage
    }


# ------------------------------------------------------------
# 2. SIDEBAR INPUTS
# ------------------------------------------------------------

st.sidebar.title("Valuation Assumptions")

ticker = st.sidebar.text_input(
    "Company Ticker",
    value="MRK"
).strip().upper()

if not ticker:
    st.error("Please enter a valid ticker.")
    st.stop()


# ------------------------------------------------------------
# CSV FALLBACK
# ------------------------------------------------------------

st.sidebar.subheader("Data Fallback")

st.sidebar.caption(
    "Upload previously downloaded CSV statements if Yahoo Finance "
    "is unavailable."
)

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

st.sidebar.subheader("Scenario Analysis")

scenario = st.sidebar.selectbox(
    "Select Scenario",
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

scenario_growth = scenario_settings[scenario]["growth"]
scenario_terminal = scenario_settings[scenario]["terminal"]


# ------------------------------------------------------------
# DCF ASSUMPTIONS
# ------------------------------------------------------------

st.sidebar.subheader("DCF Assumptions")

growth = st.sidebar.slider(
    "Forecast FCF Growth",
    min_value=-0.10,
    max_value=0.20,
    value=scenario_growth,
    step=0.005,
    format="%.2f%%"
)

terminal_growth = st.sidebar.slider(
    "Terminal Growth",
    min_value=0.000,
    max_value=0.050,
    value=scenario_terminal,
    step=0.005,
    format="%.2f%%"
)

years = st.sidebar.slider(
    "Forecast Years",
    min_value=3,
    max_value=10,
    value=5
)


# ------------------------------------------------------------
# CAPM ASSUMPTIONS
# ------------------------------------------------------------

st.sidebar.subheader("CAPM / WACC")

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate",
    min_value=0.0,
    max_value=0.15,
    value=0.04,
    step=0.005,
    format="%.3f"
)

market_return = st.sidebar.number_input(
    "Expected Market Return",
    min_value=0.0,
    max_value=0.25,
    value=0.10,
    step=0.005,
    format="%.3f"
)

manual_beta = st.sidebar.number_input(
    "Beta Override (0 = use Yahoo beta)",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.05
)

manual_cost_debt = st.sidebar.number_input(
    "Pre-Tax Cost of Debt",
    min_value=0.0,
    max_value=0.20,
    value=0.055,
    step=0.005,
    format="%.3f"
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
    min_value=-0.05,
    max_value=0.05,
    value=0.0,
    step=0.0025,
    format="%.2f%%"
)


# ------------------------------------------------------------
# COMPARABLE COMPANIES
# ------------------------------------------------------------

st.sidebar.subheader("Comparable Companies")

peer_input = st.sidebar.text_input(
    "Peer Tickers (comma separated)",
    value="ABBV, BMY, PFE, JNJ, GILD"
)

peers = [
    x.strip().upper()
    for x in peer_input.split(",")
    if x.strip()
]

peers = list(dict.fromkeys(peers))

peers = [p for p in peers if p != ticker]


# ------------------------------------------------------------
# 3. DOWNLOAD COMPANY DATA
# ------------------------------------------------------------

with st.spinner(f"Loading {ticker} financial statements..."):

    income, income_source = get_statement(
        ticker, "income", income_csv
    )

    balance, balance_source = get_statement(
        ticker, "balance", balance_csv
    )

    cashflow, cashflow_source = get_statement(
        ticker, "cashflow", cashflow_csv
    )

try:
    info = get_stock_info(ticker)

except Exception:
    info = {}


if income.empty and balance.empty and cashflow.empty:

    st.error(
        "No financial statements available. "
        "Check the ticker or upload cached CSV files in the sidebar."
    )

    st.stop()


# ------------------------------------------------------------
# 4. FUNDAMENTAL DATA EXTRACTION
# ------------------------------------------------------------

# Income statement

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

# Cash flow statement

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

# Free cash flow

fcf = operating_cash_flow - capex.abs()

# Balance sheet

total_assets = pick(
    balance,
    "Total Assets"
)

current_assets = pick(
    balance,
    "Current Assets"
)

current_liabilities = pick(
    balance,
    "Current Liabilities"
)

total_debt_series = pick(
    balance,
    "Total Debt",
    "Long Term Debt And Capital Lease Obligation"
)

cash_series = pick(
    balance,
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments"
)

equity_series = pick(
    balance,
    "Stockholders Equity",
    "Total Stockholder Equity",
    "Total Equity Gross Minority Interest"
)

# Debt and cash

debt = get_value(
    balance,
    "Total Debt",
    "Long Term Debt And Capital Lease Obligation",
    "Long Term Debt",
    default=np.nan
)

cash = get_value(
    balance,
    "Cash And Cash Equivalents",
    "Cash Cash Equivalents And Short Term Investments",
    default=np.nan
)

# Avoid silently assuming missing cash or debt equals zero.
if pd.isna(cash):
    cash = 0.0
    st.warning("Cash balance unavailable. Zero is being used in the DCF bridge.")

if pd.isna(debt):
    debt = 0.0
    st.warning("Debt balance unavailable. Zero is being used in the DCF bridge.")

shares = scalar(
    info.get("sharesOutstanding"),
    np.nan
)

beta = scalar(
    info.get("beta"),
    np.nan
)

if manual_beta > 0:
    beta = manual_beta

market_price = scalar(
    info.get("currentPrice"),
    np.nan
)

if pd.isna(market_price):
    market_price = scalar(
        info.get("regularMarketPrice"),
        np.nan
    )

if pd.isna(market_price):
    try:
        market_price = get_price_history(ticker)
    except Exception:
        market_price = np.nan


# ------------------------------------------------------------
# 5. FUNDAMENTAL ANALYSIS
# ------------------------------------------------------------

st.header("Part 1 | Fundamentals & Free Cash Flow")

st.caption(
    f"Data sources: Income — {income_source} | "
    f"Balance Sheet — {balance_source} | "
    f"Cash Flow — {cashflow_source}"
)

# Combine financial data

fundamentals = pd.DataFrame({
    "Revenue": revenue,
    "Net Income": net_income,
    "Operating Cash Flow": operating_cash_flow,
    "Capital Expenditure": capex,
    "Free Cash Flow": fcf
})

fundamentals = fundamentals.sort_index(ascending=False).head(4)

fundamentals["Revenue Growth"] = (
    fundamentals["Revenue"].pct_change(-1) * 100
)

fundamentals["Net Margin"] = (
    fundamentals["Net Income"]
    / fundamentals["Revenue"] * 100
)

fundamentals["FCF Margin"] = (
    fundamentals["Free Cash Flow"]
    / fundamentals["Revenue"] * 100
)

# Ratios

current_assets_value = get_value(
    balance,
    "Current Assets"
)

current_liabilities_value = get_value(
    balance,
    "Current Liabilities"
)

equity = get_value(
    balance,
    "Stockholders Equity",
    "Total Stockholder Equity",
    "Total Equity Gross Minority Interest"
)

current_ratio = safe_ratio(
    current_assets_value,
    current_liabilities_value
)

debt_equity = safe_ratio(
    debt,
    equity
)

# Metrics

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Market Price",
    format_money(market_price)
)

col2.metric(
    "Market Beta",
    f"{beta:.2f}" if pd.notna(beta) else "N/A"
)

col3.metric(
    "Current Ratio",
    f"{current_ratio:.2f}" if pd.notna(current_ratio) else "N/A"
)

col4.metric(
    "Debt / Equity",
    f"{debt_equity:.2f}" if pd.notna(debt_equity) else "N/A"
)

st.subheader("Historical Financial Performance")

display_fundamentals = fundamentals.copy()

for column in [
    "Revenue",
    "Net Income",
    "Operating Cash Flow",
    "Capital Expenditure",
    "Free Cash Flow"
]:
    display_fundamentals[column] = (
        display_fundamentals[column] / 1e9
    )

st.dataframe(
    display_fundamentals.round(2),
    width="stretch"
)

st.caption("Financial statement amounts are presented in billions of USD.")

# Financial chart

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
    title="Revenue, Net Income and Free Cash Flow",
    xaxis_title="Fiscal Year",
    yaxis_title="USD Billions",
    template="plotly_dark",
    height=430,
    hovermode="x unified"
)

st.plotly_chart(
    fig_fundamentals,
    width="stretch"
)


# ------------------------------------------------------------
# 6. THREE-YEAR AVERAGE FCF
# ------------------------------------------------------------

st.subheader("Free Cash Flow Analysis")

historical_fcf = fcf.dropna().sort_index()

recent_fcf = historical_fcf.tail(3)

if len(recent_fcf) < 3:

    st.warning(
        "Fewer than three years of FCF are available. "
        "The average uses the available observations."
    )

base_fcf = scalar(recent_fcf.mean())

fcf_col1, fcf_col2 = st.columns(2)

fcf_col1.metric(
    "3-Year Average FCF",
    format_billions(base_fcf)
)

fcf_col2.metric(
    "Latest FCF",
    format_billions(
        historical_fcf.iloc[-1]
        if not historical_fcf.empty else np.nan
    )
)


# ------------------------------------------------------------
# 7. CAPM AND WACC
# ------------------------------------------------------------

st.header("Part 2 | DCF Valuation")

st.subheader("Cost of Equity — CAPM")

if pd.isna(beta):

    st.warning(
        "Yahoo Finance beta is unavailable. "
        "Enter a beta override in the sidebar."
    )

    st.stop()

cost_of_equity = (
    risk_free_rate
    + beta * (market_return - risk_free_rate)
)

# Market capitalization

market_cap = scalar(
    info.get("marketCap"),
    np.nan
)

if pd.isna(market_cap) and pd.notna(shares) and pd.notna(market_price):
    market_cap = shares * market_price

# Market value weights

if pd.isna(market_cap) or market_cap <= 0:

    st.error(
        "Market capitalization is unavailable. "
        "WACC cannot be calculated reliably."
    )

    st.stop()

total_capital = market_cap + debt

equity_weight = market_cap / total_capital

debt_weight = debt / total_capital

after_tax_cost_debt = (
    manual_cost_debt * (1 - tax_rate)
)

base_wacc = (
    equity_weight * cost_of_equity
    + debt_weight * after_tax_cost_debt
)

adjusted_wacc = base_wacc + wacc_adjustment

# Display CAPM / WACC

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Cost of Equity",
    f"{cost_of_equity:.2%}"
)

c2.metric(
    "After-Tax Cost of Debt",
    f"{after_tax_cost_debt:.2%}"
)

c3.metric(
    "Baseline WACC",
    f"{base_wacc:.2%}"
)

c4.metric(
    "Adjusted WACC",
    f"{adjusted_wacc:.2%}",
    delta=f"{wacc_adjustment:.2%}"
)

st.write("**Capital Structure**")

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
    "Cost of debt is an adjustable assumption. "
    "Market capitalization and debt are used to determine capital weights."
)


# ------------------------------------------------------------
# 8. DCF VALUATION
# ------------------------------------------------------------

st.subheader("Discounted Cash Flow Model")

if adjusted_wacc <= terminal_growth:

    st.error(
        "DCF cannot be calculated: terminal growth must be "
        "strictly below WACC. Adjust your assumptions in the sidebar."
    )

    st.stop()

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

except ValueError as error:

    st.error(str(error))
    st.stop()


# DCF forecast table

forecast_table = pd.DataFrame({
    "Year": np.arange(1, years + 1),
    "Projected FCF": dcf["projected"],
    "Discount Factor": (
        1 / (1 + adjusted_wacc)
        ** np.arange(1, years + 1)
    ),
    "Present Value of FCF": dcf["pv"]
})

st.dataframe(
    forecast_table.style.format({
        "Projected FCF": "${:,.0f}",
        "Discount Factor": "{:.4f}",
        "Present Value of FCF": "${:,.0f}"
    }),
    width="stretch"
)


# DCF bridge

dcf_bridge = pd.DataFrame({
    "Valuation Component": [
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

st.subheader("Enterprise Value to Equity Value Bridge")

st.dataframe(
    dcf_bridge.style.format({
        "Value": "${:,.0f}"
    }),
    width="stretch"
)


# Intrinsic value

intrinsic_value = dcf["intrinsic"]

upside = (
    (intrinsic_value / market_price - 1) * 100
    if pd.notna(market_price) and market_price > 0
    else np.nan
)

v1, v2, v3, v4 = st.columns(4)

v1.metric(
    "DCF Intrinsic Value",
    format_money(intrinsic_value)
)

v2.metric(
    "Current Market Price",
    format_money(market_price)
)

v3.metric(
    "Upside / Downside",
    f"{upside:.2f}%" if pd.notna(upside) else "N/A"
)

v4.metric(
    "Terminal Value / EV",
    f"{dcf['terminal_percentage']:.2f}%"
)

# Valuation comparison chart

if pd.notna(market_price):

    valuation_chart = pd.DataFrame({
        "Valuation": [
            "Intrinsic Value",
            "Market Price"
        ],
        "Share Price": [
            intrinsic_value,
            market_price
        ]
    })

    fig_valuation = px.bar(
        valuation_chart,
        x="Valuation",
        y="Share Price",
        text="Share Price",
        title="DCF Intrinsic Value vs. Market Price",
        template="plotly_dark"
    )

    fig_valuation.update_traces(
        texttemplate="$%{text:.2f}",
        textposition="outside"
    )

    st.plotly_chart(
        fig_valuation,
        width="stretch"
    )


# ------------------------------------------------------------
# 9. COMPARABLE COMPANY ANALYSIS
# ------------------------------------------------------------

st.header("Part 3 | Comparable Company Analysis")

st.write(
    "The following analysis compares MRK against pharmaceutical peers "
    "using valuation multiples from Yahoo Finance."
)

peer_rows = []

for peer in peers:

    try:

        peer_info = get_stock_info(peer)

        peer_rows.append({
            "Ticker": peer,
            "P/E": scalar(peer_info.get("trailingPE")),
            "EV/EBITDA": scalar(peer_info.get("enterpriseToEbitda")),
            "P/S": scalar(
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

if not peer_df.empty:

    st.subheader("Peer Valuation Multiples")

    st.dataframe(
        peer_df.style.format({
            "P/E": "{:.2f}x",
            "EV/EBITDA": "{:.2f}x",
            "P/S": "{:.2f}x"
        }, na_rep="N/A"),
        width="stretch"
    )

    # Median multiples

    median_pe = peer_df["P/E"].dropna().median()

    median_ev_ebitda = peer_df["EV/EBITDA"].dropna().median()

    median_ps = peer_df["P/S"].dropna().median()

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

    st.subheader("Peer Median Multiples")

    st.dataframe(
        peer_display.style.format({
            "P/E": "{:.2f}x",
            "EV/EBITDA": "{:.2f}x",
            "P/S": "{:.2f}x"
        }, na_rep="N/A").highlight_max(
            axis=0,
            subset=["P/E", "EV/EBITDA", "P/S"],
            color="#355E3B"
        ),
        width="stretch"
    )


# ------------------------------------------------------------
# 10. IMPLIED SHARE PRICE FROM COMPARABLES
# ------------------------------------------------------------

st.subheader("Comparable Company Implied Valuation")

# Company metrics

company_eps = scalar(
    info.get("trailingEps"),
    np.nan
)

company_ebitda = scalar(
    info.get("ebitda"),
    np.nan
)

company_revenue = get_value(
    income,
    "Total Revenue",
    "Operating Revenue"
)

# P/E implied price

implied_pe = (
    median_pe * company_eps
    if pd.notna(median_pe) and pd.notna(company_eps)
    and company_eps > 0 and median_pe > 0
    else np.nan
)

# EV/EBITDA implied price

implied_ev_ebitda = np.nan

if (
    pd.notna(median_ev_ebitda)
    and pd.notna(company_ebitda)
    and pd.notna(shares)
    and median_ev_ebitda > 0
    and company_ebitda > 0
    and shares > 0
):

    implied_ev = median_ev_ebitda * company_ebitda

    implied_equity = implied_ev + cash - debt

    implied_ev_ebitda = implied_equity / shares


# P/S implied price

implied_ps = (
    median_ps * company_revenue / shares
    if pd.notna(median_ps)
    and pd.notna(company_revenue)
    and pd.notna(shares)
    and median_ps > 0
    and shares > 0
    else np.nan
)


comps_values = pd.DataFrame({
    "Method": [
        "P/E",
        "EV/EBITDA",
        "P/S"
    ],

    "Implied Share Price": [
        implied_pe,
        implied_ev_ebitda,
        implied_ps
    ]
})

st.dataframe(
    comps_values.style.format({
        "Implied Share Price": "${:.2f}"
    }, na_rep="N/A"),
    width="stretch"
)


# ------------------------------------------------------------
# 11. VALUATION COMPARISON
# ------------------------------------------------------------

st.subheader("Three-Way Valuation Comparison")

valuation_comparison = pd.DataFrame({
    "Method": [
        "DCF Intrinsic Value",
        "Comparable Valuation — P/E",
        "Comparable Valuation — EV/EBITDA",
        "Comparable Valuation — P/S",
        "Current Market Price"
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
    x="Method",
    y="Share Price",
    text="Share Price",
    title="MRK Valuation Methods Compared",
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

# Explain differences

st.subheader("Valuation Interpretation")

st.markdown("""
**Why valuation methods differ:**

- **DCF:** Estimates intrinsic value based on projected future free cash flow,
  discount rates and terminal growth.
- **P/E:** Reflects the market's valuation of earnings relative to peer companies.
- **EV/EBITDA:** Compares operating enterprise value while accounting for
  differences in capital structure.
- **P/S:** Compares market capitalization relative to sales.

Differences between the methods can reflect growth expectations, profitability,
capital structure, risk, accounting differences and market sentiment.
""")


# ------------------------------------------------------------
# 12. SENSITIVITY ANALYSIS
# ------------------------------------------------------------

st.header("Part 4 | Sensitivity Analysis")

st.write(
    "The table measures how intrinsic value changes when WACC "
    "and terminal growth assumptions change."
)

# Grid centered on base case

wacc_values = np.linspace(
    max(-0.05, adjusted_wacc - 0.02),
    adjusted_wacc + 0.02,
    9
)

growth_values = np.linspace(
    max(-0.02, terminal_growth - 0.02),
    terminal_growth + 0.02,
    9
)

sensitivity = pd.DataFrame(
    index=np.round(wacc_values * 100, 2),
    columns=np.round(growth_values * 100, 2),
    dtype=float
)

for w in wacc_values:

    for g in growth_values:

        if w <= g:
            sensitivity.loc[
                round(w * 100, 2),
                round(g * 100, 2)
            ] = np.nan

        else:

            try:

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
                    round(w * 100, 2),
                    round(g * 100, 2)
                ] = result["intrinsic"]

            except ValueError:
                pass


st.subheader("WACC vs. Terminal Growth")

# Heatmap

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=sensitivity.values,
        x=sensitivity.columns,
        y=sensitivity.index,
        colorscale="RdYlGn",
        colorbar=dict(title="Value / Share"),
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
        x=[terminal_growth * 100],
        y=[adjusted_wacc * 100],
        mode="markers+text",
        marker=dict(
            symbol="star",
            size=18,
            color="white",
            line=dict(color="black", width=1)
        ),
        text=["BASE CASE"],
        textposition="top center",
        name="Base Case"
    )
)

fig_heatmap.update_layout(
    title="DCF Sensitivity: WACC vs. Terminal Growth",
    xaxis_title="Terminal Growth (%)",
    yaxis_title="WACC (%)",
    template="plotly_dark",
    height=550
)

st.plotly_chart(
    fig_heatmap,
    width="stretch"
)

st.caption(
    "The white star identifies the base-case assumptions. "
    "Green indicates higher intrinsic value; red indicates lower intrinsic value."
)

# Numerical sensitivity table

st.subheader("Sensitivity Table — Intrinsic Value per Share")

st.dataframe(
    sensitivity.style.format(
        "${:.2f}",
        na_rep="N/A"
    ).background_gradient(
        cmap="RdYlGn",
        axis=None
    ),
    width="stretch"
)


# ------------------------------------------------------------
# 13. WHICH ASSUMPTION MOVES VALUATION MOST?
# ------------------------------------------------------------

st.subheader("Sensitivity Interpretation")

# Compare WACC and terminal growth impact

valid_sensitivity = sensitivity.stack().dropna()

if not valid_sensitivity.empty:

    wacc_impact = sensitivity.max(axis=1).max() - sensitivity.min(axis=1).min()

    growth_impact = sensitivity.max(axis=0).max() - sensitivity.min(axis=0).min()

    # Compare full-range variation along each dimension
    wacc_range = (
        sensitivity.iloc[0].dropna().mean()
        - sensitivity.iloc[-1].dropna().mean()
    )

    growth_range = (
        sensitivity.iloc[:, -1].dropna().mean()
        - sensitivity.iloc[:, 0].dropna().mean()
    )

    sensitivity_col1, sensitivity_col2 = st.columns(2)

    sensitivity_col1.metric(
        "WACC Range Impact",
        f"${abs(wacc_range):.2f}"
    )

    sensitivity_col2.metric(
        "Terminal Growth Range Impact",
        f"${abs(growth_range):.2f}"
    )

    if abs(wacc_range) > abs(growth_range):

        st.info(
            "WACC has the larger valuation impact across the tested grid."
        )

    else:

        st.info(
            "Terminal growth has the larger valuation impact across the tested grid."
        )

    if pd.notna(market_price):

        st.write("**Relative to the current market price:**")

        cheap_cells = (
            sensitivity < market_price
        ).sum().sum()

        expensive_cells = (
            sensitivity > market_price
        ).sum().sum()

        total_cells = sensitivity.notna().sum().sum()

        st.write(
            f"- Intrinsic value below market price: "
            f"{cheap_cells} of {total_cells} valid scenarios."
        )

        st.write(
            f"- Intrinsic value above market price: "
            f"{expensive_cells} of {total_cells} valid scenarios."
        )


# ------------------------------------------------------------
# 14. OPTIONAL SECOND SENSITIVITY GRID
# ------------------------------------------------------------

st.subheader("Additional Sensitivity: Forecast Growth vs. WACC")

forecast_growth_values = np.linspace(
    growth - 0.03,
    growth + 0.03,
    7
)

forecast_wacc_values = np.linspace(
    max(-0.05, adjusted_wacc - 0.02),
    adjusted_wacc + 0.02,
    7
)

growth_sensitivity = pd.DataFrame(
    index=np.round(forecast_wacc_values * 100, 2),
    columns=np.round(forecast_growth_values * 100, 2),
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
            round(w * 100, 2),
            round(g * 100, 2)
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


# ------------------------------------------------------------
# 15. BULL / BASE / BEAR SCENARIO SUMMARY
# ------------------------------------------------------------

st.header("Part 5 | Bull, Base and Bear Scenarios")

scenario_results = []

for scenario_name, assumptions in scenario_settings.items():

    scenario_g = assumptions["growth"]
    scenario_tg = assumptions["terminal"]

    if adjusted_wacc <= scenario_tg:
        scenario_value = np.nan

    else:

        try:

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

        except ValueError:

            scenario_value = np.nan

    scenario_results.append({
        "Scenario": scenario_name,
        "Forecast Growth": scenario_g,
        "Terminal Growth": scenario_tg,
        "WACC": adjusted_wacc,
        "Intrinsic Value": scenario_value
    })


scenario_df = pd.DataFrame(scenario_results)

if pd.notna(market_price):

    scenario_df["Upside / Downside"] = (
        scenario_df["Intrinsic Value"] / market_price - 1
    ) * 100

st.dataframe(
    scenario_df.style.format({
        "Forecast Growth": "{:.2%}",
        "Terminal Growth": "{:.2%}",
        "WACC": "{:.2%}",
        "Intrinsic Value": "${:.2f}",
        "Upside / Downside": "{:.2f}%"
    }, na_rep="N/A"),
    width="stretch"
)

# Scenario chart

fig_scenarios = px.bar(
    scenario_df,
    x="Scenario",
    y="Intrinsic Value",
    color="Scenario",
    text="Intrinsic Value",
    title="MRK Bull / Base / Bear Valuation",
    template="plotly_dark",
    color_discrete_map={
        "Bull": "green",
        "Base": "royalblue",
        "Bear": "red"
    }
)

fig_scenarios.update_traces(
    texttemplate="$%{text:.2f}",
    textposition="outside"
)

if pd.notna(market_price):

    fig_scenarios.add_hline(
        y=market_price,
        line_dash="dash",
        annotation_text="Market Price"
    )

st.plotly_chart(
    fig_scenarios,
    width="stretch"
)


# ------------------------------------------------------------
# 16. FINAL SUMMARY
# ------------------------------------------------------------

st.header("Investment Valuation Summary")

summary1, summary2, summary3 = st.columns(3)

summary1.metric(
    "DCF Intrinsic Value",
    format_money(intrinsic_value)
)

summary2.metric(
    "Market Price",
    format_money(market_price)
)

summary3.metric(
    "Implied Upside / Downside",
    f"{upside:.2f}%" if pd.notna(upside) else "N/A"
)

st.markdown("---")

st.markdown("""
### Key Investment Considerations

1. **Fundamentals:** Evaluate historical revenue, net income and free cash flow
   before relying on projected growth.

2. **Capital Structure:** WACC reflects the company's cost of equity,
   after-tax cost of debt and market-value financing weights.

3. **DCF Valuation:** Terminal value represents the discounted value of
   cash flows beyond the explicit forecast period.

4. **Comparable Valuation:** Peer multiples provide a market-based reference
   point, but differences in growth and profitability can explain valuation gaps.

5. **Sensitivity:** DCF results are highly dependent on discount rate and
   terminal growth assumptions.

6. **Scenarios:** Bull, base and bear cases illustrate how different
   assumptions affect estimated intrinsic value.
""")

st.caption(
    "Educational valuation model. Financial data and valuation assumptions "
    "should be independently verified before investment decisions."
)

st.markdown("---")

st.caption(
    f"Equity Research Dashboard | {ticker} | "
    "Python • Streamlit • Yahoo Finance • DCF • CAPM"
)
