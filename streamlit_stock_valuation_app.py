import io
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="Stock Valuation Lab", page_icon="📈", layout="wide")
st.title("📈 Stock Valuation Lab")
st.caption("Fundamentals • Free Cash Flow • CAPM/WACC • DCF • Comparable Companies • Sensitivity")

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


# -----------------------------
# Safe data helpers
# -----------------------------
def pick(df, *labels, default=np.nan):
    """Return the first matching column label, else a default."""
    if not isinstance(df, pd.DataFrame):
        return default
    for label in labels:
        if label in df.columns:
            return df[label]
    return default


def as_float(value, default=np.nan):
    """Convert a scalar-like value to float without raising."""
    try:
        if value is None:
            return default
        if isinstance(value, (pd.Series, pd.DataFrame, list, tuple, np.ndarray)):
            if len(value) == 0:
                return default
            value = np.asarray(value).reshape(-1)[0]
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError, OverflowError):
        return default


def safe_divide(numerator, denominator):
    numerator = as_float(numerator)
    denominator = as_float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def fmt_money(value, decimals=2):
    value = as_float(value)
    return "N/A" if not np.isfinite(value) else f"${value:,.{decimals}f}"


def fmt_num(value, decimals=2):
    value = as_float(value)
    return "N/A" if not np.isfinite(value) else f"{value:,.{decimals}f}"


def fmt_pct(value, decimals=1):
    value = as_float(value)
    return "N/A" if not np.isfinite(value) else f"{value:.{decimals}%}"


def normalize_statement(df):
    """Transpose statements to dates-as-rows, then sort chronologically."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.T.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()]
    if out.empty:
        return out
    return out.sort_index()


def get_statement_value(statement, labels, date=None):
    """Safely get a line item for one period; missing labels return NaN."""
    if statement.empty:
        return np.nan
    row = pick(statement, *labels, default=None)
    if row is None or not isinstance(row, pd.Series):
        return np.nan
    row = pd.to_numeric(row, errors="coerce").dropna()
    if row.empty:
        return np.nan
    if date is not None and date in row.index:
        return as_float(row.loc[date])
    return as_float(row.iloc[-1])


def download_fundamentals(ticker):
    """Download statements and quote data from yfinance."""
    stock = yf.Ticker(ticker)
    income = normalize_statement(stock.income_stmt)
    balance = normalize_statement(stock.balance_sheet)
    cashflow = normalize_statement(stock.cashflow)
    info = {}
    try:
        info = stock.info or {}
    except Exception:
        info = {}
    try:
        hist = stock.history(period="1y", auto_adjust=False)
    except Exception:
        hist = pd.DataFrame()
    try:
        fast_info = stock.fast_info
        fast_price = as_float(fast_info.get("lastPrice", np.nan)) if fast_info else np.nan
    except Exception:
        fast_price = np.nan
    price = as_float(info.get("currentPrice", info.get("regularMarketPrice", np.nan)))
    if not np.isfinite(price):
        price = fast_price
    if not np.isfinite(price) and not hist.empty and "Close" in hist:
        price = as_float(hist["Close"].dropna().iloc[-1]) if not hist["Close"].dropna().empty else np.nan
    return {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "info": info,
        "history": hist,
        "price": price,
        "stock": stock,
    }


def save_statement_cache(ticker, data):
    """Save each normalized statement to CSV so a future run can work offline."""
    safe_ticker = "".join(c for c in ticker.upper() if c.isalnum() or c in "._-")
    for key in ("income", "balance", "cashflow"):
        try:
            data[key].to_csv(CACHE_DIR / f"{safe_ticker}_{key}.csv", index_label="Date")
        except Exception:
            pass
    try:
        pd.DataFrame([{
            "ticker": ticker.upper(),
            "price": data["price"],
            "shares": data["info"].get("sharesOutstanding", np.nan),
            "marketCap": data["info"].get("marketCap", np.nan),
            "beta": data["info"].get("beta", np.nan),
            "trailingPE": data["info"].get("trailingPE", np.nan),
            "enterpriseToEbitda": data["info"].get("enterpriseToEbitda", np.nan),
            "priceToSalesTrailing12Months": data["info"].get("priceToSalesTrailing12Months", np.nan),
            "totalDebt": data["info"].get("totalDebt", np.nan),
            "totalCash": data["info"].get("totalCash", np.nan),
            "bookValue": data["info"].get("bookValue", np.nan),
            "trailingEps": data["info"].get("trailingEps", np.nan),
            "ebitda": data["info"].get("ebitda", np.nan),
            "totalRevenue": data["info"].get("totalRevenue", np.nan),
            "fiftyTwoWeekLow": data["info"].get("fiftyTwoWeekLow", np.nan),
            "fiftyTwoWeekHigh": data["info"].get("fiftyTwoWeekHigh", np.nan),
        }]).to_csv(CACHE_DIR / f"{safe_ticker}_info.csv", index=False)
    except Exception:
        pass


def load_statement_cache(ticker):
    """Read normalized statement CSVs. Returns None if no usable cache exists."""
    safe_ticker = "".join(c for c in ticker.upper() if c.isalnum() or c in "._-")
    result = {}
    found = False
    for key in ("income", "balance", "cashflow"):
        path = CACHE_DIR / f"{safe_ticker}_{key}.csv"
        if path.exists():
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df.loc[~df.index.isna()].sort_index()
                result[key] = df
                found = True
            except Exception:
                result[key] = pd.DataFrame()
        else:
            result[key] = pd.DataFrame()
    info_path = CACHE_DIR / f"{safe_ticker}_info.csv"
    info = {}
    if info_path.exists():
        try:
            cached = pd.read_csv(info_path)
            if not cached.empty:
                info = cached.iloc[0].to_dict()
                found = True
        except Exception:
            pass
    if not found:
        return None
    price = as_float(info.get("price", np.nan))
    return {
        **result,
        "info": info,
        "history": pd.DataFrame(),
        "price": price,
        "stock": None,
    }


@st.cache_data(ttl=900, show_spinner=False)
def cached_yfinance_download(ticker):
    """Cache raw yfinance-derived data for 15 minutes within Streamlit."""
    data = download_fundamentals(ticker)
    # DataFrames are serializable by Streamlit cache.
    return {k: v for k, v in data.items() if k != "stock"}


def load_data(ticker, prefer_csv=False):
    if prefer_csv:
        cached = load_statement_cache(ticker)
        if cached is not None:
            return cached, "CSV cache"
        return None, "No CSV cache found for this ticker."
    try:
        data = cached_yfinance_download(ticker)
        if any(not data[k].empty for k in ("income", "balance", "cashflow")):
            save_statement_cache(ticker, data)
            return data, "Yahoo Finance"
        raise RuntimeError("Yahoo Finance returned no usable financial statements.")
    except Exception as exc:
        cached = load_statement_cache(ticker)
        if cached is not None:
            return cached, f"CSV fallback (API error: {exc})"
        return None, f"Could not load Yahoo Finance data and no CSV cache was found. Details: {exc}"


def series_for(statement, labels):
    if statement.empty:
        return pd.Series(dtype=float)
    s = pick(statement, *labels, default=None)
    if not isinstance(s, pd.Series):
        return pd.Series(dtype=float)
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    return s[~s.index.isna()].sort_index().dropna()


def latest_from_statement(statement, labels):
    s = series_for(statement, labels)
    return as_float(s.iloc[-1]) if not s.empty else np.nan


def get_fundamentals(data):
    inc, bal, cf = data["income"], data["balance"], data["cashflow"]
    revenue = series_for(inc, ["Total Revenue", "Operating Revenue", "Revenue"])
    net_income = series_for(inc, ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"])
    operating_cf = series_for(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"])
    capex = series_for(cf, ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment"])
    # Align years and calculate FCF = operating cash flow - absolute capex.
    all_dates = sorted(set(revenue.index) | set(net_income.index) | set(operating_cf.index) | set(capex.index))
    annual = pd.DataFrame(index=pd.DatetimeIndex(all_dates, name="Date"))
    annual["Revenue"] = revenue
    annual["Net Income"] = net_income
    annual["Operating Cash Flow"] = operating_cf
    annual["CapEx"] = capex
    annual["FCF"] = annual["Operating Cash Flow"] - annual["CapEx"].abs()
    annual["Net Margin"] = annual["Net Income"] / annual["Revenue"].replace(0, np.nan)
    annual["FCF Margin"] = annual["FCF"] / annual["Revenue"].replace(0, np.nan)
    annual = annual.sort_index().tail(4)

    cash = latest_from_statement(bal, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents And Short Term Investments"])
    debt = latest_from_statement(bal, ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"])
    current_assets = latest_from_statement(bal, ["Current Assets", "Total Current Assets"])
    current_liabilities = latest_from_statement(bal, ["Current Liabilities", "Total Current Liabilities"])
    equity_book = latest_from_statement(bal, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
    current_ratio = safe_divide(current_assets, current_liabilities)
    debt_equity = safe_divide(debt, equity_book)
    return annual, {
        "cash": cash,
        "debt": debt,
        "current_ratio": current_ratio,
        "debt_equity": debt_equity,
        "latest_revenue": as_float(annual["Revenue"].dropna().iloc[-1]) if not annual["Revenue"].dropna().empty else np.nan,
        "latest_net_income": as_float(annual["Net Income"].dropna().iloc[-1]) if not annual["Net Income"].dropna().empty else np.nan,
        "latest_fcf": as_float(annual["FCF"].dropna().iloc[-1]) if not annual["FCF"].dropna().empty else np.nan,
    }


def capm_wacc(rf, beta, erp, cost_debt, tax_rate, market_cap, debt):
    """Market-value weighted cost of equity and after-tax debt."""
    re = rf + beta * erp
    equity_value = as_float(market_cap)
    debt_value = as_float(debt)
    if not np.isfinite(equity_value) or equity_value <= 0:
        # Fallback to equity value from market price * shares, if market cap unavailable.
        equity_value = np.nan
    if not np.isfinite(debt_value) or debt_value < 0:
        debt_value = 0.0
    if not np.isfinite(equity_value):
        return re, np.nan
    total_capital = equity_value + debt_value
    if total_capital <= 0:
        return re, np.nan
    wacc = (equity_value / total_capital) * re + (debt_value / total_capital) * cost_debt * (1 - tax_rate)
    return re, wacc


def dcf_value(base_fcf, growth, terminal_g, wacc, years, cash, debt, shares):
    if not all(np.isfinite(as_float(x)) for x in [base_fcf, growth, terminal_g, wacc, cash, debt, shares]):
        return None
    if years < 1 or shares <= 0 or wacc <= terminal_g:
        return None
    forecast_years = np.arange(1, int(years) + 1)
    projected = base_fcf * (1 + growth) ** forecast_years
    pv = projected / (1 + wacc) ** forecast_years
    terminal = projected[-1] * (1 + terminal_g) / (wacc - terminal_g)
    pv_terminal = terminal / (1 + wacc) ** int(years)
    enterprise = float(np.nansum(pv) + pv_terminal)
    equity = enterprise + cash - debt
    return {
        "projected": projected,
        "pv": pv,
        "terminal": terminal,
        "pv_terminal": pv_terminal,
        "enterprise": enterprise,
        "equity": equity,
        "intrinsic": equity / shares,
        "terminal_pct": safe_divide(pv_terminal, enterprise),
    }


def peer_metrics(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return {
        "Ticker": ticker.upper(),
        "P/E": as_float(info.get("trailingPE")),
        "EV/EBITDA": as_float(info.get("enterpriseToEbitda")),
        "P/S": as_float(info.get("priceToSalesTrailing12Months")),
        "Price": as_float(info.get("currentPrice", info.get("regularMarketPrice"))),
        "EPS": as_float(info.get("trailingEps")),
        "EBITDA": as_float(info.get("ebitda")),
        "Revenue": as_float(info.get("totalRevenue")),
        "Market Cap": as_float(info.get("marketCap")),
        "Debt": as_float(info.get("totalDebt")),
        "Cash": as_float(info.get("totalCash")),
        "Shares": as_float(info.get("sharesOutstanding")),
        "52W Low": as_float(info.get("fiftyTwoWeekLow")),
        "52W High": as_float(info.get("fiftyTwoWeekHigh")),
    }


def peer_implied_values(peers, company_metrics):
    results = []
    for multiple, metric in [("P/E", "EPS"), ("EV/EBITDA", "EBITDA"), ("P/S", "Revenue")]:
        valid = peers[multiple].replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[valid >= 0]
        median_multiple = valid.median() if not valid.empty else np.nan
        company_metric = as_float(company_metrics.get(metric))
        if not np.isfinite(median_multiple) or not np.isfinite(company_metric):
            implied = np.nan
        elif multiple == "P/E":
            implied = median_multiple * company_metric
        elif multiple == "EV/EBITDA":
            ev = median_multiple * company_metric
            implied = safe_divide(ev + company_metrics["cash"] - company_metrics["debt"], company_metrics["shares"])
        else:
            ev = np.nan
            # P/S is equity value / sales, so implied market cap = median P/S * sales.
            implied = safe_divide(median_multiple * company_metric, company_metrics["shares"])
        results.append({"Multiple": multiple, "Company Metric": metric, "Peer Median": median_multiple, "Company Metric Value": company_metric, "Implied Value / Share": implied, "Peers Used": int(valid.count())})
    return pd.DataFrame(results)


# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Company & data")
ticker = st.sidebar.text_input("Ticker", value="MRK").strip().upper()
data_mode = st.sidebar.radio(
    "Data source",
    ["Try Yahoo Finance; fall back to CSV", "Use cached CSV only"],
    help="Successful Yahoo Finance downloads are saved in data_cache/. You can also supply your own statement CSVs there.",
)
st.sidebar.caption("CSV filenames: TICKER_income.csv, TICKER_balance.csv, TICKER_cashflow.csv, and optionally TICKER_info.csv.")

st.sidebar.header("DCF assumptions")
forecast_years = st.sidebar.slider("Forecast years", min_value=3, max_value=10, value=5)
growth = st.sidebar.slider("Forecast FCF growth", min_value=-0.20, max_value=0.40, value=0.06, step=0.01, format="%.0f%%")
terminal_g = st.sidebar.slider("Terminal growth", min_value=-0.01, max_value=0.06, value=0.025, step=0.0025, format="%.2f%%")

st.sidebar.header("CAPM & WACC inputs")
rf = st.sidebar.slider("Risk-free rate", min_value=0.00, max_value=0.10, value=0.04, step=0.0025, format="%.2f%%")
beta_input = st.sidebar.number_input("Beta", min_value=0.0, max_value=4.0, value=1.0, step=0.05)
erp = st.sidebar.slider("Equity risk premium", min_value=0.00, max_value=0.12, value=0.05, step=0.0025, format="%.2f%%")
cost_debt = st.sidebar.slider("Pre-tax cost of debt", min_value=0.00, max_value=0.20, value=0.05, step=0.0025, format="%.2f%%")
tax_rate = st.sidebar.slider("Tax rate", min_value=0.00, max_value=0.50, value=0.21, step=0.01, format="%.0f%%")
wacc_adjustment = st.sidebar.slider("WACC adjustment around CAPM baseline", min_value=-0.05, max_value=0.08, value=0.00, step=0.0025, format="%+.2f%%")

st.sidebar.header("Scenario")
scenario = st.sidebar.selectbox("Apply scenario preset", ["Custom", "Bull", "Base", "Bear"])
scenario_values = {
    "Bull": (0.12, 0.03, -0.01),
    "Base": (0.06, 0.025, 0.00),
    "Bear": (-0.02, 0.015, 0.02),
}
if scenario != "Custom":
    growth, terminal_g, scenario_wacc_adjustment = scenario_values[scenario]
    wacc_adjustment = scenario_wacc_adjustment

if not ticker:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

prefer_csv = data_mode == "Use cached CSV only"
with st.spinner(f"Loading financial statements for {ticker}..."):
    data, source_message = load_data(ticker, prefer_csv=prefer_csv)

if data is None:
    st.error(source_message)
    st.markdown(
        "To use CSV fallback, create a `data_cache` folder beside this script and add statement files named "
        "`TICKER_income.csv`, `TICKER_balance.csv`, and `TICKER_cashflow.csv`. "
        "Each file should have statement line items as columns and dates as the index."
    )
    st.stop()

st.caption(f"Data source: **{source_message}**. Yahoo Finance data may be incomplete or delayed; verify before making investment decisions.")

annual, fundamentals = get_fundamentals(data)
info = data.get("info", {}) or {}
price = as_float(data.get("price"))
if not np.isfinite(price):
    price = as_float(info.get("currentPrice", info.get("regularMarketPrice")))
shares = as_float(info.get("sharesOutstanding"))
if not np.isfinite(shares) or shares <= 0:
    market_cap = as_float(info.get("marketCap"))
    shares = safe_divide(market_cap, price)
else:
    market_cap = as_float(info.get("marketCap"))
if not np.isfinite(market_cap) and np.isfinite(price) and np.isfinite(shares):
    market_cap = price * shares

cash = as_float(fundamentals["cash"])
if not np.isfinite(cash):
    cash = as_float(info.get("totalCash"), default=0.0)
debt = as_float(fundamentals["debt"])
if not np.isfinite(debt):
    debt = as_float(info.get("totalDebt"), default=0.0)

beta_live = as_float(info.get("beta"))
if not np.isfinite(beta_live):
    beta_live = beta_input
# User can adjust beta; the live beta seeds the control above in a future rerun, but this run uses the explicit sidebar value.
beta = beta_input
re, capm_wacc = capm_wacc(rf, beta, erp, cost_debt, tax_rate, market_cap, debt)
if not np.isfinite(capm_wacc):
    capm_wacc = re
wacc = capm_wacc + wacc_adjustment

# Use a 3-year average FCF base as specified.
fcf_series = series_for(data["cashflow"], ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"])
capex_series = series_for(data["cashflow"], ["Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment"])
fcf_series = (fcf_series - capex_series.abs()).dropna()
base_fcf = as_float(fcf_series.tail(3).mean()) if not fcf_series.empty else np.nan

# Company metrics for peer-implied valuation.
company_metrics = {
    "EPS": as_float(info.get("trailingEps")),
    "EBITDA": as_float(info.get("ebitda")),
    "Revenue": as_float(info.get("totalRevenue")),
    "cash": cash,
    "debt": debt,
    "shares": shares,
}
if not np.isfinite(company_metrics["Revenue"]):
    company_metrics["Revenue"] = fundamentals["latest_revenue"]
if not np.isfinite(company_metrics["EBITDA"]):
    # A rough EBITDA fallback: operating income + D&A if both are present.
    operating_income = latest_from_statement(data["income"], ["Operating Income"])
    da = latest_from_statement(data["cashflow"], ["Depreciation And Amortization", "Depreciation Amortization Depletion"])
    company_metrics["EBITDA"] = operating_income + da if np.isfinite(operating_income) and np.isfinite(da) else np.nan
if not np.isfinite(company_metrics["EPS"]) and np.isfinite(fundamentals["latest_net_income"]) and np.isfinite(shares) and shares > 0:
    company_metrics["EPS"] = fundamentals["latest_net_income"] / shares

# -----------------------------
# Part 1: fundamentals
# -----------------------------
st.header(f"{ticker} — Fundamentals & Free Cash Flow")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Market price", fmt_money(price))
k2.metric("3-year average FCF", fmt_money(base_fcf, 0))
k3.metric("Current ratio", fmt_num(fundamentals["current_ratio"]))
k4.metric("Debt / equity", fmt_num(fundamentals["debt_equity"]))

if annual.empty:
    st.warning("No annual statement data was available. DCF and fundamentals will be limited until statement data is loaded.")
else:
    display_annual = annual.copy()
    for col in ["Revenue", "Net Income", "Operating Cash Flow", "CapEx", "FCF"]:
        display_annual[col] = display_annual[col].map(lambda x: fmt_money(x, 0))
    for col in ["Net Margin", "FCF Margin"]:
        display_annual[col] = display_annual[col].map(fmt_pct)
    display_annual.index = display_annual.index.strftime("%Y")
    st.dataframe(display_annual, use_container_width=True)

with st.expander("View downloaded statements (date-sorted)"):
    st.subheader("Income statement")
    st.dataframe(data["income"].T if not data["income"].empty else pd.DataFrame(), use_container_width=True)
    st.subheader("Balance sheet")
    st.dataframe(data["balance"].T if not data["balance"].empty else pd.DataFrame(), use_container_width=True)
    st.subheader("Cash flow")
    st.dataframe(data["cashflow"].T if not data["cashflow"].empty else pd.DataFrame(), use_container_width=True)

# -----------------------------
# Part 2: DCF valuation
# -----------------------------
st.header("DCF Valuation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAPM cost of equity", fmt_pct(re))
c2.metric("Baseline CAPM/WACC", fmt_pct(capm_wacc))
c3.metric("Selected WACC", fmt_pct(wacc))
c4.metric("Base FCF (3-year avg.)", fmt_money(base_fcf, 0))

dcf = None
if not np.isfinite(base_fcf):
    st.error("FCF could not be calculated from the available cash-flow statement.")
elif terminal_g >= wacc:
    st.error(f"Terminal growth ({terminal_g:.2%}) must be below WACC ({wacc:.2%}). Adjust the sidebar assumptions.")
elif not np.isfinite(shares) or shares <= 0:
    st.error("Shares outstanding are unavailable, so per-share valuation cannot be calculated.")
else:
    dcf = dcf_value(base_fcf, growth, terminal_g, wacc, forecast_years, cash, debt, shares)
    if dcf is None:
        st.error("DCF could not be calculated. Check the assumptions and company data.")
    else:
        upside = safe_divide(dcf["intrinsic"], price) - 1 if np.isfinite(price) and price != 0 else np.nan
        a, b, c, d = st.columns(4)
        a.metric("DCF intrinsic value / share", fmt_money(dcf["intrinsic"]), delta=fmt_pct(upside) if np.isfinite(upside) else None)
        b.metric("Enterprise value", fmt_money(dcf["enterprise"], 0))
        c.metric("Equity value", fmt_money(dcf["equity"], 0))
        d.metric("PV terminal value / EV", fmt_pct(dcf["terminal_pct"]))
        st.caption("Enterprise value = PV of forecast FCF + PV of terminal value. Equity value = enterprise value + cash − debt.")

        forecast_df = pd.DataFrame({
            "Year": np.arange(1, forecast_years + 1),
            "Projected FCF": dcf["projected"],
            "PV of FCF": dcf["pv"],
        })
        st.dataframe(forecast_df.style.format({"Projected FCF": "${:,.0f}", "PV of FCF": "${:,.0f}"}), use_container_width=True)

# -----------------------------
# Part 3: comparable company analysis
# -----------------------------
st.header("Comparable Company Analysis")
default_peers = "PFE, ABBV, BMY, AMGN, GILD"
peer_text = st.text_input("Peer tickers (comma-separated)", value=default_peers)
peer_tickers = list(dict.fromkeys([p.strip().upper() for p in peer_text.split(",") if p.strip() and p.strip().upper() != ticker]))
peer_rows = []
if peer_tickers:
    with st.spinner("Loading peer valuation multiples..."):
        for peer in peer_tickers:
            peer_rows.append(peer_metrics(peer))
peers = pd.DataFrame(peer_rows)
comp_values = pd.DataFrame()
if not peers.empty:
    st.subheader("Peer valuation multiples")
    st.dataframe(
        peers[["Ticker", "P/E", "EV/EBITDA", "P/S"]].style.format({
            "P/E": "{:.2f}x", "EV/EBITDA": "{:.2f}x", "P/S": "{:.2f}x"
        }, na_rep="N/A"),
        use_container_width=True,
    )
    comp_values = peer_implied_values(peers, company_metrics)
    st.subheader("Median peer multiples and implied values")
    st.dataframe(
        comp_values.style.format({
            "Peer Median": "{:.2f}x",
            "Company Metric Value": "{:,.2f}",
            "Implied Value / Share": "${:,.2f}",
        }, na_rep="N/A").highlight_max(subset=["Peer Median"], color="#d9ead3"),
        use_container_width=True,
    )
else:
    st.info("Enter one or more peer tickers to calculate comparable-company valuation.")

comps_valid = comp_values["Implied Value / Share"].dropna() if not comp_values.empty else pd.Series(dtype=float)
comps_median_value = as_float(comps_valid.median()) if not comps_valid.empty else np.nan
compare_df = pd.DataFrame([
    {"Method": "Comps median of implied values", "Value / Share": comps_median_value},
    {"Method": "DCF intrinsic value", "Value / Share": dcf["intrinsic"] if dcf else np.nan},
    {"Method": "Current market price", "Value / Share": price},
])
st.subheader("Valuation comparison")
st.dataframe(compare_df.style.format({"Value / Share": "${:,.2f}"}, na_rep="N/A"), use_container_width=True)

if np.isfinite(comps_median_value) and dcf:
    st.write(
        f"**Why the estimates may differ:** DCF is driven by the chosen FCF base, forecast growth, WACC, and terminal growth. "
        f"Comparable-company values reflect how the selected peers are priced today and can be affected by differences in growth, "
        f"product mix, patent exposure, leverage, and one-off earnings. The current market price also incorporates expectations "
        f"not captured in a simple model. Treat these as scenarios, not precise targets."
    )

# -----------------------------
# Part 4: sensitivity analysis
# -----------------------------
st.header("Sensitivity Analysis")
st.write("Each cell recalculates DCF value per share. The base-case WACC and terminal-growth assumptions are included in the grid.")
if np.isfinite(base_fcf) and np.isfinite(shares) and shares > 0:
    wacc_grid = np.unique(np.round(np.clip(wacc + np.array([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]), -0.02, 0.30), 4))
    # Ensure selected WACC is explicitly present.
    wacc_grid = np.unique(np.append(wacc_grid, round(wacc, 4)))
    tg_grid = np.unique(np.round(np.clip(terminal_g + np.array([-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]), -0.05, 0.15), 4))
    tg_grid = np.unique(np.append(tg_grid, round(terminal_g, 4)))
    sens = pd.DataFrame(index=wacc_grid, columns=tg_grid, dtype=float)
    for r in wacc_grid:
        for g in tg_grid:
            if g < r:
                val = dcf_value(base_fcf, growth, g, r, forecast_years, cash, debt, shares)
                sens.loc[r, g] = val["intrinsic"] if val else np.nan
            else:
                sens.loc[r, g] = np.nan
    sens.index.name = "WACC \\ Terminal growth"
    sens.columns = [f"{g:.2%}" for g in sens.columns]
    sens_display = sens.copy()
    sens_display.index = [f"{x:.2%}" for x in sens_display.index]
    st.dataframe(
        sens_display.style.background_gradient(cmap="RdYlGn", axis=None).format("${:,.2f}", na_rep="N/A"),
        use_container_width=True,
    )
    st.caption("Green cells indicate higher modeled value; red cells indicate lower modeled value. N/A cells violate terminal growth < WACC.")
    base_case_value = dcf["intrinsic"] if dcf else np.nan
    st.write(f"**Base case:** WACC {wacc:.2%}, terminal growth {terminal_g:.2%}, value {fmt_money(base_case_value)} per share.")
    valid_vals = sens.to_numpy(dtype=float)
    finite_vals = valid_vals[np.isfinite(valid_vals)]
    if finite_vals.size:
        st.write(f"**Grid range:** {fmt_money(np.min(finite_vals))} to {fmt_money(np.max(finite_vals))} per share.")
        if np.isfinite(price):
            st.write(
                f"At this grid's assumptions, cells above the current price ({fmt_money(price)}) imply modeled undervaluation; "
                "cells below it imply modeled overvaluation."
            )
    st.info("In this Gordon-growth DCF, terminal growth and WACC often dominate because terminal value is a large share of enterprise value. Forecast FCF growth also matters by changing each projected cash flow.")
else:
    st.warning("Sensitivity grid requires a usable FCF base and shares outstanding.")

# -----------------------------
# Part 5 extension: football field
# -----------------------------
st.header("Football-Field Valuation")
ranges = []
if np.isfinite(base_fcf) and np.isfinite(shares) and shares > 0:
    valid_sens_values = sens.to_numpy(dtype=float) if "sens" in locals() else np.array([])
    valid_sens_values = valid_sens_values[np.isfinite(valid_sens_values)] if valid_sens_values.size else np.array([])
    if valid_sens_values.size:
        ranges.append(("DCF sensitivity range", float(np.min(valid_sens_values)), float(np.max(valid_sens_values))))
if not comp_values.empty:
    comp_vals = comp_values["Implied Value / Share"].dropna()
    if not comp_vals.empty:
        ranges.append(("Comps range", float(comp_vals.min()), float(comp_vals.max())))
low52 = as_float(info.get("fiftyTwoWeekLow"))
high52 = as_float(info.get("fiftyTwoWeekHigh"))
if not np.isfinite(low52) and not data.get("history", pd.DataFrame()).empty:
    low52 = as_float(data["history"]["Low"].min()) if "Low" in data["history"] else np.nan
if not np.isfinite(high52) and not data.get("history", pd.DataFrame()).empty:
    high52 = as_float(data["history"]["High"].max()) if "High" in data["history"] else np.nan
if np.isfinite(low52) and np.isfinite(high52):
    ranges.append(("52-week price range", low52, high52))

if ranges:
    fig = go.Figure()
    for label, low, high in ranges:
        if np.isfinite(low) and np.isfinite(high):
            fig.add_trace(go.Bar(
                name=label,
                y=[label],
                x=[max(0, high - low)],
                base=[low],
                orientation="h",
                hovertemplate=f"{label}<br>Low: $%{{base:.2f}}<br>High: $%{{x+base:.2f}}<extra></extra>",
            ))
    if np.isfinite(price):
        fig.add_vline(x=price, line_dash="dash", annotation_text=f"Market {fmt_money(price)}")
    if dcf:
        fig.add_vline(x=dcf["intrinsic"], line_dash="dot", annotation_text=f"DCF base {fmt_money(dcf['intrinsic'])}")
    fig.update_layout(
        title=f"{ticker} valuation ranges per share",
        xaxis_title="Value per share ($)",
        yaxis_title="Valuation method",
        barmode="overlay",
        height=360,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Football-field chart needs valid DCF sensitivity values, peer multiples, or 52-week price data.")

st.divider()
st.caption(
    "Educational use only, not investment advice. Financial data can be delayed, incomplete, or mislabeled by the source. "
    "Check filings and reconcile definitions (especially debt, cash, CapEx, EBITDA, and shares) before relying on results."
)
