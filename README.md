# Merck & Co., Inc. (MRK) Equity Valuation Lab

**Created by:** Khulan Ganzorig  
**Program:** MS in Business Analytics, University at Buffalo

## Project Overview

The Merck & Co., Inc. (MRK) Equity Valuation Lab is an interactive financial analysis and equity valuation dashboard developed using Python and Streamlit.

The project evaluates Merck's financial performance and estimates its intrinsic stock value through fundamental financial statement analysis, Discounted Cash Flow (DCF) valuation, and Comparable Company Analysis (CCA).

The dashboard also incorporates sensitivity analysis and scenario modeling to examine how changes in key financial assumptions affect the estimated intrinsic value of the company.

## Project Objectives

- Analyze historical financial statements and key financial indicators.
- Evaluate operating performance and Free Cash Flow (FCF).
- Estimate intrinsic equity value using a DCF model.
- Compare MRK's valuation multiples with selected pharmaceutical peers.
- Examine the impact of WACC and terminal growth assumptions.
- Evaluate alternative Bull, Base, and Bear valuation scenarios.

## Dashboard Structure

### Part 1 — Fundamentals & Free Cash Flow

Analyzes annual financial statement data, including:

- Income Statement
- Balance Sheet
- Cash Flow Statement
- Revenue and Net Income
- Operating Cash Flow
- Capital Expenditures (CapEx)
- Free Cash Flow (FCF)
- Net Profit Margin and FCF Margin
- Current Ratio and Debt-to-Equity Ratio

**Free Cash Flow formula:**

FCF = Operating Cash Flow − |Capital Expenditures|

The model uses the average of the latest three available annual FCF observations as the starting point for the DCF valuation.

### Part 2 — Discounted Cash Flow (DCF) Valuation

Estimates MRK's intrinsic value per share using a DCF model.

Key adjustable assumptions include:

- Forecast FCF Growth Rate
- Terminal Growth Rate
- Risk-Free Rate
- Beta
- Equity Risk Premium
- Cost of Debt
- Tax Rate
- Weighted Average Cost of Capital (WACC)
- Forecast Period

The model projects future Free Cash Flow, discounts projected cash flows to present value, and calculates terminal value using the Gordon Growth Model.

**Terminal Value formula:**

Terminal Value = FCF in Final Forecast Year × (1 + Terminal Growth Rate) / (WACC − Terminal Growth Rate)

Enterprise value is adjusted for cash and debt to estimate equity value, which is divided by shares outstanding to calculate intrinsic value per share.

### Part 3 — Comparable Company Analysis

Compares Merck with selected pharmaceutical companies using market valuation multiples.

**Selected peer companies:**
- Pfizer (PFE)
- AbbVie (ABBV)
- Bristol Myers Squibb (BMY)
- Amgen (AMGN)
- Gilead Sciences (GILD)

Valuation multiples include:

- Price-to-Earnings (P/E)
- Enterprise Value-to-EBITDA (EV/EBITDA)
- Price-to-Sales (P/S)

Peer median multiples are applied to MRK's corresponding financial metrics to estimate implied share values.

The results are compared against the DCF valuation and current market price.

### Part 4 — Sensitivity Analysis

Examines how changes in two important DCF assumptions affect intrinsic value:

- Weighted Average Cost of Capital (WACC)
- Terminal Growth Rate

An interactive valuation heatmap displays the estimated intrinsic value per share under different combinations of assumptions.

The base-case assumptions are highlighted to make comparison easier.

The analysis also evaluates which assumption has a greater impact on the valuation range.

### Part 5 — Bull / Base / Bear Scenario Modeling

Evaluates three alternative valuation scenarios:

- **Bear Case:** Lower FCF growth, lower terminal growth, and higher WACC.
- **Base Case:** Moderate growth and baseline discount rate assumptions.
- **Bull Case:** Higher FCF growth, higher terminal growth, and lower WACC.

The dashboard presents a scenario comparison table and interactive chart to illustrate the effect of different assumptions on estimated intrinsic value per share.

## Technologies & Libraries

The project is developed using:

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- yfinance
- pathlib

## Data Source

Historical financial statements, company information, market prices, and valuation metrics are retrieved through the Yahoo Finance API using the `yfinance` Python library.

The application also includes a local CSV caching system to preserve previously retrieved financial data.

### Data Management

Financial data is stored in the `data_cache` directory.

The application supports two data retrieval modes:

1. **Yahoo Finance with CSV Fallback:** Retrieves current available data from Yahoo Finance and uses locally cached CSV files if retrieval fails.
2. **CSV Only:** Loads previously saved financial data without making a new Yahoo Finance request.

The application uses Streamlit caching to reduce repeated data requests.

## Installation & Setup

### 1. Install Python Libraries

```bash
pip install streamlit pandas numpy plotly yfinance
```

### 2. Project Structure

```text
MRK-Equity-Valuation-Lab/
│
├── app.py
├── README.md
│
└── data_cache/
    ├── MRK_income.csv
    ├── MRK_balance.csv
    ├── MRK_cashflow.csv
    └── MRK_info.csv
```

The `data_cache` directory and CSV files are generated automatically when financial data is successfully retrieved and saved.

### 3. Run the Application

Open Terminal, navigate to the project directory, and execute:

```bash
streamlit run app.py
```

The dashboard will open in your default web browser.

## Limitations & Assumptions

- Yahoo Finance data availability and financial statement labels may vary.
- Certain financial metrics may be unavailable, resulting in N/A values.
- DCF valuation is sensitive to assumptions such as FCF growth, WACC, and terminal growth.
- Comparable company valuations depend on peer selection and the availability of market multiples.
- Scenario assumptions are illustrative and do not represent guaranteed future performance.
- The model relies on available financial information and does not replace a detailed review of company filings.

## Disclaimer

This project is developed for educational and academic purposes only. It does not constitute investment advice, a recommendation to buy or sell securities, or a guarantee of future investment performance.

Financial data may be incomplete or delayed. Users should independently verify financial information, review company filings, and evaluate assumptions before making investment decisions.

---

**Merck & Co., Inc. (MRK) Equity Valuation Lab**  
*MS in Business Analytics | University at Buffalo*
