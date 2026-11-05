import yfinance as yf
import streamlit as st

ticker = yf.Ticker("MRK")

st.subheader("MRK Data Diagnostics")

st.write("Company information:")
st.write(ticker.info)

st.write("Income Statement:")
st.dataframe(ticker.income_stmt)

st.write("Balance Sheet:")
st.dataframe(ticker.balance_sheet)

st.write("Cash Flow:")
st.dataframe(ticker.cashflow)
