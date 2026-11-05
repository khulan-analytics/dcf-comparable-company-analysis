import yfinance as yf
import streamlit as st

for symbol in ["MRK", "AAPL", "MSFT"]:
    stock = yf.Ticker(symbol)

    st.subheader(symbol)

    st.write("Price:")
    st.write(stock.history(period="5d"))

    st.write("Income Statement:")
    st.write(stock.income_stmt)

    st.write("Balance Sheet:")
    st.write(stock.balance_sheet)

    st.write("Cash Flow:")
    st.write(stock.cashflow)
