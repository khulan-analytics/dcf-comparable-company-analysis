import yfinance as yf

mrk = yf.Ticker("MRK")

print("PRICE DATA:")
print(mrk.history(period="5d"))

print("\nINCOME STATEMENT:")
print(mrk.income_stmt)

print("\nBALANCE SHEET:")
print(mrk.balance_sheet)

print("\nCASH FLOW:")
print(mrk.cashflow)
