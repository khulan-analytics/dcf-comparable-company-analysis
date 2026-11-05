cash_only = get_series(
    balance,
    "Cash And Cash Equivalents"
)

combined_cash = get_series(
    balance,
    "Cash Cash Equivalents And Short Term Investments"
)

if not cash_only.empty:
    cash_series = cash_only
    short_investments = get_series(
        balance,
        "Other Short Term Investments",
        "Short Term Investments"
    )
else:
    cash_series = combined_cash
    short_investments = pd.Series(dtype=float)
