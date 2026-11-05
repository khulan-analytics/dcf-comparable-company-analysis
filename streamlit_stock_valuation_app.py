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
