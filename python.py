"""Explore SaaS sales, create charts, compare forecasts, and export Power BI data.

Place SaaS-Sales.csv beside this script before running it. All output files are
saved in the same folder; the downloaded source file is never overwritten.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Resolve file paths relative to this script so it works from any terminal folder.
PROJECT_DIR = Path(__file__).resolve().parent
SOURCE_FILE = PROJECT_DIR / "SaaS-Sales.csv"

# Load the source data as a table and remove accidental spaces in column labels.
df = pd.read_csv(SOURCE_FILE)
df.columns = df.columns.str.strip()

# Convert key fields to useful types. Invalid values become missing (NaT/NaN),
# making data-quality issues visible in the checks below.
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

# Inspect the size, sample rows, data types, missing values, duplicate row IDs,
# and the time span before starting the analysis.
print("Dataset size:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())
print("\nDuplicate Row IDs:", df["Row ID"].duplicated().sum())
print("\nOrder date range:", df["Order Date"].min(), "to", df["Order Date"].max())

# Derive calendar fields from Order Date so sales can be grouped by year,
# month, quarter, or full year-month. YearMonth is set to the month's first day.
df["Year"] = df["Order Date"].dt.year
df["Month"] = df["Order Date"].dt.month
df["Quarter"] = df["Order Date"].dt.quarter
df["YearMonth"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()

# Each CSV row is an order line, so add the line-item sales for each month to
# produce one monthly total for trend analysis and forecasting.
monthly_sales = (
    df.groupby("YearMonth", as_index=False)["Sales"]
      .sum()
      .rename(columns={"YearMonth": "Date", "Sales": "MonthlySales"})
)

# Print high-level totals and comparisons that help describe business results.
print("\nTotal sales:", round(df["Sales"].sum(), 2))
print("Total profit:", round(df["Profit"].sum(), 2))
print("Number of months:", len(monthly_sales))

print("\nSales by year:")
print(df.groupby("Year")["Sales"].sum().round(2))

print("\nTop 10 products by sales:")
print(
    df.groupby("Product")["Sales"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .round(2)
)

print("\nMonthly sales:")
print(monthly_sales.to_string(index=False))

# Save analysis-ready tables as separate files; the source CSV stays unchanged.
df.to_csv(PROJECT_DIR / "saas_sales_cleaned.csv", index=False)
monthly_sales.to_csv(PROJECT_DIR / "monthly_sales.csv", index=False)

# Prepare grouped metrics for the EDA charts.
yearly_sales = df.groupby("Year")["Sales"].sum().sort_index()
top_products = (
    df.groupby("Product")["Sales"]
      .sum()
      .sort_values()
      .tail(10)
)
industry_sales = df.groupby("Industry")["Sales"].sum().sort_values(ascending=False)

# Profit is also summarized by customer segment for possible future analysis.
profit_by_segment = df.groupby("Segment")["Profit"].sum().sort_values()

# Build a four-panel EDA figure: monthly trend, annual totals, leading products,
# and sales totals by industry.
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].plot(monthly_sales["Date"], monthly_sales["MonthlySales"], marker="o")
axes[0, 0].set_title("Monthly Sales Trend")
axes[0, 0].set_xlabel("Month")
axes[0, 0].set_ylabel("Sales")
axes[0, 0].tick_params(axis="x", rotation=45)

axes[0, 1].bar(yearly_sales.index.astype(str), yearly_sales.values, color="teal")
axes[0, 1].set_title("Sales by Year")
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Sales")

axes[1, 0].barh(top_products.index, top_products.values, color="cornflowerblue")
axes[1, 0].set_title("Top 10 Products by Sales")
axes[1, 0].set_xlabel("Sales")

axes[1, 1].bar(industry_sales.index.astype(str), industry_sales.values, color="darkorange")
axes[1, 1].set_title("Sales by Industry")
axes[1, 1].set_xlabel("Industry")
axes[1, 1].set_ylabel("Sales")
axes[1, 1].tick_params(axis="x", rotation=45)

# Tighten spacing, export the figure for the project/report, and show it on screen.
plt.tight_layout()
plt.savefig(PROJECT_DIR / "eda_sales_charts.png", dpi=150)
plt.show()

# Convert monthly totals into a regularly spaced time series (month-start dates).
sales_series = (
    monthly_sales.set_index("Date")["MonthlySales"]
    .asfreq("MS")
    .astype(float)
)

# Keep the latest 12 months (2023 in this dataset) aside as a holdout test set.
# Fit and compare methods using only earlier months to avoid looking ahead.
train = sales_series.iloc[:-12]
test = sales_series.iloc[-12:]

# A simple seasonal baseline predicts each test month using the same month
# from the previous year (for example, Jan 2023 uses Jan 2022 sales).
baseline_predictions = sales_series.shift(12).loc[test.index]

# Fit Holt-Winters on training months. It models a changing level, a damped
# upward/downward trend, and a repeating 12-month seasonal pattern.
hw_model = ExponentialSmoothing(
    train,
    trend="add",
    damped_trend=True,
    seasonal="add",
    seasonal_periods=12,
).fit()

# Generate one prediction for each test month and align the dates for comparison.
hw_predictions = hw_model.forecast(12)
hw_predictions.index = test.index

# Mean Absolute Error (MAE) is the average currency error. Mean Absolute
# Percentage Error (MAPE) expresses average absolute error as a percentage.
baseline_mae = (test - baseline_predictions).abs().mean()
hw_mae = (test - hw_predictions).abs().mean()
baseline_mape = ((test - baseline_predictions).abs() / test).mean() * 100
# Use Holt-Winters predictions here (not the baseline) to calculate its MAPE.
hw_mape = ((test - hw_predictions).abs() / test).mean() * 100

print("\nForecast evaluation on 2023:")
print(f"Seasonal baseline — MAE: {baseline_mae:.2f}, MAPE: {baseline_mape:.1f}%")
print(f"Holt-Winters      — MAE: {hw_mae:.2f}, MAPE: {hw_mape:.1f}%")

# Create the six month period immediately after the last historical month.
future_dates = pd.date_range(
    start=sales_series.index[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS",
)

# Select the model with the lower holdout MAE, then refit it using all observed
# months before producing the six-month projection.
if hw_mae < baseline_mae:
    final_model = ExponentialSmoothing(
        sales_series,
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=12,
    ).fit()
    future_values = final_model.forecast(6).values
    chosen_method = "Holt-Winters"
else:
    # For this seasonal baseline, repeat the latest six months from one year ago.
    future_values = sales_series.iloc[-12:-6].values
    chosen_method = "Seasonal baseline"

# Store future dates and predicted sales in a separate forecast table.
forecast = pd.DataFrame({"Date": future_dates, "ForecastSales": future_values})

print(f"\nSelected method: {chosen_method}")
print("\nSales forecast for the next 6 months:")
print(
    forecast.to_string(
        index=False,
        formatters={"ForecastSales": "{:,.2f}".format},
    )
)
forecast.to_csv(PROJECT_DIR / "sales_forecast.csv", index=False)

# Plot actual history, the 2023 validation forecast, and the post-data projection.
plt.figure(figsize=(12, 5))
plt.plot(sales_series.index, sales_series.values, label="Historical monthly sales")
plt.plot(
    test.index,
    hw_predictions.values,
    linestyle="--",
    label="Holt-Winters validation forecast",
)
plt.plot(
    future_dates,
    future_values,
    marker="o",
    linestyle="--",
    label="Six-month projection",
)
plt.title("SaaS Sales: History and Forecast")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PROJECT_DIR / "sales_forecast_chart.png", dpi=150)
plt.show()

# Combine monthly actuals and future predictions into one table for Power BI.
# The Series field becomes the legend that distinguishes Actual from Forecast.
actual_for_dashboard = monthly_sales.rename(
    columns={"MonthlySales": "Sales"}
).copy()
actual_for_dashboard["Series"] = "Actual"

forecast_for_dashboard = forecast.rename(
    columns={"ForecastSales": "Sales"}
).copy()
forecast_for_dashboard["Series"] = "Forecast"

powerbi_forecast = pd.concat(
    [actual_for_dashboard, forecast_for_dashboard],
    ignore_index=True,
)
powerbi_forecast.to_csv(PROJECT_DIR / "powerbi_sales_forecast.csv", index=False)
print("\nCreated powerbi_sales_forecast.csv")
