# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 12:51:17 2026

@author: ilsem
"""

"""
=============================================================
H30 Absolute Error Comparison Plots
Generates two plots for Horizon H30:
  - lag30 (lookback=30 days) from forecast_outputs5
  - lag60 (lookback=60 days) from forecast_outputs3

Both plots share the same X and Y axis scale for direct
visual comparison between the two look-back configurations.

Output saved in: forecast_outputs_graficasH30/
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

# ============================================================
# 1. PATHS CONFIGURATION
# ============================================================

BASE = r"C:\Users\ilsem\Documents\Thesis - memoire"

# Input: predictions CSV files from each lag configuration
PATH_LAG30 = os.path.join(BASE, "forecast_outputs5", "lag30",
                           "predictions_all.csv")
PATH_LAG60 = os.path.join(BASE, "forecast_outputs3",
                           "predictions_all.csv")

# Output folder for the generated plots
OUTPUT_FOLDER = os.path.join(BASE, "forecast_outputs_graficasH30")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. MODEL COLORS AND LABELS
# ============================================================

MODEL_COLORS = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
MODEL_LABELS = {
    'A': 'Model A (full sentiment)',
    'B': 'Model B (review metrics)',
    'C': 'Model C (baseline)'
}

# ============================================================
# 3. LOAD AND FILTER DATA FOR H30
# ============================================================

print("Loading prediction files ...")

# Load lag30 predictions and filter for H30 only
df_lag30 = pd.read_csv(PATH_LAG30)
df_lag30['date'] = pd.to_datetime(df_lag30[['year', 'month', 'day']])
h30_lag30 = df_lag30[df_lag30['horizon'] == 30].copy()

# Load lag60 predictions and filter for H30 only
df_lag60 = pd.read_csv(PATH_LAG60)
df_lag60['date'] = pd.to_datetime(df_lag60[['year', 'month', 'day']])
h30_lag60 = df_lag60[df_lag60['horizon'] == 30].copy()

print(f"  Lag30 H30: {len(h30_lag30):,} rows, "
      f"dates {h30_lag30['date'].min().date()} to "
      f"{h30_lag30['date'].max().date()}")
print(f"  Lag60 H30: {len(h30_lag60):,} rows, "
      f"dates {h30_lag60['date'].min().date()} to "
      f"{h30_lag60['date'].max().date()}")

# ============================================================
# 4. AGGREGATE ABSOLUTE ERROR BY DATE AND MODEL
# ============================================================

def aggregate_by_date(df):
    """
    Sum absolute error across all product categories per date per model.
    Returns a dict: model_label -> DataFrame with columns [date, absolute_error]
    """
    result = {}
    for model_label in ['A', 'B', 'C']:
        sub = df[df['model'] == model_label]
        if sub.empty:
            continue
        daily = (sub.groupby('date')['absolute_error']
                    .sum()
                    .reset_index()
                    .sort_values('date'))
        result[model_label] = daily
    return result

daily_lag30 = aggregate_by_date(h30_lag30)
daily_lag60 = aggregate_by_date(h30_lag60)

# ============================================================
# 5. COMPUTE SHARED AXIS RANGES
#
# Both plots will use the same X range (full span of both datasets)
# and the same Y range (global min/max of absolute error across both)
# to allow direct visual comparison.
# ============================================================

# Shared X range: from the earliest date across both datasets
# to the latest date across both datasets
all_dates = pd.concat([
    h30_lag30['date'], h30_lag60['date']
])
x_min = all_dates.min()
x_max = all_dates.max()

# Shared Y range: from global min to global max of absolute error
# Add a small buffer (5%) above and below for visual clarity


# Calculate Y range AFTER aggregation by date
# so the scale reflects the summed values across all categories
all_daily_errors = pd.concat([
    daily['absolute_error']
    for daily_data in [daily_lag30, daily_lag60]
    for daily in daily_data.values()
])
y_min = all_daily_errors.min() * 0.92
y_max = all_daily_errors.max() * 1.05


print(f"\nShared X range: {x_min.date()} to {x_max.date()}")
print(f"Shared Y range: {y_min:.1f} to {y_max:.1f}")

# ============================================================
# 6. HELPER: PLOT ONE CONFIGURATION
# ============================================================

def plot_h30(ax, daily_data, lag_label, x_min, x_max, y_min, y_max):
    """
    Plot absolute error lines for all three models on a given axes object.

    Parameters
    ----------
    ax         : matplotlib Axes
    daily_data : dict of model -> DataFrame (date, absolute_error)
    lag_label  : str — e.g. 'lag30 (lookback = 30 days)'
    x_min      : datetime — shared x axis minimum
    x_max      : datetime — shared x axis maximum
    y_min      : float — shared y axis minimum
    y_max      : float — shared y axis maximum
    """
    for model_label, daily in daily_data.items():
        ax.plot(daily['date'], daily['absolute_error'],
                color=MODEL_COLORS[model_label],
                linewidth=1.4,
                label=MODEL_LABELS[model_label])

    # Apply shared axis limits for consistent comparison
    
    ax.set_ylim(y_min, y_max)

    # X-axis: show a tick every 5 days for readability
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

    # Y-axis: clean integer ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.tick_params(axis='y', labelsize=8)

    ax.set_title(f'Absolute error by model — Horizon H30 | {lag_label}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Date', fontsize=9)
    ax.set_ylabel('Total absolute error (all categories)', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)


# ============================================================
# 7. GENERATE INDIVIDUAL PLOTS
# ============================================================

print("\nGenerating individual plots ...")

# --- Plot 1: lag30 ---
fig, ax = plt.subplots(figsize=(13, 5))
plot_h30(ax, daily_lag30,
         lag_label='lag30 (lookback = 30 days)',
         x_min=x_min, x_max=x_max,
         y_min=y_min, y_max=y_max)
plt.tight_layout()
out_path = os.path.join(OUTPUT_FOLDER, "H30_lag30.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Saved: H30_lag30.png")

# --- Plot 2: lag60 ---
fig, ax = plt.subplots(figsize=(13, 5))
plot_h30(ax, daily_lag60,
         lag_label='lag60 (lookback = 60 days)',
         x_min=x_min, x_max=x_max,
         y_min=y_min, y_max=y_max)
plt.tight_layout()
out_path = os.path.join(OUTPUT_FOLDER, "H30_lag60.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Saved: H30_lag60.png")

# ============================================================
# 8. GENERATE COMBINED PLOT (both configurations stacked)
# ============================================================

print("Generating combined plot ...")

fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

plot_h30(axes[0], daily_lag30,
         lag_label='lag30 (lookback = 30 days)',
         x_min=x_min, x_max=x_max,
         y_min=y_min, y_max=y_max)

plot_h30(axes[1], daily_lag60,
         lag_label='lag60 (lookback = 60 days)',
         x_min=x_min, x_max=x_max,
         y_min=y_min, y_max=y_max)

fig.suptitle('Absolute error by model — Horizon H30 | Look-back comparison',
             fontsize=13, fontweight='bold', y=1.01)

plt.tight_layout(h_pad=3.0)
out_path = os.path.join(OUTPUT_FOLDER, "H30_comparison_combined.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Saved: H30_comparison_combined.png")

# ============================================================
# 9. print results
# ============================================================

print("\n" + "=" * 60)
print(f"All plots saved to:\n  {OUTPUT_FOLDER}")
print("=" * 60)
print("\nFiles generated:")
for fn in sorted(os.listdir(OUTPUT_FOLDER)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_FOLDER, fn)) / 1024
    print(f"  {fn:45s} {size_kb:8.1f} KB")

