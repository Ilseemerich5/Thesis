# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:38:57 2026

@author: ilsem
"""

"""
=============================================================
Training History Comparison Plots — Model A
Compares training curves across different look-back windows
for each forecasting horizon where Model A performs best.

Output: 4 PNG files saved in loss_plots_per_horizon_model_a/
  - history_comparison_H1.png   (lag1, lag14, lag60)
  - history_comparison_H7.png   (lag7, lag30, lag60)
  - history_comparison_H14.png  (lag30 only)
  - history_comparison_all.png  (all above in one figure)
=============================================================
"""

"""
=============================================================
Training History Comparison Plots — Model A
Compares training curves across different look-back windows
for each forecasting horizon where Model A performs best.

Output: 4 PNG files saved in loss_plots_per_horizon_model_a/
  - history_comparison_H1.png   (lag1, lag14, lag60)
  - history_comparison_H7.png   (lag7, lag30, lag60)
  - history_comparison_H14.png  (lag30 only)
  - history_comparison_all.png  (all above in one figure)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
# 1. PATHS CONFIGURATION
# ============================================================

# Root paths for each lag configuration
BASE = r"C:\Users\ilsem\Documents\Thesis - memoire"

PATHS = {
    'lag1':  os.path.join(BASE, "forecast_outputs4", "lag1"),
    'lag7':  os.path.join(BASE, "forecast_outputs4", "lag7"),
    'lag14': os.path.join(BASE, "forecast_outputs4", "lag14"),
    'lag30': os.path.join(BASE, "forecast_outputs4", "lag30"),
    'lag60': os.path.join(BASE, "forecast_outputs3"),   # lag60 from previous run
}

# Output folder for all generated plots
OUTPUT_FOLDER = os.path.join(BASE, "loss_plots_per_horizon_model_a")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. DEFINE WHICH HISTORY FILES TO COMPARE PER HORIZON
#
# Structure: horizon -> list of (lag_label, lag_key, color)
# Only configurations where Model A wins are included.
# ============================================================

HORIZON_CONFIGS = {
    'H1': [
        ('Lag 1',  'lag1',  '#1f77b4'),
        ('Lag 14', 'lag14', '#ff7f0e'),
        ('Lag 60', 'lag60', '#2ca02c'),
    ],
    'H7': [
        ('Lag 7',  'lag7',  '#1f77b4'),
        ('Lag 30', 'lag30', '#ff7f0e'),
        ('Lag 60', 'lag60', '#2ca02c'),
    ],
    'H14': [
        ('Lag 30', 'lag30', '#1f77b4'),
    ],
}

# Numeric horizon values (used to build file names)
HORIZON_VALUES = {
    'H1':  1,
    'H7':  7,
    'H14': 14,
}


# ============================================================
# 3. HELPER: LOAD HISTORY CSV
# ============================================================

def load_history(lag_key, horizon_num):
    """
    Load the training history CSV for Model A at a given
    lag configuration and forecasting horizon.

    Parameters
    ----------
    lag_key      : str — e.g. 'lag14'
    horizon_num  : int — e.g. 1, 7, 14

    Returns
    -------
    DataFrame with columns 'loss' and optionally 'val_loss',
    or None if the file is not found.
    """
    folder   = PATHS[lag_key]
    filename = f"history_Model_A_H{horizon_num}.csv"
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(f"  WARNING: File not found -> {filepath}")
        return None

    df = pd.read_csv(filepath)
    print(f"  Loaded: {filepath} ({len(df)} epochs)")
    return df


# ============================================================
# 4. HELPER: PLOT ONE HORIZON COMPARISON
# ============================================================

def plot_horizon_comparison(ax, horizon_label, configs, horizon_num,
                             show_xlabel=True):
    """
    Draw training and validation loss curves for multiple lag
    configurations on the same axes object.

    Parameters
    ----------
    ax            : matplotlib Axes object
    horizon_label : str  — e.g. 'H1'
    configs       : list of (label, lag_key, color)
    horizon_num   : int  — numeric horizon value
    show_xlabel   : bool — whether to show x-axis label
    """
    for lag_label, lag_key, color in configs:

        df = load_history(lag_key, horizon_num)
        if df is None:
            continue

        epochs = np.arange(1, len(df) + 1)

        # Plot train loss as solid line
        ax.plot(epochs, df['loss'],
                color=color, linewidth=1.5, linestyle='-',
                label=f'{lag_label} — Train')

        # Plot val loss as dashed line if available
        if 'val_loss' in df.columns:
            val = df['val_loss'].dropna()
            if len(val) > 0:
                ax.plot(np.arange(1, len(val) + 1), val,
                        color=color, linewidth=1.5, linestyle='--',
                        label=f'{lag_label} — Val', alpha=0.7)

        # Mark the best epoch with a vertical dotted line
        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            best_epoch = df['val_loss'].idxmin() + 1
        else:
            best_epoch = df['loss'].idxmin() + 1

        ax.axvline(x=best_epoch, color=color, linestyle=':',
                   alpha=0.5, linewidth=1.0)

    # Axis formatting
    ax.set_title(f'Horizon {horizon_label} — Model A (Look-back comparison)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss (MSE)', fontsize=9)
    if show_xlabel:
        ax.set_xlabel('Epoch', fontsize=9)

    # Single column legend — no gray annotation text
    ax.legend(fontsize=8, loc='upper right',
              ncol=1,              # single column
              framealpha=0.8)

    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


# ============================================================
# 5. GENERATE INDIVIDUAL PLOTS (one per horizon)
# ============================================================

print("=" * 60)
print("GENERATING INDIVIDUAL HORIZON PLOTS")
print("=" * 60)

for horizon_label, configs in HORIZON_CONFIGS.items():

    horizon_num = HORIZON_VALUES[horizon_label]
    print(f"\nProcessing {horizon_label} ...")

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_horizon_comparison(
        ax=ax,
        horizon_label=horizon_label,
        configs=configs,
        horizon_num=horizon_num,
        show_xlabel=True
    )

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_FOLDER,
                            f"history_comparison_{horizon_label}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: history_comparison_{horizon_label}.png")


# ============================================================
# 6. GENERATE COMBINED PLOT (all horizons in one figure)
# ============================================================

print("\nGenerating combined plot ...")

fig = plt.figure(figsize=(15, 13))

# Use GridSpec for a clean 3-panel layout
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

axes = [fig.add_subplot(gs[i]) for i in range(3)]

horizon_list = list(HORIZON_CONFIGS.items())

for i, (horizon_label, configs) in enumerate(horizon_list):
    horizon_num = HORIZON_VALUES[horizon_label]
    show_xlabel = (i == len(horizon_list) - 1)   # only show on last panel

    plot_horizon_comparison(
        ax=axes[i],
        horizon_label=horizon_label,
        configs=configs,
        horizon_num=horizon_num,
        show_xlabel=show_xlabel
    )

# Global title for the combined figure
fig.suptitle(
    'Training History Comparison — Model A across Look-back Windows',
    fontsize=13, fontweight='bold', y=1.01
)

out_path = os.path.join(OUTPUT_FOLDER, "history_comparison_all.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  -> Saved: history_comparison_all.png")


# ============================================================
# 7. print results
# ============================================================

print("\n" + "=" * 60)
print(f"All plots saved to:\n  {OUTPUT_FOLDER}")
print("=" * 60)
print("\nFiles generated:")
for fn in sorted(os.listdir(OUTPUT_FOLDER)):
    size_kb = os.path.getsize(os.path.join(OUTPUT_FOLDER, fn)) / 1024
    print(f"  {fn:45s} {size_kb:8.1f} KB")

