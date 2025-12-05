#!/usr/bin/env python
"""
Mission Time-Series Visualizations (One Folder Output)

Reads missions_timeseries.csv and, for each design_id, overlays all of its
missions on a SINGLE chart per metric. ALL charts are saved into one folder:

missions_timeseries_plots/
    design_<ID>_altitude.png
    design_<ID>_speed.png
    design_<ID>_distance.png
    design_<ID>_power.png
    design_<ID>_battery.png
    design_<ID>_phase.png
    
# © 2025 DBbun LLC — All rights reserved.
#
# This file is part of the DBbun Synthetic Missions System for the DARPA Lift Challenge.
# Its use is strictly limited to internal research and development associated with
# DARPA Lift Challenge participation. Redistribution, publication, commercial use,
# or post-challenge retention is prohibited without written authorization from DBbun LLC.
#
# License: DBbun DARPA Lift Challenge R&D License v1.0 (see LICENSE.md)
# Contact: contact@dbbun.com
# Website: https://dbbun.com/
# CAGE: 16VU3 | UEI: QY39Y38E6WG8

"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
INPUT_FILE = "missions_timeseries.csv"
OUTPUT_DIR = "missions_timeseries_plots"
STYLE = "default"     # clean white theme
TIME_COL = "t_s"      # matches generator output


# -------------------------------
# Helpers
# -------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_one(fig, filename: str):
    """Save into the one shared folder."""
    ensure_dir(OUTPUT_DIR)
    out_file = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


def overlay_plot(df_design, design_id, y_col, ylabel, title, prefix):
    """Overlay all missions of one design for a given metric and save."""
    if y_col not in df_design.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for mission_id, df_m in df_design.groupby("mission_id"):
        df_m = df_m.sort_values(by=TIME_COL)
        ax.plot(df_m[TIME_COL], df_m[y_col], linewidth=1.5, alpha=0.6)

    ax.set_title(f"{title} — {design_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    save_one(fig, f"{prefix}_{design_id}.png")


def overlay_phase(df_design, design_id):
    """Stacked categorical visualization of phase sequences."""
    if "phase" not in df_design.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    phases = sorted(df_design["phase"].astype(str).unique())
    p2i = {p: i for i, p in enumerate(phases)}

    for idx, (mission_id, df_m) in enumerate(df_design.groupby("mission_id")):
        df_m = df_m.sort_values(by=TIME_COL)
        y_vals = df_m["phase"].astype(str).map(p2i) + idx * (len(phases) + 1)
        ax.step(df_m[TIME_COL], y_vals, linewidth=1.2, alpha=0.7, where='post')

    ax.set_title(f"Phase Timeline — {design_id} (all missions)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase (stacked)")
    ax.grid(True, linestyle="--", alpha=0.3)

    save_one(fig, f"phase_{design_id}.png")


# -------------------------------
# MAIN
# -------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found.")
        return

    plt.style.use(STYLE)
    ts = pd.read_csv(INPUT_FILE)

    required = {"design_id", "mission_id", TIME_COL}
    missing = required - set(ts.columns)
    if missing:
        print(f"ERROR: missing required columns: {missing}")
        print("Found:", list(ts.columns))
        return

    ts = ts.sort_values(by=["design_id", "mission_id", TIME_COL])

    for design_id, df_design in ts.groupby("design_id"):

        overlay_plot(df_design, design_id,
                     "altitude_m", "Altitude (m)",
                     "Altitude vs Time",
                     prefix="altitude")

        overlay_plot(df_design, design_id,
                     "speed_mps", "Speed (m/s)",
                     "Speed vs Time",
                     prefix="speed")

        overlay_plot(df_design, design_id,
                     "distance_m", "Horizontal Distance (m)",
                     "Horizontal Distance vs Time",
                     prefix="distance")

        overlay_plot(df_design, design_id,
                     "power_W", "Power (W)",
                     "Instantaneous Power vs Time",
                     prefix="power")

        overlay_plot(df_design, design_id,
                     "battery_remaining_Wh", "Battery Remaining (Wh)",
                     "Battery Energy Remaining vs Time",
                     prefix="battery")

        overlay_phase(df_design, design_id)

    print("\nAll plots generated →", OUTPUT_DIR)


if __name__ == "__main__":
    main()
