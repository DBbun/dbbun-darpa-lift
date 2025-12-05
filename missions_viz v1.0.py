#!/usr/bin/env python
"""
Missions Plot Generator

Automatically loads 'missions.csv' in the same folder
and generates summary charts into 'missions_plots/'.

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
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------
# Configuration
# ------------------------------
INPUT_FILE = "missions.csv"     # Must be in the same directory
OUTPUT_DIR = "missions_plots"   # Auto-created
STYLE = "default"               # Clean white style


# ------------------------------
# Helpers
# ------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(fig: plt.Figure, name: str):
    ensure_dir(OUTPUT_DIR)
    filepath = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


# ------------------------------
# Main
# ------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found in this folder.")
        return

    plt.style.use(STYLE)
    missions = pd.read_csv(INPUT_FILE)

    # Coerce booleans if they are 0/1
    for col in ["success", "is_qualifying_run"]:
        if col in missions.columns:
            missions[col] = missions[col].astype(int)

    # ===== HISTOGRAMS (numeric) =====
    hist_specs = [
        ("wind_speed_kts", "Wind Speed Distribution", "Wind Speed (kts)", "steelblue"),
        ("turbulence_index", "Turbulence Index Distribution", "Turbulence Index", "seagreen"),
        ("total_time_s", "Total Mission Time Distribution", "Total Time (s)", "darkorchid"),
        ("energy_used_Wh", "Energy Used Distribution", "Energy Used (Wh)", "tomato"),
        ("battery_energy_remaining_Wh", "Battery Energy Remaining Distribution", "Remaining Energy (Wh)", "goldenrod"),
        ("payload_to_aircraft_ratio", "Payload-to-Aircraft Ratio Distribution", "Payload-to-Aircraft Ratio", "dodgerblue"),
        ("qualifying_score", "Qualifying Score Distribution", "Qualifying Score", "orange"),
    ]

    for col, title, xlabel, color in hist_specs:
        if col in missions.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(missions[col].dropna(), bins=30, color=color, edgecolor="black")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            save_fig(fig, f"hist_{col}.png")

    # ===== SUCCESS / FAILURE BAR CHART =====
    if "success" in missions.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        counts = missions["success"].value_counts().sort_index()
        labels = ["Fail", "Success"]
        values = [counts.get(0, 0), counts.get(1, 0)]
        ax.bar(labels, values, color=["firebrick", "seagreen"], edgecolor="black")
        ax.set_title("Mission Success vs Failure")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        save_fig(fig, "success_vs_failure.png")

    # ===== FAILURE PHASE COUNTS (only failed missions) =====
    if {"success", "failure_phase"}.issubset(missions.columns):
        failed = missions[missions["success"] == 0]
        if not failed.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts = failed["failure_phase"].fillna("unknown").value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="gray", edgecolor="black")
            ax.set_title("Failure Phase Counts (Failed Missions)")
            ax.set_xlabel("Failure Phase")
            ax.set_ylabel("Count")
            ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            save_fig(fig, "failure_phase_counts.png")

    # ===== FAILURE REASON COUNTS (only failed missions) =====
    if {"success", "failure_reason"}.issubset(missions.columns):
        failed = missions[missions["success"] == 0]
        if not failed.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            counts = failed["failure_reason"].fillna("unknown").value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="#ff9966", edgecolor="black")
            ax.set_title("Failure Reason Counts (Failed Missions)")
            ax.set_xlabel("Failure Reason")
            ax.set_ylabel("Count")
            ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            save_fig(fig, "failure_reason_counts.png")

    # ===== RULE VIOLATION COUNTS (non-empty only) =====
    if "rule_violation" in missions.columns:
        non_empty = missions["rule_violation"].fillna("").replace("", np.nan).dropna()
        if not non_empty.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts = non_empty.value_counts()
            ax.bar(counts.index.astype(str), counts.values, color="#9999ff", edgecolor="black")
            ax.set_title("Rule Violation Counts")
            ax.set_xlabel("Rule Violation")
            ax.set_ylabel("Count")
            ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right")
            ax.grid(True, linestyle="--", alpha=0.3, axis="y")
            save_fig(fig, "rule_violation_counts.png")

    # ===== SCATTER PLOTS =====
    def scatter_if(cols, filename, title, xlabel, ylabel, color="dodgerblue"):
        x_col, y_col = cols
        if {x_col, y_col}.issubset(missions.columns):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                missions[x_col], missions[y_col],
                alpha=0.6, color=color, edgecolor="none"
            )
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            save_fig(fig, filename)

    scatter_if(
        ("total_time_s", "energy_used_Wh"),
        "scatter_total_time_vs_energy_used.png",
        "Total Time vs Energy Used",
        "Total Time (s)",
        "Energy Used (Wh)",
        color="teal",
    )

    scatter_if(
        ("wind_speed_kts", "turbulence_index"),
        "scatter_wind_vs_turbulence.png",
        "Wind Speed vs Turbulence Index",
        "Wind Speed (kts)",
        "Turbulence Index",
        color="purple",
    )

    scatter_if(
        ("payload_to_aircraft_ratio", "qualifying_score"),
        "scatter_payload_ratio_vs_qualifying_score.png",
        "Payload Ratio vs Qualifying Score",
        "Payload-to-Aircraft Ratio",
        "Qualifying Score",
        color="darkorange",
    )

    # ===== BOX PLOT: TOTAL TIME BY SUCCESS =====
    if {"success", "total_time_s"}.issubset(missions.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        data = [
            missions.loc[missions["success"] == 0, "total_time_s"].dropna(),
            missions.loc[missions["success"] == 1, "total_time_s"].dropna(),
        ]
        ax.boxplot(data, labels=["Fail", "Success"], showfliers=False)
        ax.set_title("Total Mission Time by Outcome")
        ax.set_ylabel("Total Time (s)")
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        save_fig(fig, "box_total_time_by_success.png")

    print("\nAll mission plots generated successfully!")
    print(f"Check folder: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
