#!/usr/bin/env python
"""
Designs Plot Generator

Automatically loads 'designs.csv' in the same folder
and generates summary charts into 'designs_plots/'.

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
INPUT_FILE = "designs.csv"     # Must be in the same directory
OUTPUT_DIR = "designs_plots"   # Auto-created
STYLE = "default"              # Clean white style


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


def build_animal_counts(df: pd.DataFrame) -> Counter:
    def split_animals(x):
        if not isinstance(x, str):
            return []
        return [a.strip() for a in x.split(",") if a.strip()]

    counter = Counter()
    for lst in df["animals"].fillna("").apply(split_animals):
        counter.update(lst)
    return counter


# ------------------------------
# Main
# ------------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found in this folder.")
        return

    plt.style.use(STYLE)
    designs = pd.read_csv(INPUT_FILE)

    # ===== HISTOGRAMS (numeric) =====
    histograms = [
        ("empty_mass_kg", "Empty Mass Distribution (kg)", "Empty Mass (kg)", "steelblue"),
        ("payload_mass_kg", "Payload Mass Distribution (kg)", "Payload Mass (kg)", "seagreen"),
        ("payload_to_aircraft_ratio", "Payload-to-Aircraft Ratio Distribution", "Ratio", "darkorchid"),
        ("trait_count", "Trait Count Distribution", "Trait Count", "dodgerblue"),
        ("design_qualifying_score", "Design Qualifying Score Distribution", "Qualifying Score", "orange"),
    ]

    for col, title, xlabel, color in histograms:
        if col in designs.columns:
            data = designs[col].dropna()

            fig, ax = plt.subplots(figsize=(8, 5))

            if col == "trait_count":
                # Integer-only bins and ticks
                data = data.astype(int)
                min_tc = data.min()
                max_tc = data.max()
                bins = range(min_tc, max_tc + 2)  # edges at integers
                ax.hist(data, bins=bins, color=color, edgecolor="black", align="left")
                ax.set_xticks(range(min_tc, max_tc + 1))
            else:
                ax.hist(data, bins=20, color=color, edgecolor="black")

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            save_fig(fig, f"hist_{col}.png")

    # ===== FIXED ROTOR COUNT DISTRIBUTION (unique integer categories only) =====
    if "rotor_count" in designs.columns:
        counts = designs["rotor_count"].dropna().astype(int)
        if not counts.empty:
            unique_counts = sorted(counts.unique())  # Only real rotor values
            bins = [u - 0.5 for u in unique_counts] + [unique_counts[-1] + 0.5]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(counts, bins=bins, color="#ff6f4c", edgecolor="black")
            
            ax.set_title("Rotor Count Distribution", fontsize=18)
            ax.set_xlabel("Rotor Count", fontsize=14)
            ax.set_ylabel("Count", fontsize=14)
            ax.set_xticks(unique_counts)
            ax.grid(True, linestyle="--", alpha=0.35)
            
            save_fig(fig, "rotor_count_distribution.png")

    # ===== BAR CHARTS =====
    if "energy_system_type" in designs.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        designs["energy_system_type"].value_counts().plot(
            kind="bar", ax=ax, color="gray", edgecolor="black"
        )
        ax.set_title("Energy System Type Counts")
        ax.set_xlabel("Energy System Type")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        save_fig(fig, "energy_system_type_counts.png")

    # Animal inspiration
    if "animals" in designs.columns:
        animal_counts = build_animal_counts(designs)
        if animal_counts:
            labels = list(animal_counts.keys())
            values = [animal_counts[k] for k in labels]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(labels, values, color="goldenrod", edgecolor="black")
            ax.set_title("Animal Inspiration Frequency")
            ax.set_ylabel("Count")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(True, alpha=0.3)
            save_fig(fig, "animal_inspiration_frequency.png")

    # ===== SCATTERS =====
    if {"trait_count", "design_qualifying_score"}.issubset(designs.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            designs["trait_count"],
            designs["design_qualifying_score"],
            alpha=0.7,
            color="dodgerblue",
            edgecolor="none",
        )
        ax.set_title("Trait Count vs Qualifying Score")
        ax.set_xlabel("Trait Count")
        ax.set_ylabel("Qualifying Score")
        ax.grid(True, alpha=0.3)
        save_fig(fig, "scatter_trait_vs_score.png")

    if {"mtow_kg", "payload_mass_kg"}.issubset(designs.columns):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(
            designs["mtow_kg"],
            designs["payload_mass_kg"],
            alpha=0.7,
            color="firebrick",
            edgecolor="none",
        )
        ax.set_title("MTOW vs Payload Mass")
        ax.set_xlabel("MTOW (kg)")
        ax.set_ylabel("Payload Mass (kg)")
        ax.grid(True, alpha=0.3)
        save_fig(fig, "scatter_mtow_vs_payload.png")

    # ===== CORRELATION HEATMAP =====
    cols = [
        "empty_mass_kg",
        "payload_mass_kg",
        "payload_to_aircraft_ratio",
        "rotor_count",
        "mtow_kg",
        "trait_count",
        "design_qualifying_score",
    ]
    present_cols = [c for c in cols if c in designs.columns]
    if len(present_cols) >= 2:
        corr = designs[present_cols].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(present_cols)))
        ax.set_yticks(np.arange(len(present_cols)))
        ax.set_xticklabels(present_cols, rotation=45, ha="right")
        ax.set_yticklabels(present_cols)
        ax.set_title("Correlation Heatmap")
        fig.colorbar(im, ax=ax)
        save_fig(fig, "correlation_heatmap.png")

    print("\nAll plots generated successfully!")
    print(f"Check folder: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
