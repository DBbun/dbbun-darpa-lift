import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

teams = ["AVIDrone", "MTech", "Xtreme Aerial", "H-Squared", "MacGyver", "DefendTex"]
real_ratio = [3.85, 3.66, 3.44, 2.96, 2.49, 6.11]
placement = ["1st\n$1.25M", "2nd\n$750K", "3rd\n$500K", "4th", "5th", "crashed,\ndelisted"]
placement_color = ["#2a7f3f", "#2a7f3f", "#2a7f3f", "#4a4a4a", "#4a4a4a", "#b23a2f"]

mc_mean = [0.734, 0.719, 0.713, 0.722, 0.740, 0.700]
mc_stdev = [0.083, 0.106, 0.097, 0.092, 0.086, 0.106]
mc_min = [0.457, 0.307, 0.429, 0.420, 0.435, 0.410]
mc_max = [0.861, 0.876, 0.860, 0.881, 0.866, 0.862]

# noise-corrected population "true top tier" benchmark (n=40 sample, 2500 missions each,
# re-measured from designs that scored a "perfect" 1.0 on the original 10-mission run)
pop_mean = 0.784
pop_min = 0.577
pop_max = 0.859

x = np.arange(len(teams))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                gridspec_kw={"height_ratios": [1, 1.3]})

# --- Panel A: real achieved ratio ---
bars = ax1.bar(x, real_ratio, color=placement_color, width=0.55, zorder=3)
for xi, r, p in zip(x, real_ratio, placement):
    ax1.text(xi, r + 0.12, f"{r:.2f}:1", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_ylabel("Real payload-to-aircraft ratio", fontsize=10)
ax1.set_title("Real DARPA Lift Challenge outcomes vs. simulator-predicted reliability\n(6 real designs, reconstructed and run through the actual mission engine)",
              fontsize=11, fontweight="bold")
ax1.set_ylim(0, 7.2)
ax1.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
for xi, p in zip(x, placement):
    ax1.text(xi, -0.55, p, ha="center", va="top", fontsize=8, color="#333333")

# --- Panel B: simulated rank_score, Monte Carlo mean +/- stdev, with min-max whiskers ---
err_lo = [min(m, s) for m, s in zip(mc_mean, mc_stdev)]  # clip lower error at 0
err_hi = mc_stdev
ax2.errorbar(x, mc_mean, yerr=[err_lo, err_hi], fmt="o", color="#1f5fa8",
             ecolor="#1f5fa8", elinewidth=2.2, capsize=6, markersize=8,
             label="Monte Carlo v2 mean rank_score \u00b1 1 stdev (100 draws \u00d7 400 missions, energy-capacity-corrected)", zorder=4)
ax2.scatter(x, mc_min, marker="_", s=220, color="#8fa8c9", zorder=3)
ax2.scatter(x, mc_max, marker="_", s=220, color="#8fa8c9", zorder=3)
for xi, lo, hi in zip(x, mc_min, mc_max):
    ax2.plot([xi, xi], [lo, hi], color="#8fa8c9", linewidth=1, linestyle="--", zorder=2)

ax2.axhspan(pop_min, pop_max, color="#e8b84b", alpha=0.25, zorder=1,
            label="Noise-corrected population benchmark range\n(designs that scored 1.0 on 10 missions, re-measured on 2500)")
ax2.axhline(pop_mean, color="#b3841a", linestyle="--", linewidth=1.3, zorder=2,
            label=f"Population benchmark mean = {pop_mean:.3f}")

ax2.set_ylabel("Simulated design_rank_score", fontsize=10)
ax2.set_ylim(-0.05, 1.0)
ax2.set_xticks(x)
ax2.set_xticklabels(teams, fontsize=10)
ax2.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
ax2.legend(loc="lower center", fontsize=8, framealpha=0.9)

fig.text(0.01, 0.005,
         "Light dashed whiskers = full min-max range across 100 parameter draws. "
         "Green/gray/red labels = real prize tier / non-podium / crashed.",
         fontsize=7.5, color="#555555")

plt.tight_layout(rect=[0, 0.02, 1, 1])
out_path = r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\paper\figures\real_vs_simulated_comparison.png"
plt.savefig(out_path, dpi=200)
print(f"saved: {out_path}")
