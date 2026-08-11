import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

teams = ["AVIDrone", "MTech", "Xtreme Aerial", "H-Squared", "MacGyver", "DefendTex"]
real_ratio = np.array([3.85, 3.66, 3.44, 2.96, 2.49, 6.11])
placement = ["1st, $1.25M", "2nd, $750K", "3rd, $500K", "4th", "5th", "crashed, delisted"]
color = ["#2a7f3f", "#2a7f3f", "#2a7f3f", "#4a4a4a", "#4a4a4a", "#b23a2f"]

mean_rank = np.array([0.734, 0.719, 0.713, 0.722, 0.740, 0.700])
stdev = np.array([0.083, 0.106, 0.097, 0.092, 0.086, 0.106])

fig, ax = plt.subplots(figsize=(8, 6.5))

ax.errorbar(real_ratio, mean_rank, yerr=stdev, fmt="none", ecolor="#999999",
            elinewidth=1.5, capsize=5, zorder=2)
for r, m, c, name in zip(real_ratio, mean_rank, color, teams):
    ax.scatter([r], [m], s=140, color=c, zorder=3, edgecolor="white", linewidth=1.2)

offsets = {
    "AVIDrone": (0.08, 0.012), "MTech": (0.08, -0.022), "Xtreme Aerial": (0.08, 0.012),
    "H-Squared": (0.08, -0.022), "MacGyver": (0.08, 0.012), "DefendTex": (-0.55, 0.012),
}
for r, m, name in zip(real_ratio, mean_rank, teams):
    dx, dy = offsets[name]
    ax.annotate(name, (r, m), xytext=(r + dx, m + dy), fontsize=9.5, fontweight="bold")

# linear trend across the 5 real winners only (excludes DefendTex, which didn't score)
winners_mask = np.array([True, True, True, True, True, False])
z = np.polyfit(real_ratio[winners_mask], mean_rank[winners_mask], 1)
xs = np.linspace(2.2, 4.1, 20)
ax.plot(xs, np.polyval(z, xs), linestyle="--", color="#1f5fa8", linewidth=1.3, alpha=0.7,
        label="Linear fit, 5 real winners only (Spearman \u03c1 = \u22120.30, n=5)")

ax.set_xlabel("Real payload-to-aircraft ratio achieved", fontsize=11)
ax.set_ylabel("Simulated mean design_rank_score\n(Monte Carlo v2, energy-capacity-corrected, 100 draws \u00d7 400 missions)", fontsize=10)
ax.set_title("Real competition outcome vs. simulator-predicted reliability\ndoes the simulator's ranking track real placement?", fontsize=11.5, fontweight="bold")
ax.set_ylim(0.55, 0.85)
ax.set_xlim(2.0, 6.6)
ax.grid(linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

fig.text(0.01, 0.02,
         "Error bars = \u00b1 1 stdev across 100 plausible-parameter draws per design.",
         fontsize=7.5, color="#555555")
fig.text(0.01, 0.005,
         "Green = podium finish, gray = scored but off-podium, red = crashed/delisted.",
         fontsize=7.5, color="#555555")

ax.set_xlim(2.0, 6.8)
plt.tight_layout(rect=[0, 0.045, 1, 1])
out_path = r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\paper\figures\correlation_scatter.png"
plt.savefig(out_path, dpi=200)
print(f"saved: {out_path}")
