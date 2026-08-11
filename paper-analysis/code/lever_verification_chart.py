import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

teams = ["MTech", "MacGyver"]
configs = ["A: baseline", "B: +touchdown/climb\nratio improvement", "C: +cruise speed\nonly", "D: B + C\ncombined"]
success = {
    "MTech":    [76.3, 78.5, 76.0, 79.2],
    "MacGyver": [76.6, 80.9, 77.8, 77.7],
}
colors = ["#9fb4c7", "#2a7f3f", "#b3841a", "#1f5fa8"]

fig, ax = plt.subplots(figsize=(10, 6.5))

n_teams = len(teams)
n_cfg = len(configs)
bar_w = 0.19
group_gap = 1.3
x_base = np.arange(n_teams) * group_gap

for ci in range(n_cfg):
    xs = x_base + (ci - 1.5) * (bar_w + 0.025)
    vals = [success[t][ci] for t in teams]
    ax.bar(xs, vals, width=bar_w, color=colors[ci], label=configs[ci], zorder=3,
           edgecolor="white", linewidth=0.7)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x_base)
ax.set_xticklabels(teams, fontsize=13, fontweight="bold")
ax.set_ylabel("Simulated success rate (2500-mission precise measurement)", fontsize=10.5)
ax.set_ylim(70, 84)
ax.set_title("Controlled verification: landing-gear/descent-rate tuning is a real,\nconfirmed lever — cruise speed is not",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=9, frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.annotate("Largest single gain found:\n+4.3 pts from touchdown/climb\nratio alone", xy=(x_base[1] + (1-1.5)*(bar_w+0.025), 80.9),
            xytext=(x_base[1] + 0.55, 82.5), fontsize=9, fontweight="bold", color="#2a7f3f", ha="center",
            arrowprops=dict(arrowstyle="->", color="#2a7f3f", lw=1.2))
ax.annotate("Combining with cruise speed\nis WORSE than touchdown\nimprovement alone here",
            xy=(x_base[1] + (3-1.5)*(bar_w+0.025), 77.7),
            xytext=(x_base[1] + 0.55, 73.2), fontsize=9, fontweight="bold", color="#b23a2f", ha="center",
            arrowprops=dict(arrowstyle="->", color="#b23a2f", lw=1.2))

fig.text(0.02, 0.02,
         "Battery energy margin held fixed at the already-validated good value (2.5x) throughout, isolating these two new levers.",
         fontsize=8, color="#555555")
fig.text(0.02, 0.002,
         "Cruise speed's apparent benefit in the population comparison did not replicate under this controlled test.",
         fontsize=8, color="#555555")

plt.tight_layout(rect=[0, 0.11, 1, 1])
out_path = r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\paper\figures\lever_verification_chart.png"
plt.savefig(out_path, dpi=200)
print(f"saved: {out_path}")
