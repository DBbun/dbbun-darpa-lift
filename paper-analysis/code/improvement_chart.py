import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

teams = ["AVIDrone", "MTech", "Xtreme Aerial", "H-Squared", "MacGyver"]
configs = ["A: baseline", "B: +battery margin", "C: B + 8% lighter airframe"]
success = {
    "AVIDrone":      [73.0, 76.6, 72.3],
    "MTech":         [74.6, 75.0, 76.1],
    "Xtreme Aerial": [75.8, 76.2, 75.3],
    "H-Squared":     [75.3, 74.8, 74.3],
    "MacGyver":      [76.4, 76.9, 74.8],
}
ratio = {
    "AVIDrone":      [3.85, 3.85, 4.18],
    "MTech":         [3.66, 3.66, 3.98],
    "Xtreme Aerial": [3.44, 3.44, 3.73],
    "H-Squared":     [2.96, 2.96, 3.22],
    "MacGyver":      [2.49, 2.49, 2.71],
}

colors = ["#9fb4c7", "#1f5fa8", "#b23a2f"]
fig, ax = plt.subplots(figsize=(11, 6.5))

n_teams = len(teams)
n_cfg = len(configs)
bar_w = 0.24
group_gap = 1.0
x_base = np.arange(n_teams) * group_gap

for ci in range(n_cfg):
    xs = x_base + (ci - 1) * (bar_w + 0.03)
    vals = [success[t][ci] for t in teams]
    bars = ax.bar(xs, vals, width=bar_w, color=colors[ci], label=configs[ci], zorder=3,
                   edgecolor="white", linewidth=0.6)
    for x, v, t in zip(xs, vals, teams):
        r = ratio[t][ci]
        ax.text(x, v + 0.6, f"{v:.1f}%", ha="center", fontsize=8.2, fontweight="bold")
        ax.text(x, v - 3.2, f"{r:.2f}:1", ha="center", fontsize=7.3, color="white", fontweight="bold")

ax.set_xticks(x_base)
ax.set_xticklabels([f"{t}\n(real: {ratio[t][0]:.2f}:1)" for t in teams], fontsize=9.5)
ax.set_ylabel("Simulated success rate (2000-mission precise measurement)", fontsize=10.5)
ax.set_ylim(60, 82)
ax.set_title("What would make each real winner better? Battery energy margin helps almost\n"
             "everyone; going lighter to chase ratio only pays off for some designs",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9.5, frameon=False)
ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

fig.text(0.02, 0.02,
         "White numbers inside each bar = resulting payload-to-aircraft ratio. MTech is the standout: going lighter",
         fontsize=8, color="#555555")
fig.text(0.02, 0.002,
         "pushes it to 3.98:1 (near the DARPA 4:1 full-prize threshold) while INCREASING predicted reliability.",
         fontsize=8, color="#555555")

plt.tight_layout(rect=[0, 0.1, 1, 1])
out_path = r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\paper\figures\improvement_chart.png"
plt.savefig(out_path, dpi=200)
print(f"saved: {out_path}")
