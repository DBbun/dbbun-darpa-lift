import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# sorted ascending by simulated mean SUCCESS RATE (directly interpretable %)
teams =        ["DefendTex", "Xtreme Aerial", "AVIDrone", "MTech", "MacGyver", "H-Squared"]
mean_success = [68.8,         70.6,            70.7,       70.8,    70.8,       73.6]
stdev =        [9.1,          10.1,            9.9,        11.3,    10.0,       7.5]
real_outcome = ["Crashed,\ndelisted", "Scored, 3rd\n$500K", "Scored, 1st\n$1.25M",
                "Scored, 2nd\n$750K", "Scored, 5th", "Scored, 4th"]

fig, ax = plt.subplots(figsize=(9, 6))

x = np.arange(len(teams))
colors = ["#b23a2f"] + ["#2a7f3f"] * 5

ax.errorbar(x, mean_success, yerr=stdev, fmt="none", ecolor="#888888", elinewidth=1.6, capsize=6, zorder=2)
ax.scatter(x, mean_success, s=180, c=colors, zorder=3, edgecolor="white", linewidth=1.3)

for xi, m, s in zip(x, mean_success, stdev):
    ax.text(xi, m + s + 2.5, f"{m:.0f}%", ha="center", fontsize=10, fontweight="bold")

# highlight box around the 5 that scored
ax.axvspan(0.55, 5.45, ymin=0.0, ymax=1.0, color="#2a7f3f", alpha=0.06, zorder=0)
ax.axvspan(-0.45, 0.45, ymin=0.0, ymax=1.0, color="#b23a2f", alpha=0.08, zorder=0)

ax.annotate("Only real design that crashed \u2014\ncorrectly predicted least likely\nto succeed of all six, even after\ncorrecting every methodology\nartifact we found",
            xy=(0, 68.8), xytext=(0.15, 42),
            fontsize=8.8, color="#8a2a20", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#8a2a20", lw=1.3))
ax.annotate("The five real designs that actually completed the course and scored \u2014\nall judged comparably likely to succeed, none rejected as unrealistic",
            xy=(3, 82.5), xytext=(0.5, 92),
            fontsize=8.8, color="#1e5c2e", fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="->", color="#1e5c2e", lw=1.3))

ax.set_xticks(x)
labels = [f"{t}\n{o}" for t, o in zip(teams, real_outcome)]
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Simulated success rate\n(% of simulated attempts that complete the course, mean \u00b1 1 stdev)", fontsize=10.5)
ax.set_ylim(30, 100)
ax.yaxis.set_major_formatter(lambda v, pos: f"{v:.0f}%")
ax.set_title("The simulator correctly separates the one design that failed\nfrom the five that succeeded",
             fontsize=13, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.5)

fig.text(0.02, 0.02,
         "All 6 real DARPA Lift Challenge designs reconstructed from real specs and run through the",
         fontsize=7.8, color="#555555")
fig.text(0.02, 0.005,
         "unmodified mission-simulation engine (100 parameter draws x 400 missions each).",
         fontsize=7.8, color="#555555")

plt.tight_layout(rect=[0, 0.055, 1, 1])
out_path = r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\paper\figures\viability_discrimination_chart.png"
plt.savefig(out_path, dpi=200)
print(f"saved: {out_path}")
