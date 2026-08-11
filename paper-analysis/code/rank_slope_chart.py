import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 5 real winners, ranked 1 (best) to 5 (worst) by REAL placement
teams_by_real_rank = ["AVIDrone", "MTech", "Xtreme Aerial", "H-Squared", "MacGyver"]
real_ratio = {"AVIDrone": 3.85, "MTech": 3.66, "Xtreme Aerial": 3.44, "H-Squared": 2.96, "MacGyver": 2.49}
real_prize = {"AVIDrone": "1st \u2014 $1.25M", "MTech": "2nd \u2014 $750K", "Xtreme Aerial": "3rd \u2014 $500K",
              "H-Squared": "4th (no prize)", "MacGyver": "5th (no prize)"}

# same 5, ranked 1 (best) to 5 (worst) by SIMULATED mean rank_score (Monte Carlo v2)
sim_mean_rank_score = {"AVIDrone": 0.734, "MTech": 0.719, "Xtreme Aerial": 0.713, "H-Squared": 0.722, "MacGyver": 0.740}
teams_by_sim_rank = sorted(sim_mean_rank_score, key=lambda k: -sim_mean_rank_score[k])

real_pos = {name: i for i, name in enumerate(teams_by_real_rank)}   # 0 = best
sim_pos = {name: i for i, name in enumerate(teams_by_sim_rank)}     # 0 = best

colors = {"AVIDrone": "#1f5fa8", "MTech": "#2a7f3f", "Xtreme Aerial": "#b3841a",
          "H-Squared": "#7b4fa0", "MacGyver": "#b23a2f"}

fig, ax = plt.subplots(figsize=(7.5, 6.5))

LEFT_X, RIGHT_X = 0.0, 1.0
for name in teams_by_real_rank:
    y_left = -real_pos[name]
    y_right = -sim_pos[name]
    ax.plot([LEFT_X, RIGHT_X], [y_left, y_right], color=colors[name], linewidth=2.2,
            marker="o", markersize=9, zorder=3, alpha=0.9)

for name in teams_by_real_rank:
    y_left = -real_pos[name]
    ax.text(LEFT_X - 0.05, y_left, f"{name}\n({real_ratio[name]:.2f}:1, {real_prize[name]})",
            ha="right", va="center", fontsize=9.5, fontweight="bold", color=colors[name])

for name in teams_by_sim_rank:
    y_right = -sim_pos[name]
    ax.text(RIGHT_X + 0.05, y_right, f"{name}\n(score={sim_mean_rank_score[name]:.3f})",
            ha="left", va="center", fontsize=9.5, fontweight="bold", color=colors[name])

ax.text(LEFT_X, 0.55, "REAL\nplacement (1st \u2192 5th)", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.text(RIGHT_X, 0.55, "SIMULATED ranking\n(by mean predicted reliability)", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_xlim(-0.9, 1.9)
ax.set_ylim(-4.9, 1.9)
ax.axis("off")
ax.set_title("Does the simulator's predicted ranking match the real competition ranking?\n"
             "Straight-across lines = agreement. Crossing lines = disagreement.",
             fontsize=11.5, fontweight="bold", pad=16)

fig.text(0.02, 0.01,
         "Real placement is a fact (achieved ratio). Simulated ranking is the mean of 100 plausible-parameter "
         "Monte Carlo draws x 400 missions each per design (energy-capacity-corrected model). "
         "The near-total order scramble (Spearman \u03c1 = \u22120.30, n=5) shows the simulator, at current "
         "spec-uncertainty levels, cannot reproduce fine-grained real placement.",
         fontsize=7.8, color="#444444", wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out_path = r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\paper\figures\rank_slope_chart.png"
plt.savefig(out_path, dpi=200)
print(f"saved: {out_path}")
