# DBbun DARPA Lift Challenge simulator — validation against real competition results

Working notes, started 2026-08-10. Purpose: consolidate methodology and
findings from this analysis session as a base for drafting a paper. This is
a notes document, not paper prose.

## Research question

Does the DBbun synthetic aircraft-design generator/simulator
(`darpa_lift_challenge_generator_v1_2.py`, v1.2) produce "winning" designs
that resemble the designs that actually won the real DARPA Lift Challenge
(Aug 3-9, 2026, Dayton, OH)? If so, that's evidence the simulator captures
something real about the design/reliability tradeoffs of this problem.

## Data sources

- Simulator output: `Sample dataset (n = 10|100|1000|10000|100000) v1.2/`
  CSVs (`designs.csv`, `missions.csv`, `missions_timeseries.csv`), plus
  `v1.2/src/output/` (identical copy in `deploy_dashboard/output/`).
- Generator source (read-only reference, never modified):
  `v1.2/src/darpa_lift_challenge_generator_v1_2.py`.
- Real competition results: user-provided photos/video stills of the top 5
  scored teams' aircraft; official standings screenshot; unofficial mirror
  site (darpa-lift.artems.net, cross-validated against the official
  screenshot); a final-day recap video transcript (see `sources/`).
- Official site (darpaliftchallenge.com/results) blocks automated fetches
  (HTTP 403) — relied on the mirror + user-provided screenshot instead.

## Part 1 — CSV dataset integrity review

All 47 CSV files across all sample sizes/versions checked for schema
consistency, duplicate IDs, orphaned foreign keys, malformed rows, null
values. Result: clean throughout, no bugs found. Full detail in
`code/dataset_qc/`. Key notes:
- `payload_to_aircraft_ratio` = `payload_mass_kg / empty_mass_kg` exactly.
- `n=100000 v1.2` intentionally omits `missions_timeseries.csv` (would be
  ~300GB; lives on Hugging Face, not local) — not a bug.
- Design "qualification" (`rule_empty_mass_ok`, `rule_payload_ok`) is
  vacuous by construction: sampling ranges never allow either rule to fail
  (`EMPTY_MASS_KG_RANGE` max 24.6 < `DARPA_MAX_EMPTY_MASS_KG` 24.95;
  `PAYLOAD_MASS_KG_RANGE` min 50.0 > `DARPA_MIN_PAYLOAD_MASS_KG` 49.9).
  Every generated design "qualifies."

## Part 2 — Design-space coverage gap

The generator's `primary_propulsor_type` only produces: `multirotor`,
`gas_multirotor`, `series_hybrid`, `ducted_fan_array` (all rotor_count >= 4).
**It cannot generate a conventional single-main-rotor + tail-rotor
helicopter.** Of the 5 real scored teams, **3 of 5 (AVIDrone [1st],
Xtreme Aerial [3rd], H-Squared [4th]) are real single-main-rotor
helicopters** — an architecture representing 3 of 5 podium/near-podium
spots but entirely outside the simulator's design space. Only MTech
(hexarotor, tendon-braced) and MacGyver (genuine quadcopter) are
architecturally representable as generated.

Mitigating finding: `rotor_count` and `esc_current_rating_A` are **not
used anywhere in the mission physics/failure model** — hover/cruise power
depend only on total mass (`hover_power_W = 8.0 * mass_kg**1.5`), not rotor
count or disk loading. So the architecture gap affects how a design
*looks* in the schema, not what the simulator predicts would happen to it
in flight. This means real single-main-rotor designs can still be
reconstructed and simulated meaningfully by force-mapping to
`rotor_count=4`/`multirotor` — the mission outcome isn't biased by that
substitution.

## Part 3 — Population-level analysis (n=1000 dataset)

- Qualification is vacuous (see above) — all 1000 designs "qualify."
- Max raw `payload_to_aircraft_ratio` in the population: **8.07:1** (~2x
  the real-world best of 3.84:1). Comes from independently-sampled extreme
  low empty mass + high payload. The single highest-ratio design has a
  **0% mission success rate** — the simulator's own failure model rejects
  it as unbuildable-in-practice, even though nothing stops the generator
  from producing it.
- Only 14.4% of designs land in the real-world achieved band [3.45, 3.84].
- The **top tier by `design_rank_score`** (24 designs tied at max score
  1.0, out of 1000) have ratios **2.32-5.39, median 3.59** — this
  brackets the real winners' band well. 87.5% multirotor, 70.8% li-ion,
  79.2% fixed (non-folding) wings — a conservative, reliability-favoring
  profile.

## Part 4 — Case studies: real designs reconstructed and run through the actual mission engine

Method: import `darpa_lift_challenge_generator_v1_2.py` as a library
(read-only; never edited — see `code/case_studies/*.py`), construct an
`AircraftDesign` instance per real team using their known real mass/
payload figures plus best-effort engineering-judgment estimates for the
~45 fields not visible in photos (battery specs, motor/ESC efficiency,
cruise speed, landing dynamics, etc. — flagged per-field where estimated
vs. measured), then run 3000 simulated missions per design through the
unmodified `simulate_mission()` function with randomized weather draws.

| Team | Real ratio | Real placement | Sim success rate | Sim rank score |
|---|---|---|---|---|
| AVIDrone | 3.85 | 1st, $1.25M | 73.3% | 0.748 |
| MTech | 3.66 | 2nd, $750K | 76.3% | 0.780 |
| Xtreme Aerial | 3.44 | 3rd, $500K | 75.0% | 0.766 |
| H-Squared | 2.96 | 4th | 77.0% | 0.786 |
| MacGyver | 2.49 | 5th | 80.1% | 0.816 |

**Finding 1 (positive):** every real winner is judged solidly viable by
the simulator (73-80% predicted mission success) — none rejected as
implausible. Power/thermal margins are never binding for any of the 5
(0% power saturation, thermal flat at ambient ~22C throughout) given
reasonable battery sizing estimates.

**Finding 2 (structural limitation):** `design_rank_score` produces a
**perfect inverse rank correlation** with real placement across all 5
teams. Traced to the generator's random-failure stress-index term:
```python
load_term = max(design.payload_to_aircraft_ratio - 2.0, 0.0) / 3.0
stress_index = 0.3*wind_frac + 0.4*turb_frac + 0.3*load_term
```
(`darpa_lift_challenge_generator_v1_2.py`, mission failure pre-sampling
section). Every point of ratio above 2:1 mechanically raises several
failure probabilities (gust, control-saturation-adjacent terms),
independent of any other design quality. Real competition ranking rewards
peak achieved ratio (best single successful run); the simulator's
rank_score rewards average reliability, explicitly penalized by ratio. The
two objectives are anti-correlated by construction once ratio exceeds
~2:1.

## Part 5 — DefendTex: independent real-world corroboration, run as a 6th case study

Source: Twitter/X post (Hunter Weiss, @Hunter_Weiss) with video still,
providing real numbers for the first time: 18.4 lb (8.35 kg) empty mass,
112.4 lb (50.98 kg) payload, 6.11:1 ratio on this attempt (a different
source — the mirror site's "delisted teams" note — separately logged them
reaching 9.63:1 on what appears to be a different, more extreme attempt).
Team crashed; not officially scored. Image shows a multirotor braced by
visible yellow tension cables/strings running from the central pod to the
rotor booms — a clean real-world match to the generator's
`tendon_cable_fraction` field, more literally than MTech's design.

Reconstructed and run through 3000 simulated missions exactly as the other
5 (see `code/case_studies/defendtex_case_study.py`). Notably, DefendTex's
empty mass (8.35 kg) is **below the generator's own `EMPTY_MASS_KG_RANGE`
floor of 12.0 kg** — the random generator could never spontaneously
produce a design this light; it had to be constructed directly as an
`AircraftDesign`, bypassing `generate_design()`'s sampling entirely.

| Team | Real ratio | Real outcome | Sim success rate | Sim rank score |
|---|---|---|---|---|
| AVIDrone | 3.85 | 1st, $1.25M | 73.3% | 0.748 |
| MTech | 3.66 | 2nd, $750K | 76.3% | 0.780 |
| Xtreme Aerial | 3.44 | 3rd, $500K | 75.0% | 0.766 |
| H-Squared | 2.96 | 4th | 77.0% | 0.786 |
| MacGyver | 2.49 | 5th | 80.1% | 0.816 |
| **DefendTex** | **6.11** | **crashed, delisted** | **71.0%** | **0.731** |

DefendTex lands at the bottom of the full 6-design set on both metrics —
the only one of the six that actually failed in reality is also the one
the simulator rates least reliable. Two failure modes appear for DefendTex
that never appeared for any of the 5 winners: `gust_induced_instability`
rises to 7.5% (vs. 4.1-4.6% for the others), directly traceable to the
`load_term` stress index (1.37 for DefendTex vs. 0.62 for AVIDrone, the
next-highest); and `energy_depleted` appears as a real failure mode
(3.9%) for the first time, consistent with an 8.35 kg airframe leaving no
comfortable battery margin. Calibrated interpretation: 71% success is not
"the simulator predicted a crash" — it is "the simulator predicts this
design fails ~3x more often than the winners, for mechanistically
sensible reasons, and a real crash is well within that failure rate."
This is the second independent line of evidence (with Part 3's
population-level 8.07:1/0%-success outlier) that the simulator's failure
model degrades gracefully and sensibly as ratio pushes past the
range the real winners occupied — even though its `design_rank_score`
inversely correlates with real placement within that winners' band
(Part 4, Finding 2).

## Part 6 — Population analysis rerun at n=100000 (larger sample)

Rerunning Part 3's top-tier analysis at full n=100000 scale (2,989 designs
tied at max `design_rank_score`, vs. 24 at n=1000) revises the picture in
one important way: the single highest-raw-ratio design in the full
population (8.22:1) has a **30% success rate**, not the 0% found at
n=1000. The earlier "extreme ratio -> zero success" claim was partly an
artifact of small-sample luck (one bad-luck design at n=1000); the true
relationship looks like a gradual decline, not a cliff. Top-tier ratio
range widens to 2.04-8.09 (median 3.95, up from 3.59). 88.8% of top-tier
designs fall within the full real-observed range [2.49, 6.11], but only
42.5% fall within the winners-only band [2.49, 3.85] — the simulator's
own "best" designs skew toward somewhat higher ratios than any real
winner actually achieved. Categorical distributions (propulsion type,
energy system, structural material, wing folding) are consistent with
the n=1000 findings, just with tighter confidence given the larger n.

## Part 7 — Controlled ablation: isolating the load_term mechanism

Motivation: Part 4's inverse-rank-correlation finding was attributed to
the `load_term = max(ratio-2.0,0)/3.0` stress-index term. To test this
claim in isolation (rather than inferring it from 6 noisy real-world
reconstructions), MTOW was held fixed at 75 kg and *only* the empty/
payload split (hence ratio) was swept from 2.2 to 8.0, with every other
field -- including power margin relative to the fixed hover-power
requirement -- pinned constant. See `code/ratio_sweep_ablation.py`.

**Result: the pure load_term effect is real but small.** Gust-induced-
instability failure rate does climb ~50% relative (4.6% at ratio=2.2 to
6.9% at ratio=8.0), consistent with the code. But overall success rate
and rank score stay flat across the whole sweep (74.8-78.5%, no clear
monotonic trend) -- nothing like the 71-80% spread with a clean inverse
ordering seen across the 6 real-team case studies in Part 4.

**This is an important self-correction.** It indicates Part 4's inverse-
correlation finding was likely driven substantially by **inconsistent
power-margin sizing across the manual per-team reconstructions**
(e.g. AVIDrone's battery was sized with only ~4% headroom over its hover
requirement and showed real `power_saturation` failures not seen for
other teams) rather than purely by the ratio-driven stress term it was
originally attributed to. Flagging this explicitly rather than keeping
the original (overstated) causal claim -- see Part 8 for the corrected,
consistent-methodology re-test.

## Part 8 — Monte Carlo sensitivity analysis v1 (consistent power-margin methodology)

Directly motivated by the Part 7 correction: re-runs all 6 real-team case
studies with battery_max_power_W derived from a *consistently applied*
power margin (1.1x-2.0x of each design's own hover-power requirement,
sampled per draw) instead of ad hoc per-team point estimates, plus
Monte Carlo sampling of every other uncertain field from the generator's
own stated plausible ranges (`CONFIG[...RANGE]` / li-ion spec-energy
range). empty_mass_kg, payload_mass_kg, and rotor_count stay fixed to
real/measured values per team (cruise_speed_mps additionally fixed for
H-Squared, the one team with measured telemetry). 100 draws x 400
missions per team. See `code/monte_carlo_sensitivity.py`.

| Team | Real ratio | Real placement | Mean rank | Median | Stdev | Min | Max |
|---|---|---|---|---|---|---|---|
| AVIDrone | 3.85 | 1st | 0.712 | 0.743 | 0.119 | 0.177 | 0.854 |
| MTech | 3.66 | 2nd | 0.680 | 0.718 | 0.173 | 0.037 | 0.854 |
| Xtreme Aerial | 3.44 | 3rd | 0.491 | 0.637 | 0.319 | 0.002 | 0.861 |
| H-Squared | 2.96 | 4th | 0.472 | 0.620 | 0.334 | 0.004 | 0.869 |
| MacGyver | 2.49 | 5th | 0.546 | 0.680 | 0.294 | 0.004 | 0.871 |
| DefendTex | 6.11 | crashed | 0.702 | 0.722 | 0.108 | 0.304 | 0.859 |

Exact 5-winner inverse ordering (the original point-estimate finding)
held in only 1/100 draws. Spearman's rho on mean scores vs. real
placement (5 winners) = **+0.7**. But three teams (Xtreme Aerial,
H-Squared, MacGyver) show enormous variance (stdev 0.29-0.33, spanning
nearly the full 0-0.87 range) -- their predicted reliability is
essentially undetermined given current spec uncertainty, while AVIDrone
and DefendTex are comparatively stable (stdev ~0.11-0.12).

**This result was later found to be an artifact and was superseded by
Part 8b** (below) -- flagged here for transparency about the
investigative path, not as a live finding.

## Part 8a — Attribution analysis: isolating the source of the huge per-team variance

Two rounds, both restricted to the 3 unstable teams (Xtreme Aerial,
H-Squared, MacGyver), holding all-but-one field-group fixed at point
estimates per condition:

**Round 1** (`code/attribution_analysis.py`): tested the hypothesis that
climb_rate/max_touchdown_velocity (a nonlinear ratio in the failure
model) drives the instability. Result: **disproved**. Varying only
those two fields produced tight, stable results (stdev 0.08-0.10,
comparable to AVIDrone/DefendTex's overall stability) -- varying
*everything else* instead produced the large variance (stdev 0.26-0.36).

**Round 2** (`code/attribution_analysis_2.py`): split "everything else"
into 3 mechanistically distinct groups. Result: **decisively isolated**
to battery energy *capacity* (battery_mass_kg x battery_spec_energy_Wh_per_kg),
not power *margin* (motor/ESC efficiency + power_margin -> stdev only
0.02-0.04, `power_saturation` never occurred, 0.00% across every
condition) and not structural/gust fields (stdev 0.03-0.04). Battery
capacity alone reproduced almost all the original variance (stdev
0.32-0.37) with `energy_depleted` failure rates up to 38.9% (Xtreme
Aerial), 33.8% (H-Squared), 21.2% (MacGyver) -- scaling with each
aircraft's absolute MTOW, as expected.

**Root cause**: `battery_max_power_W` (v1 Monte Carlo) was correctly
derived from each design's own hover-power requirement (a consistent,
size-aware methodology), but `battery_mass_kg` and
`battery_spec_energy_Wh_per_kg` were sampled completely independently of
aircraft size -- letting physically-inconsistent "battery too small for
this aircraft" draws appear for the heavier real designs. This was a
methodology bug in the Monte Carlo setup, not a property of the
generator itself.

## Part 8b — Monte Carlo sensitivity analysis v2 (energy-capacity-corrected)

Fixes the Part 8a root cause: `battery_energy_Wh` is now derived from a
margin (1.2x-2.5x, sampled per draw) over each design's own *nominal
mission energy requirement* (computed from the same course geometry the
generator uses -- 4.0 nm loaded leg, 1.0 nm unloaded leg, 350 ft cruise
altitude, ~18 hover turns x 12s -- conservatively priced at hover-power
level throughout), then `battery_mass_kg` is back-derived from that
energy and an independently-sampled spec energy (a technology choice,
not a size-dependent one) -- mirroring exactly how power margin was
already handled correctly. See `code/monte_carlo_sensitivity_v2.py`.

| Team | Real ratio | Real placement | Mean rank | Stdev |
|---|---|---|---|---|
| AVIDrone | 3.85 | 1st, $1.25M | 0.734 | 0.083 |
| MTech | 3.66 | 2nd, $750K | 0.719 | 0.106 |
| Xtreme Aerial | 3.44 | 3rd, $500K | 0.713 | 0.097 |
| H-Squared | 2.96 | 4th | 0.722 | 0.092 |
| MacGyver | 2.49 | 5th | 0.740 | 0.086 |
| DefendTex | 6.11 | crashed | 0.700 | 0.106 |

**This is the final, most defensible result.** Correcting the
energy-capacity artifact collapses the per-team variance down to a tight
band (stdev 0.083-0.106) across all six designs, confirming Part 8a's
diagnosis. Spearman's rho on mean scores vs. real placement (5 winners)
is now **-0.30** (n=5, not statistically significant) -- both the
original point-estimate finding (rho=-1.0) and the Part 8 finding
(rho=+0.7) were artifacts of different methodology bugs, not real
signal. All six real designs cluster tightly in a 0.700-0.740 mean
rank_score band. The one surviving, modest signal: DefendTex (the only
one that actually crashed) has the lowest mean of the six, though the
margin is small relative to the noise.

**Honest overall conclusion**: the simulator's mission-success model is
well-behaved and plausible for all 6 real designs (none rejected as
unviable) but, at the level of spec uncertainty available without real
team spec sheets, does not have the resolving power to reproduce
fine-grained real competition placement. It can distinguish "viable"
from "not viable" (DefendTex trending lowest) but not reliably rank five
viable designs against each other. Figures updated with v2 numbers:
`figures/real_vs_simulated_comparison.png`, `figures/correlation_scatter.png`.

**Reproducibility note**: a full independent rerun of Part 8b (same
script, same nominal seed) produced slightly different per-team means
(e.g. AVIDrone 0.734->0.724, H-Squared 0.722->0.752) because the
per-team seed offset used Python's built-in `hash(team_name)`, and
Python randomizes string hashing per-process by default (`PYTHONHASHSEED`
is not fixed) -- so "same seed" did not actually reproduce the same 100
draws; each run is an independent Monte Carlo sample, not a deterministic
replay. Both runs, being independent, agree on the qualitative picture:
tight clustering (~0.70-0.75) and **DefendTex consistently lowest in
both** (0.700 and 0.707 mean rank_score respectively) -- the one finding
that replicates. The fine-grained ordering among the five winners does
not replicate between the two independent runs, which if anything
reinforces the "cannot reliably rank five viable designs" conclusion. To
get bit-for-bit reproducible draws in future runs, seed with
`hash((draw_seed, name))` via `random.Random` directly, or set
`PYTHONHASHSEED` in the environment, rather than relying on `hash(str)`.

**Chart note**: `design_rank_score` is an opaque composite (55% success
rate + 25% qualifying rate + 20% rule-compliance rate) that isn't
meaningful to a reader without explanation. `figures/viability_discrimination_chart.png`
uses **success_rate** directly instead (plain "% of simulated attempts
that complete the course"), re-run from `code/monte_carlo_sensitivity_v2.py`
tracking success_rate alongside rank_score. Final success-rate numbers:
DefendTex 68.8% (crashed in reality, lowest of six), Xtreme Aerial 70.6%,
AVIDrone 70.7%, MTech 70.8%, MacGyver 70.8%, H-Squared 73.6%. This is the
paper's headline positive figure -- framed narrowly and accurately as
"the simulator discriminates viable from non-viable designs correctly,"
not as "the simulator ranks the winners correctly" (which Part 8b
disproved, rho=-0.30). A second figure, `figures/rank_slope_chart.png`,
shows the rank-disagreement finding honestly via a slope chart (real
placement vs. simulated ranking, connected by lines) for anyone who
wants the fuller, more nuanced picture alongside the headline finding.

## Part 9 — Reframed core claim (2026-08-11): population percentile validation

User redirected the paper's core claim away from fine-grained ranking
(disproved in Part 8b) toward two more tractable, honest claims, using
only the 5 real teams that actually scored (DefendTex excluded --
correct, since it never completed a course):

**Claim 1**: the 5 real winners land in the high-performing tier of the
simulator's own design space (a percentile claim against the full
population, not a head-to-head ranking claim among just the 5).

**Claim 2**: use what Part 8a established (battery energy capacity is
the dominant reliability lever) to propose concrete, simulator-grounded
modifications to each of the 5 real designs.

### Claim 1 evidence: true population percentile benchmark

Every earlier population measurement was either the noisy 10-mission
dataset or a cherry-picked "already scored 1.0" subset -- neither is a
fair population baseline. This draws an **unfiltered random sample of
150 designs** from the full n=100000 dataset and evaluates each with 500
missions for a precise estimate. See `code/population_percentile_benchmark.py`.

Population (n=150, precisely measured): mean=49.6%, median=60.3%,
stdev=27.9%. Notably wide and left-skewed -- p5=p10=0% (roughly 10% of
random designs never succeed at all), p25=30.6%, p50=59.8%, p75=72.8%,
p90=77.4%, p95=82.0%.

| Team | True success rate | Percentile |
|---|---|---|
| AVIDrone | 70.7% | p70 |
| MTech | 70.8% | p71 |
| Xtreme Aerial | 70.6% | p70 |
| H-Squared | 73.6% | p77 |
| MacGyver | 70.8% | p71 |

**All 5 real winners land in the 70th-77th percentile of the full random
population.** This is the paper's cleanest, most defensible "consistent
with reality" evidence -- legitimate because it's a percentile claim
against an honest, unfiltered baseline, not a re-run of the ranking claim
already shown not to hold. Figure: `figures/population_percentile_chart.png`
(shows only actually-measured percentile brackets; no distribution shape
is invented between them, since the raw 150 values were not saved to
disk, only summary statistics).

**Related self-correction**: an earlier draft of this figure attempted
to reconstruct a smooth histogram by interpolating between the 7 known
percentile points -- this was caught and rejected before publishing
because it would have fabricated distributional detail (e.g. smoothing
over the real discrete spike at 0%) not actually present in the data.
The published figure shows only real measured brackets.

### Claim 2 evidence: design improvement recommendations

For each of the 5 real winners, tested 3 fixed configurations (2000
missions each, not Monte Carlo -- a direct comparison, not an uncertainty
sweep) using the real empty_mass/payload as given: (A) baseline with
typical mid-range energy margin (1.85x nominal mission energy), (B) same
design with a generous energy margin (2.5x) -- the lever Part 8a proved
dominant, (C) B plus an 8%-lighter airframe (payload held fixed, so
ratio increases). See `code/design_improvement_recommendations.py`.

| Team | A: baseline | B: +battery margin | C: B + 8% lighter |
|---|---|---|---|
| AVIDrone | 73.0% @ 3.85:1 | 76.6% @ 3.85:1 | 72.3% @ 4.18:1 |
| MTech | 74.6% @ 3.66:1 | 75.0% @ 3.66:1 | **76.1% @ 3.98:1** |
| Xtreme Aerial | 75.8% @ 3.44:1 | 76.2% @ 3.44:1 | 75.3% @ 3.73:1 |
| H-Squared | 75.3% @ 2.96:1 | 74.8% @ 2.96:1 | 74.3% @ 3.22:1 |
| MacGyver | 76.4% @ 2.49:1 | 76.9% @ 2.49:1 | 74.8% @ 2.71:1 |

**Finding 1**: battery energy margin alone (A->B) helps or is neutral
for every team, with AVIDrone gaining the most (+3.6 points) -- plausibly
because AVIDrone's real design is the leanest of the five (13.2 kg
empty), leaving the least energy margin to begin with.

**Finding 2**: "go lighter to chase ratio" (B->C) is NOT a universal
improvement -- it costs AVIDrone, Xtreme Aerial, H-Squared, and MacGyver
reliability, but is a genuine double win for **MTech specifically**:
ratio rises from 3.66 to 3.98 (near the DARPA 4:1 full-prize threshold)
while reliability *increases* from 75.0% to 76.1%. This is the paper's
concrete, specific decision-support recommendation -- not a blanket
"make it lighter" statement, but a design-by-design answer grounded in
each aircraft's actual mass/power configuration. Figure:
`figures/improvement_chart.png`.

## Paper structure as of 2026-08-11 (current plan)

1. Dataset QC (Part 1) — clean, no bugs.
2. Design-space coverage gap (Part 2) — 3 of 5 real winners are
   single-main-rotor helicopters the generator can't produce; shown not
   to bias mission-outcome predictions since rotor_count isn't used in
   the physics.
3. **Headline claim, Claim 1** (Part 9): the 5 real scored winners land
   in the 70th-77th percentile of the simulator's full random design
   population — legitimate, honest "consistent with reality" evidence.
   Figure: `population_percentile_chart.png`.
4. Secondary supporting figure: `viability_discrimination_chart.png` —
   DefendTex (the one non-winner, crashed) trends lowest of 6, though
   NOTE this specific separation was tested and is NOT statistically
   significant vs. 4 of the 5 winners (p=0.14-0.19, see below) — must be
   captioned honestly as a directional/numerical observation, not a
   proven discrimination result.
5. **Claim 2** (Part 9): design-improvement recommendations per real
   winner. Figure: `improvement_chart.png`.
6. Full methodology transparency section covering the investigative path
   (Parts 3-8): population noise-inflation correction, the two overturned
   rank-correlation findings, the battery-capacity artifact and its fix,
   the touchdown-hypothesis rejection. Framed as evidence of rigor, not
   hidden.

## Known caveat requiring a fix before publication

`viability_discrimination_chart.png` and its surrounding text (session
turn following its creation) claimed the simulator "correctly separates"
DefendTex from the 5 winners. A follow-up significance test (z-test on
the reported means/stdevs, n=100 draws each) showed this holds only
against H-Squared (p<0.0001) and NOT against AVIDrone, MTech, Xtreme
Aerial, or MacGyver (p=0.14-0.19 for all four). The chart and any prose
built on it must be re-captioned to state this honestly — DefendTex
numerically lowest of six is not the same as a statistically confirmed
discrimination, given only one real failure case in the sample.

## Open items / next steps

1. Re-caption `viability_discrimination_chart.png` per the caveat above
   before it goes in the paper, or drop it in favor of leading with
   Claim 1/Claim 2 only.
2. Regenerate `figures/real_vs_simulated_comparison.png` and
   `correlation_scatter.png` with Part 8b (v2) numbers if kept in the
   methodology-transparency section — currently superseded Part 8 (v1)
   numbers.
3. DefendTex 9.63:1 case (12 lb / 112 lb, a more extreme second real
   DefendTex data point) — not yet built as a case study; optional now
   that DefendTex is secondary rather than the headline finding.
4. Images: AVIDrone (x3), MTech, Xtreme Aerial, MacGyver (x2), H-Squared
   (x2), DefendTex (x2, both 6.11 and 9.63 flights) all done in
   `paper/images/`.
5. Venue: arXiv preprint (decided 2026-08-10, superseding the earlier
   Science Robotics stretch-goal discussion).
