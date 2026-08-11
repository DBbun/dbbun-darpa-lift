"""
Noise-inflation test: the n=100000 v1.2 dataset has 2,989 designs with a
"perfect" design_rank_score of 1.0 -- but that score comes from only 10
simulated missions per design (MISSIONS_PER_DESIGN=10 in the original
generation run). With only 10 trials, many mediocre-to-good designs can
post a perfect record by pure chance.

This script samples 40 of those "perfect" designs, reconstructs each one
EXACTLY from its CSV row (including animals/traits, which affect several
failure-probability terms), and re-evaluates each with 2500 missions
instead of 10 -- the same scale used for the 6 real-team case studies.
If the noise-inflation hypothesis is right, most should regress well
below 1.0 once measured precisely.

Does not touch darpa_lift_challenge_generator_v1_2.py -- only imports it,
and only reads designs.csv (never writes to the dataset).
"""
import sys
import csv
import random
import statistics

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
DESIGNS_PATH = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 100000) v1.2\designs.csv"

N_SAMPLE = 40
N_MISSIONS = 2500


def row_to_design(row):
    b = lambda v: v == "True"
    f = float
    i = int
    return gen.AircraftDesign(
        design_id=row["design_id"],
        animals=row["animals"].split(",") if row["animals"] else [],
        traits=row["traits"].split(",") if row["traits"] else [],
        animal_count=i(row["animal_count"]), trait_count=i(row["trait_count"]),
        empty_mass_kg=f(row["empty_mass_kg"]), payload_mass_kg=f(row["payload_mass_kg"]),
        rotor_count=i(row["rotor_count"]), max_twr=f(row["max_twr"]),
        burst_power_factor=f(row["burst_power_factor"]), burst_duration_s=f(row["burst_duration_s"]),
        unsteady_lift_gain=f(row["unsteady_lift_gain"]),
        energy_system_type=row["energy_system_type"],
        energy_system_description=row["energy_system_description"],
        energy_density_class=row["energy_density_class"], power_class=row["power_class"],
        tech_maturity_class=row["tech_maturity_class"],
        energy_system_extra_failure_risk=f(row["energy_system_extra_failure_risk"]),
        battery_mass_kg=f(row["battery_mass_kg"]),
        battery_spec_energy_Wh_per_kg=f(row["battery_spec_energy_Wh_per_kg"]),
        battery_energy_Wh=f(row["battery_energy_Wh"]),
        battery_nominal_voltage_V=f(row["battery_nominal_voltage_V"]),
        battery_max_power_W=f(row["battery_max_power_W"]),
        supercap_mass_kg=f(row["supercap_mass_kg"]), supercap_energy_Wh=f(row["supercap_energy_Wh"]),
        supercap_max_power_W=f(row["supercap_max_power_W"]),
        motor_type=row["motor_type"], motor_efficiency=f(row["motor_efficiency"]),
        esc_efficiency=f(row["esc_efficiency"]), esc_current_rating_A=f(row["esc_current_rating_A"]),
        structural_material=row["structural_material"],
        structural_material_class=row["structural_material_class"],
        structural_extra_failure_risk=f(row["structural_extra_failure_risk"]),
        rotor_blade_material=row["rotor_blade_material"], landing_gear_material=row["landing_gear_material"],
        frame_stiffness_longitudinal=f(row["frame_stiffness_longitudinal"]),
        tendon_cable_fraction=f(row["tendon_cable_fraction"]), gust_rejection_gain=f(row["gust_rejection_gain"]),
        landing_gear_mass_kg=f(row["landing_gear_mass_kg"]),
        max_touchdown_velocity_mps=f(row["max_touchdown_velocity_mps"]),
        cruise_speed_mps=f(row["cruise_speed_mps"]), climb_rate_mps=f(row["climb_rate_mps"]),
        mode_count=i(row["mode_count"]),
        mtow_kg=f(row["mtow_kg"]), payload_to_aircraft_ratio=f(row["payload_to_aircraft_ratio"]),
        rule_empty_mass_ok=b(row["rule_empty_mass_ok"]), rule_payload_ok=b(row["rule_payload_ok"]),
        design_qualifying=b(row["design_qualifying"]), design_qualifying_score=f(row["design_qualifying_score"]),
        design_summary=row["design_summary"],
        propulsion_architecture=row["propulsion_architecture"],
        primary_propulsor_type=row["primary_propulsor_type"],
        secondary_propulsor_type=row["secondary_propulsor_type"],
        secondary_propulsor_fraction=f(row["secondary_propulsor_fraction"]),
        propulsion_hover_power_factor=f(row["propulsion_hover_power_factor"]),
        propulsion_cruise_power_factor=f(row["propulsion_cruise_power_factor"]),
        propulsion_arch_extra_failure_risk=f(row["propulsion_arch_extra_failure_risk"]),
        wing_foldable=b(row["wing_foldable"]), wing_deploy_time_s=f(row["wing_deploy_time_s"]),
        wing_deploy_failure_risk=f(row["wing_deploy_failure_risk"]),
        image_prompt=row["image_prompt"],
        design_success_rate=f(row["design_success_rate"]),
        design_qualifying_rate=f(row["design_qualifying_rate"]),
        design_rule_penalty_rate=f(row["design_rule_penalty_rate"]),
        design_rank_score=f(row["design_rank_score"]),
        design_stars=i(row["design_stars"]),
    )


print("Loading designs.csv and filtering to design_rank_score == 1.0 ...")
with open(DESIGNS_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    perfect_rows = [row for row in reader if row["design_rank_score"] == "1.0"]
print(f"Found {len(perfect_rows)} designs with a 'perfect' (10-mission) rank_score of 1.0")

sample_rng = random.Random(20260810)
sample_rows = sample_rng.sample(perfect_rows, N_SAMPLE)

print(f"\nRe-evaluating {N_SAMPLE} of them with {N_MISSIONS} missions each (vs. the original 10) ...\n")

true_rank_scores = []
true_success_rates = []
w_success = CONFIG["RANK_W_SUCCESS_RATE"]
w_qual = CONFIG["RANK_W_QUAL_RATE"]
w_pen = CONFIG["RANK_W_RULE_PENALTY"]

for idx, row in enumerate(sample_rows):
    design = row_to_design(row)
    rng = random.Random(sample_rng.randint(0, 2**31 - 1))
    results = []
    for _ in range(N_MISSIONS):
        env = gen.generate_environment(rng)
        mission, _ts = gen.simulate_mission(design, env, rng)
        results.append(mission)
    n = len(results)
    success_rate = sum(1 for m in results if m.success) / n
    qualifying_rate = sum(1 for m in results if m.is_qualifying_run) / n
    rule_penalty_rate = sum(1 for m in results if m.rule_violation) / n
    rank_score = success_rate * w_success + qualifying_rate * w_qual + (1.0 - rule_penalty_rate) * w_pen
    true_rank_scores.append(rank_score)
    true_success_rates.append(success_rate)
    print(f"  [{idx+1}/{N_SAMPLE}] {design.design_id}  ratio={design.payload_to_aircraft_ratio:.2f}  "
          f"original(n=10)=1.000  true(n={N_MISSIONS})_success={success_rate:.3f}  true_rank={rank_score:.3f}")

print(f"\n=== Summary: true rank_score of designs that scored a 'perfect' 1.0 on only 10 missions ===")
print(f"mean={statistics.mean(true_rank_scores):.3f}  median={statistics.median(true_rank_scores):.3f}  "
      f"stdev={statistics.stdev(true_rank_scores):.3f}  min={min(true_rank_scores):.3f}  max={max(true_rank_scores):.3f}")

still_near_perfect = sum(1 for s in true_rank_scores if s >= 0.95)
print(f"\nStill >= 0.95 true rank_score under precise re-measurement: {still_near_perfect}/{N_SAMPLE} ({100*still_near_perfect/N_SAMPLE:.1f}%)")
below_08 = sum(1 for s in true_rank_scores if s < 0.80)
print(f"Regressed below 0.80: {below_08}/{N_SAMPLE} ({100*below_08/N_SAMPLE:.1f}%)")
