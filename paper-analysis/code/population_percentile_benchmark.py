"""
Population percentile benchmark: where does a ~70-74% true success rate
(what the 5 real DARPA Lift Challenge winners scored, precisely measured)
actually fall within the full population's TRUE success-rate distribution?

Everything measured about the n=100000 population so far was either the
noisy 10-mission-per-design dataset, or a cherry-picked "already scored
1.0" subset. This draws an UNFILTERED random sample of 150 designs from
across the whole population and evaluates each with 500 missions (enough
to be a precise, low-noise estimate, unlike the original 10-mission
figures) to build a real distribution to benchmark the winners against.

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

N_SAMPLE = 150
N_MISSIONS = 500


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


print("Loading designs.csv ...")
with open(DESIGNS_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
print(f"Loaded {len(all_rows)} designs. Drawing an UNFILTERED random sample of {N_SAMPLE}.")

sample_rng = random.Random(20260811)
sample_rows = sample_rng.sample(all_rows, N_SAMPLE)

true_success_rates = []
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
    true_success_rates.append(success_rate)
    if (idx + 1) % 25 == 0:
        print(f"  [{idx+1}/{N_SAMPLE}] done")

true_success_rates.sort()
print(f"\n=== Full-population true success rate distribution (n={N_SAMPLE} random designs, {N_MISSIONS} missions each) ===")
print(f"mean={statistics.mean(true_success_rates):.3f}  median={statistics.median(true_success_rates):.3f}  stdev={statistics.stdev(true_success_rates):.3f}")
print(f"min={true_success_rates[0]:.3f}  max={true_success_rates[-1]:.3f}")

percentiles = [5, 10, 25, 50, 75, 90, 95]
for p in percentiles:
    idx = int(round(p / 100 * (len(true_success_rates) - 1)))
    print(f"  p{p}: {true_success_rates[idx]:.3f}")

# where do the 5 real winners' success rates (0.706-0.736, from Monte Carlo v2) fall?
winner_rates = {"AVIDrone": 0.707, "MTech": 0.708, "Xtreme Aerial": 0.706, "H-Squared": 0.736, "MacGyver": 0.708}
print(f"\n=== Percentile rank of the 5 real winners within this population distribution ===")
for name, rate in winner_rates.items():
    rank = sum(1 for v in true_success_rates if v <= rate)
    pct = 100 * rank / len(true_success_rates)
    print(f"  {name}: success_rate={rate:.3f} -> percentile {pct:.1f} (better than {rank}/{len(true_success_rates)} random population designs)")
