"""
DefendTex case study: real team that crashed attempting an extreme ratio
(6.11:1 this attempt; other sources logged up to 9.63:1 on a different
attempt). Empty mass (8.35 kg) is BELOW the generator's entire
EMPTY_MASS_KG_RANGE floor (12.0 kg) -- the simulator could never
spontaneously generate a design this light. Built here as a direct
AircraftDesign construction (bypassing generate_design's random sampling
entirely), which is legal -- the CONFIG ranges only constrain the random
generator, not manual construction.

Tests the population-level finding from Part 3 (n=1000 dataset: the single
highest-ratio synthetic design, 8.07:1, had 0% predicted success) against
a real team that pushed even further and actually crashed.

Does not touch darpa_lift_challenge_generator_v1_2.py -- only imports it.
"""
import sys
import random
from collections import Counter

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
LI_ION = gen.ENERGY_SYSTEM_PROFILES["li_ion"]

EMPTY_MASS_KG = 8.347    # 18.4 lb -- below the generator's 12.0 kg sampling floor
PAYLOAD_MASS_KG = 50.98  # 112.4 lb
ROTOR_COUNT = 4          # uncertain from image, using schema floor as before

battery_mass_kg = 1.5
battery_spec_energy = 220.0
battery_energy_Wh = battery_mass_kg * battery_spec_energy
battery_voltage_V = 48.0
battery_max_power_W = 5000.0

per_rotor_peak_W = battery_max_power_W / ROTOR_COUNT
per_rotor_peak_A = per_rotor_peak_W / battery_voltage_V
esc_current_rating_A = per_rotor_peak_A * 1.5

mtow_kg = EMPTY_MASS_KG + PAYLOAD_MASS_KG
ratio = PAYLOAD_MASS_KG / EMPTY_MASS_KG

defendtex = gen.AircraftDesign(
    design_id="DLIFT_DEFENDTEX_CASE",
    animals=[], traits=[], animal_count=0, trait_count=0,

    empty_mass_kg=EMPTY_MASS_KG, payload_mass_kg=PAYLOAD_MASS_KG,

    rotor_count=ROTOR_COUNT, max_twr=1.2, burst_power_factor=1.0,
    burst_duration_s=0.0, unsteady_lift_gain=0.0,

    energy_system_type="li_ion",
    energy_system_description=LI_ION["description"],
    energy_density_class=LI_ION["energy_density_class"],
    power_class=LI_ION["power_class"],
    tech_maturity_class=LI_ION["tech_maturity_class"],
    energy_system_extra_failure_risk=LI_ION["extra_failure_risk"],

    battery_mass_kg=battery_mass_kg,
    battery_spec_energy_Wh_per_kg=battery_spec_energy,
    battery_energy_Wh=battery_energy_Wh,
    battery_nominal_voltage_V=battery_voltage_V,
    battery_max_power_W=battery_max_power_W,

    supercap_mass_kg=0.0, supercap_energy_Wh=0.0, supercap_max_power_W=0.0,

    motor_type="electric_brushless_outunner",
    motor_efficiency=0.85, esc_efficiency=0.96,
    esc_current_rating_A=esc_current_rating_A,

    structural_material="carbon_composite",
    structural_material_class="light_stiff",
    structural_extra_failure_risk=0.01,
    rotor_blade_material="carbon_composite",
    landing_gear_material="composite",

    frame_stiffness_longitudinal=0.45,   # string-braced, minimal rigid structure
    tendon_cable_fraction=0.65,           # defining feature of this design
    gust_rejection_gain=1.0,
    landing_gear_mass_kg=0.5,
    max_touchdown_velocity_mps=1.5,

    cruise_speed_mps=15.0, climb_rate_mps=3.5, mode_count=2,

    mtow_kg=mtow_kg, payload_to_aircraft_ratio=ratio,

    rule_empty_mass_ok=EMPTY_MASS_KG <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
    rule_payload_ok=PAYLOAD_MASS_KG >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],

    design_qualifying=True, design_qualifying_score=ratio,

    design_summary="DefendTex case study (real-world reconstruction).",

    propulsion_architecture="pure_rotor_electric",
    primary_propulsor_type="multirotor",
    secondary_propulsor_type="none", secondary_propulsor_fraction=0.0,
    propulsion_hover_power_factor=1.0, propulsion_cruise_power_factor=1.0,
    propulsion_arch_extra_failure_risk=0.0,

    wing_foldable=False, wing_deploy_time_s=0.0, wing_deploy_failure_risk=0.0,

    image_prompt="",
)

print("=== DefendTex case-study design ===")
print(f"empty_mass_kg={EMPTY_MASS_KG:.3f} (below sim's 12.0kg sampling floor)  payload_mass_kg={PAYLOAD_MASS_KG:.2f}  ratio={ratio:.3f}")
print(f"rule_payload_ok={defendtex.rule_payload_ok} (margin: {PAYLOAD_MASS_KG - CONFIG['DARPA_MIN_PAYLOAD_MASS_KG']:.2f} kg above the 49.9kg minimum)")

N_MISSIONS = 3000
rng = random.Random(20260810)

results = []
for _ in range(N_MISSIONS):
    env = gen.generate_environment(rng)
    mission, _ts = gen.simulate_mission(defendtex, env, rng)
    results.append(mission)

n = len(results)
success_rate = sum(1 for m in results if m.success) / n
qualifying_rate = sum(1 for m in results if m.is_qualifying_run) / n
rule_penalty_rate = sum(1 for m in results if m.rule_violation) / n
w_success = CONFIG["RANK_W_SUCCESS_RATE"]
w_qual = CONFIG["RANK_W_QUAL_RATE"]
w_pen = CONFIG["RANK_W_RULE_PENALTY"]
rank_score = success_rate * w_success + qualifying_rate * w_qual + (1.0 - rule_penalty_rate) * w_pen

print(f"\n=== Results over {n} simulated missions ===")
print(f"success_rate={success_rate:.3f}  qualifying_rate={qualifying_rate:.3f}  rule_penalty_rate={rule_penalty_rate:.3f}")
print(f"design_rank_score={rank_score:.3f}")

reasons = Counter(m.failure_reason for m in results if not m.success)
print(f"\nFailure reason breakdown ({sum(reasons.values())} failures):")
for reason, cnt in reasons.most_common():
    print(f"  {reason}: {cnt} ({100*cnt/n:.1f}%)")

sat_seconds = [m.power_saturation_seconds for m in results]
thermal_peaks = [m.thermal_peak_C for m in results]
print(f"\npower_saturation_seconds: max={max(sat_seconds):.1f}  avg={sum(sat_seconds)/n:.2f}")
print(f"thermal_peak_C: max={max(thermal_peaks):.1f}  avg={sum(thermal_peaks)/n:.2f}")
