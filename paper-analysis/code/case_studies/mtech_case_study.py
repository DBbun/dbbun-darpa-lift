"""
MTech (Team #937) case study: construct a fixed AircraftDesign matching MTech's
real-world specs and run it through the ORIGINAL, UNMODIFIED generator's mission
simulation logic (imported, not copied/edited) to see what the simulator itself
predicts for this configuration.

Does not touch darpa_lift_challenge_generator_v1_2.py — only imports it.
"""
import sys
import random
from collections import Counter

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
ENERGY_PROF = gen.ENERGY_SYSTEM_PROFILES["li_ion"]

EMPTY_MASS_KG = 14.51      # 32 lb
PAYLOAD_MASS_KG = 53.07    # 117 lb
ROTOR_COUNT = 6

battery_mass_kg = 4.0
battery_spec_energy = 220.0
battery_energy_Wh = battery_mass_kg * battery_spec_energy
battery_voltage_V = 48.0
battery_max_power_W = 6000.0

per_rotor_peak_W = battery_max_power_W / ROTOR_COUNT
per_rotor_peak_A = per_rotor_peak_W / battery_voltage_V
esc_current_rating_A = per_rotor_peak_A * 1.5

mtow_kg = EMPTY_MASS_KG + PAYLOAD_MASS_KG
ratio = PAYLOAD_MASS_KG / EMPTY_MASS_KG

mtech_design = gen.AircraftDesign(
    design_id="DLIFT_MTECH_CASE",
    animals=[],
    traits=[],
    animal_count=0,
    trait_count=0,

    empty_mass_kg=EMPTY_MASS_KG,
    payload_mass_kg=PAYLOAD_MASS_KG,

    rotor_count=ROTOR_COUNT,
    max_twr=1.3,
    burst_power_factor=1.0,
    burst_duration_s=0.0,
    unsteady_lift_gain=0.0,

    energy_system_type="li_ion",
    energy_system_description=ENERGY_PROF["description"],
    energy_density_class=ENERGY_PROF["energy_density_class"],
    power_class=ENERGY_PROF["power_class"],
    tech_maturity_class=ENERGY_PROF["tech_maturity_class"],
    energy_system_extra_failure_risk=ENERGY_PROF["extra_failure_risk"],

    battery_mass_kg=battery_mass_kg,
    battery_spec_energy_Wh_per_kg=battery_spec_energy,
    battery_energy_Wh=battery_energy_Wh,
    battery_nominal_voltage_V=battery_voltage_V,
    battery_max_power_W=battery_max_power_W,

    supercap_mass_kg=0.0,
    supercap_energy_Wh=0.0,
    supercap_max_power_W=0.0,

    motor_type="electric_brushless_outunner",
    motor_efficiency=0.88,
    esc_efficiency=0.97,
    esc_current_rating_A=esc_current_rating_A,

    structural_material="carbon_composite",
    structural_material_class="light_stiff",
    structural_extra_failure_risk=0.01,
    rotor_blade_material="carbon_composite",
    landing_gear_material="composite",

    frame_stiffness_longitudinal=0.70,
    tendon_cable_fraction=0.55,
    gust_rejection_gain=1.0,
    landing_gear_mass_kg=1.0,
    max_touchdown_velocity_mps=1.5,

    cruise_speed_mps=15.0,
    climb_rate_mps=3.5,
    mode_count=2,

    mtow_kg=mtow_kg,
    payload_to_aircraft_ratio=ratio,

    rule_empty_mass_ok=EMPTY_MASS_KG <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
    rule_payload_ok=PAYLOAD_MASS_KG >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],

    design_qualifying=True,
    design_qualifying_score=ratio,

    design_summary="MTech Team #937 case study (real-world reconstruction).",

    propulsion_architecture="pure_rotor_electric",
    primary_propulsor_type="multirotor",
    secondary_propulsor_type="none",
    secondary_propulsor_fraction=0.0,
    propulsion_hover_power_factor=1.0,
    propulsion_cruise_power_factor=1.0,
    propulsion_arch_extra_failure_risk=0.0,

    wing_foldable=False,
    wing_deploy_time_s=0.0,
    wing_deploy_failure_risk=0.0,

    image_prompt="",
)

print("=== MTech case-study design ===")
print(f"empty_mass_kg={EMPTY_MASS_KG}  payload_mass_kg={PAYLOAD_MASS_KG}  ratio={ratio:.3f}")
print(f"rule_empty_mass_ok={mtech_design.rule_empty_mass_ok}  rule_payload_ok={mtech_design.rule_payload_ok}")
print(f"battery_energy_Wh={battery_energy_Wh:.0f}  battery_max_power_W={battery_max_power_W:.0f}  esc_current_rating_A={esc_current_rating_A:.1f}")

N_MISSIONS = 3000
rng = random.Random(20260810)

results = []
for i in range(N_MISSIONS):
    env = gen.generate_environment(rng)
    mission, _ts = gen.simulate_mission(mtech_design, env, rng)
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
    print(f"  {reason}: {cnt} ({100*cnt/n:.1f}% of all missions)")

sat_seconds = [m.power_saturation_seconds for m in results]
thermal_peaks = [m.thermal_peak_C for m in results]
print(f"\npower_saturation_seconds: max={max(sat_seconds):.1f}  avg={sum(sat_seconds)/n:.2f}")
print(f"thermal_peak_C: max={max(thermal_peaks):.1f}  avg={sum(thermal_peaks)/n:.2f}")

full_prize = sum(1 for m in results if m.prize_tier == "full")
partial_prize = sum(1 for m in results if m.prize_tier == "partial")
print(f"\nprize_tier: full={full_prize} ({100*full_prize/n:.1f}%)  partial={partial_prize} ({100*partial_prize/n:.1f}%)  none={n-full_prize-partial_prize} ({100*(n-full_prize-partial_prize)/n:.1f}%)")
