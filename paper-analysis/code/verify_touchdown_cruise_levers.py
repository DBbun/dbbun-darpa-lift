"""
Controlled verification of the two new levers found via the population
comparison (touchdown/climb ratio, cruise speed), for MTech and MacGyver
specifically. Battery energy margin held fixed at the already-validated
good value (2.5x nominal mission energy) throughout, so these tests
isolate the NEW levers' independent and combined contribution on top of
that established improvement, not a re-confirmation of it.

4 conditions per design, 2500 missions each (precise, not Monte Carlo):
  A) baseline: population "rest" tier averages (climb=4.05, touchdown=1.70, cruise=14.6)
  B) + touchdown/climb improvement only (climb=3.75, touchdown=2.09)
  C) + cruise speed improvement only (cruise=18.1)
  D) + both improvements together

Does not touch darpa_lift_challenge_generator_v1_2.py -- only imports it.
"""
import sys
import random

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
LI_ION = gen.ENERGY_SYSTEM_PROFILES["li_ion"]

DIST_LOADED_M = CONFIG["PAYLOAD_LEG_DISTANCE_NM"] * 1852.0
DIST_UNLOADED_M = CONFIG["RETURN_LEG_DISTANCE_NM"] * 1852.0
ALT_TARGET_M = CONFIG["CRUISE_ALTITUDE_FT"] * 0.3048
HOVER_TURN_N = sum(CONFIG["HOVER_TURN_COUNT_RANGE"]) / 2.0
HOVER_TURN_DURATION_S = CONFIG["HOVER_TURN_DURATION_S"]

N_MISSIONS = 2500
FIXED_MOTOR_EFF = 0.85
FIXED_ESC_EFF = 0.965
FIXED_BATTERY_VOLTAGE_V = 48.0
FIXED_POWER_MARGIN = 1.55
FIXED_ENERGY_MARGIN = 2.5  # already-validated good value
FIXED_GUST_GAIN = 1.25
FIXED_FRAME_STIFF = 0.65
FIXED_LANDING_GEAR_KG = 1.0

TEAMS = [
    ("MTech",    14.51, 53.07, 6, (0.3, 0.7)),
    ("MacGyver", 24.95, 62.14, 4, (0.0, 0.2)),
]

BASELINE_CLIMB, BASELINE_TOUCHDOWN, BASELINE_CRUISE = 4.05, 1.70, 14.6
IMPROVED_CLIMB, IMPROVED_TOUCHDOWN, IMPROVED_CRUISE = 3.75, 2.09, 18.1


def nominal_mission_time_s(cruise_speed_mps, climb_rate_mps):
    cruise_time = (DIST_LOADED_M + DIST_UNLOADED_M) / max(cruise_speed_mps, 0.5)
    climb_descent_time = 4.0 * ALT_TARGET_M / max(climb_rate_mps, 0.5)
    turn_time = HOVER_TURN_N * HOVER_TURN_DURATION_S
    return cruise_time + climb_descent_time + turn_time


def make_design(design_id, empty_mass_kg, payload_mass_kg, rotor_count, tendon_range,
                 climb_rate_mps, touchdown_v, cruise_speed_mps):
    mtow_kg = empty_mass_kg + payload_mass_kg
    hover_power_mech_W = CONFIG["HOVER_POWER_COEFF"] * (mtow_kg ** 1.5)
    hover_power_elec_W = hover_power_mech_W / (FIXED_MOTOR_EFF * FIXED_ESC_EFF)
    battery_max_power_W = hover_power_elec_W * FIXED_POWER_MARGIN

    nom_time_s = nominal_mission_time_s(cruise_speed_mps, climb_rate_mps)
    min_energy_Wh = hover_power_elec_W * nom_time_s / 3600.0
    battery_energy_Wh = min_energy_Wh * FIXED_ENERGY_MARGIN
    battery_spec_energy = 220.0
    battery_mass_kg = battery_energy_Wh / battery_spec_energy

    per_rotor_peak_W = battery_max_power_W / rotor_count
    per_rotor_peak_A = per_rotor_peak_W / FIXED_BATTERY_VOLTAGE_V
    esc_current_rating_A = per_rotor_peak_A * 1.5

    ratio = payload_mass_kg / empty_mass_kg
    tendon_cable_fraction = (tendon_range[0] + tendon_range[1]) / 2.0

    return gen.AircraftDesign(
        design_id=design_id,
        animals=[], traits=[], animal_count=0, trait_count=0,
        empty_mass_kg=empty_mass_kg, payload_mass_kg=payload_mass_kg,
        rotor_count=rotor_count, max_twr=1.2, burst_power_factor=1.0,
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
        battery_nominal_voltage_V=FIXED_BATTERY_VOLTAGE_V,
        battery_max_power_W=battery_max_power_W,
        supercap_mass_kg=0.0, supercap_energy_Wh=0.0, supercap_max_power_W=0.0,
        motor_type="electric_brushless_outunner",
        motor_efficiency=FIXED_MOTOR_EFF, esc_efficiency=FIXED_ESC_EFF,
        esc_current_rating_A=esc_current_rating_A,
        structural_material="carbon_composite",
        structural_material_class="light_stiff",
        structural_extra_failure_risk=0.01,
        rotor_blade_material="carbon_composite",
        landing_gear_material="composite",
        frame_stiffness_longitudinal=FIXED_FRAME_STIFF,
        tendon_cable_fraction=tendon_cable_fraction,
        gust_rejection_gain=FIXED_GUST_GAIN,
        landing_gear_mass_kg=FIXED_LANDING_GEAR_KG,
        max_touchdown_velocity_mps=touchdown_v,
        cruise_speed_mps=cruise_speed_mps, climb_rate_mps=climb_rate_mps, mode_count=2,
        mtow_kg=mtow_kg, payload_to_aircraft_ratio=ratio,
        rule_empty_mass_ok=empty_mass_kg <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
        rule_payload_ok=payload_mass_kg >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],
        design_qualifying=True, design_qualifying_score=ratio,
        design_summary="lever verification test.",
        propulsion_architecture="pure_rotor_electric",
        primary_propulsor_type="multirotor",
        secondary_propulsor_type="none", secondary_propulsor_fraction=0.0,
        propulsion_hover_power_factor=1.0, propulsion_cruise_power_factor=1.0,
        propulsion_arch_extra_failure_risk=0.0,
        wing_foldable=False, wing_deploy_time_s=0.0, wing_deploy_failure_risk=0.0,
        image_prompt="",
    )


def eval_design(design, n_missions, seed):
    rng = random.Random(seed)
    results = []
    for _ in range(n_missions):
        env = gen.generate_environment(rng)
        mission, _ts = gen.simulate_mission(design, env, rng)
        results.append(mission)
    n = len(results)
    success_rate = sum(1 for m in results if m.success) / n
    from collections import Counter
    reasons = Counter(m.failure_reason for m in results if not m.success)
    return success_rate, reasons, n


configs = [
    ("A: baseline (typical touchdown/climb/cruise)", BASELINE_CLIMB, BASELINE_TOUCHDOWN, BASELINE_CRUISE),
    ("B: + touchdown/climb improvement only",         IMPROVED_CLIMB, IMPROVED_TOUCHDOWN, BASELINE_CRUISE),
    ("C: + cruise speed improvement only",             BASELINE_CLIMB, BASELINE_TOUCHDOWN, IMPROVED_CRUISE),
    ("D: + both improvements together",                IMPROVED_CLIMB, IMPROVED_TOUCHDOWN, IMPROVED_CRUISE),
]

for (name, empty_kg, payload_kg, rotor_count, tendon_range) in TEAMS:
    print(f"\n=== {name} ===")
    for label, climb, touchdown, cruise in configs:
        d = make_design(f"{name}_{label[:1]}", empty_kg, payload_kg, rotor_count, tendon_range, climb, touchdown, cruise)
        success_rate, reasons, n = eval_design(d, N_MISSIONS, seed=2001)
        top_reasons = ", ".join(f"{k}={100*v/n:.1f}%" for k, v in reasons.most_common(3))
        print(f"  {label:>45}  success={100*success_rate:5.1f}%   top failures: {top_reasons}")
