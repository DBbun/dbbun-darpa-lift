"""
Design improvement recommendations: for each of the 5 real DARPA Lift
Challenge winners, test concrete modifications the simulator predicts
would improve outcomes, grounded in what was rigorously established
earlier this session:
  - battery ENERGY CAPACITY (not power margin, not structure/gust) is the
    dominant lever for reliability at this mass class (attribution
    analysis round 2)
  - pure ratio-chasing has only a small effect on reliability WHEN power/
    energy are properly provisioned (ratio-sweep ablation, Part 7)

Three configurations per design, all using FIXED (not Monte Carlo-sampled)
representative values for a clean, direct comparison:
  A) baseline: typical mid-range energy margin (1.85x), typical power
     margin (1.55x), real empty_mass/payload (their actual achievement)
  B) improved_battery: same empty_mass/payload, generous energy margin
     (2.5x) instead of typical
  C) improved_battery_and_ratio: generous energy margin (2.5x) AND empty
     mass reduced 8% (payload held fixed -> ratio increases)

2000 missions per configuration for a precise estimate.

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

N_MISSIONS = 2000

TEAMS = [
    ("AVIDrone",      13.2,  50.8,  4, 15.0, (0.0, 0.2)),
    ("MTech",         14.51, 53.07, 6, 15.0, (0.3, 0.7)),
    ("Xtreme Aerial",  24.95, 85.73, 4, 15.0, (0.0, 0.2)),
    ("H-Squared",     24.81, 73.44, 4, 13.8, (0.0, 0.2)),
    ("MacGyver",      24.95, 62.14, 4, 15.0, (0.0, 0.2)),
]

FIXED_MOTOR_EFF = 0.85
FIXED_ESC_EFF = 0.965
FIXED_BATTERY_VOLTAGE_V = 48.0
FIXED_POWER_MARGIN = 1.55
FIXED_CLIMB_RATE = 4.0
FIXED_TOUCHDOWN_V = 1.5
FIXED_GUST_GAIN = 1.25
FIXED_FRAME_STIFF = 0.65
FIXED_LANDING_GEAR_KG = 1.0
FIXED_TENDON_MID = lambda rng_range: (rng_range[0] + rng_range[1]) / 2.0


def nominal_mission_time_s(cruise_speed_mps, climb_rate_mps):
    cruise_time = (DIST_LOADED_M + DIST_UNLOADED_M) / max(cruise_speed_mps, 0.5)
    climb_descent_time = 4.0 * ALT_TARGET_M / max(climb_rate_mps, 0.5)
    turn_time = HOVER_TURN_N * HOVER_TURN_DURATION_S
    return cruise_time + climb_descent_time + turn_time


def make_design(design_id, empty_mass_kg, payload_mass_kg, rotor_count, cruise_speed_mps,
                 tendon_range, energy_margin):
    mtow_kg = empty_mass_kg + payload_mass_kg
    hover_power_mech_W = CONFIG["HOVER_POWER_COEFF"] * (mtow_kg ** 1.5)
    hover_power_elec_W = hover_power_mech_W / (FIXED_MOTOR_EFF * FIXED_ESC_EFF)
    battery_max_power_W = hover_power_elec_W * FIXED_POWER_MARGIN

    nom_time_s = nominal_mission_time_s(cruise_speed_mps, FIXED_CLIMB_RATE)
    min_energy_Wh = hover_power_elec_W * nom_time_s / 3600.0
    battery_energy_Wh = min_energy_Wh * energy_margin
    battery_spec_energy = 220.0
    battery_mass_kg = battery_energy_Wh / battery_spec_energy

    per_rotor_peak_W = battery_max_power_W / rotor_count
    per_rotor_peak_A = per_rotor_peak_W / FIXED_BATTERY_VOLTAGE_V
    esc_current_rating_A = per_rotor_peak_A * 1.5

    ratio = payload_mass_kg / empty_mass_kg
    tendon_cable_fraction = FIXED_TENDON_MID(tendon_range)

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
        max_touchdown_velocity_mps=FIXED_TOUCHDOWN_V,
        cruise_speed_mps=cruise_speed_mps, climb_rate_mps=FIXED_CLIMB_RATE, mode_count=2,
        mtow_kg=mtow_kg, payload_to_aircraft_ratio=ratio,
        rule_empty_mass_ok=empty_mass_kg <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
        rule_payload_ok=payload_mass_kg >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],
        design_qualifying=True, design_qualifying_score=ratio,
        design_summary="design improvement test.",
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
    return success_rate


print(f"{'Team':>15} {'Config':>28} {'ratio':>7} {'success_rate':>13}")
for (name, empty_kg, payload_kg, rotor_count, cruise, tendon_range) in TEAMS:
    d_baseline = make_design(f"{name}_A_baseline", empty_kg, payload_kg, rotor_count, cruise, tendon_range, energy_margin=1.85)
    s_baseline = eval_design(d_baseline, N_MISSIONS, seed=1001)
    print(f"{name:>15} {'A: baseline (1.85x energy)':>28} {d_baseline.payload_to_aircraft_ratio:7.2f} {100*s_baseline:12.1f}%")

    d_battery = make_design(f"{name}_B_battery", empty_kg, payload_kg, rotor_count, cruise, tendon_range, energy_margin=2.5)
    s_battery = eval_design(d_battery, N_MISSIONS, seed=1002)
    print(f"{name:>15} {'B: +battery margin (2.5x)':>28} {d_battery.payload_to_aircraft_ratio:7.2f} {100*s_battery:12.1f}%")

    empty_reduced = empty_kg * 0.92
    d_combined = make_design(f"{name}_C_combined", empty_reduced, payload_kg, rotor_count, cruise, tendon_range, energy_margin=2.5)
    s_combined = eval_design(d_combined, N_MISSIONS, seed=1003)
    print(f"{name:>15} {'C: B + 8% lighter airframe':>28} {d_combined.payload_to_aircraft_ratio:7.2f} {100*s_combined:12.1f}%")
    print()
