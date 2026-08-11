"""
Monte Carlo sensitivity analysis v2: fixes the artifact found in round-2
attribution analysis. v1 sampled battery_mass_kg and battery_spec_energy
independently of aircraft size, letting unrealistically undersized
batteries appear for heavy real aircraft (up to ~39% energy_depleted
failures for Xtreme Aerial). v2 derives battery_energy_Wh from a margin
(1.2x-2.5x) over the aircraft's own nominal-mission energy requirement --
the same principle already used for battery_max_power_W vs. hover power --
then back-derives battery_mass_kg from that energy and an independently
sampled spec energy (a technology choice, not a size-dependent one).

Nominal mission energy uses the same course geometry the generator itself
uses (4.0 nm loaded leg, 1.0 nm unloaded leg, 350 ft cruise altitude,
~18 hover turns x 12s) and treats the whole nominal duration at
hover-power level (conservative/safe overestimate vs. a real energy
budget, since much of the course is cheaper cruise flight).

Does not touch darpa_lift_challenge_generator_v1_2.py -- only imports it.
"""
import sys
import random
import statistics

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
LI_ION = gen.ENERGY_SYSTEM_PROFILES["li_ion"]

N_DRAWS = 100
N_MISSIONS_PER_DRAW = 400

DIST_LOADED_M = CONFIG["PAYLOAD_LEG_DISTANCE_NM"] * 1852.0
DIST_UNLOADED_M = CONFIG["RETURN_LEG_DISTANCE_NM"] * 1852.0
ALT_TARGET_M = CONFIG["CRUISE_ALTITUDE_FT"] * 0.3048
HOVER_TURN_N = sum(CONFIG["HOVER_TURN_COUNT_RANGE"]) / 2.0
HOVER_TURN_DURATION_S = CONFIG["HOVER_TURN_DURATION_S"]

TEAMS = [
    ("AVIDrone",      13.2,  50.8,  4, None, (0.0, 0.2), 3.85, "1st, $1.25M"),
    ("MTech",         14.51, 53.07, 6, None, (0.3, 0.7), 3.66, "2nd, $750K"),
    ("Xtreme Aerial",  24.95, 85.73, 4, None, (0.0, 0.2), 3.44, "3rd, $500K"),
    ("H-Squared",     24.81, 73.44, 4, 13.8, (0.0, 0.2), 2.96, "4th"),
    ("MacGyver",      24.95, 62.14, 4, None, (0.0, 0.2), 2.49, "5th"),
    ("DefendTex",      8.347, 50.98, 4, None, (0.3, 0.7), 6.11, "crashed, delisted"),
]


def nominal_mission_time_s(cruise_speed_mps, climb_rate_mps):
    cruise_time = (DIST_LOADED_M + DIST_UNLOADED_M) / max(cruise_speed_mps, 0.5)
    climb_descent_time = 4.0 * ALT_TARGET_M / max(climb_rate_mps, 0.5)
    turn_time = HOVER_TURN_N * HOVER_TURN_DURATION_S
    return cruise_time + climb_descent_time + turn_time


def sample_design(design_id, empty_mass_kg, payload_mass_kg, rotor_count,
                   fixed_cruise_speed, tendon_range, rng):
    motor_eff = rng.uniform(*CONFIG["MOTOR_EFFICIENCY_RANGE"])
    esc_eff = rng.uniform(*CONFIG["ESC_EFFICIENCY_RANGE"])
    battery_voltage_V = rng.uniform(36.0, 60.0)

    mtow_kg = empty_mass_kg + payload_mass_kg
    hover_power_mech_W = CONFIG["HOVER_POWER_COEFF"] * (mtow_kg ** 1.5)
    hover_power_elec_W = hover_power_mech_W / max(motor_eff * esc_eff, 1e-6)

    power_margin = rng.uniform(1.1, 2.0)
    battery_max_power_W = hover_power_elec_W * power_margin

    cruise_speed_mps = fixed_cruise_speed if fixed_cruise_speed is not None else rng.uniform(10.0, 20.0)
    climb_rate_mps = rng.uniform(*CONFIG["CLIMB_RATE_RANGE_MPS"])

    nom_time_s = nominal_mission_time_s(cruise_speed_mps, climb_rate_mps)
    min_energy_Wh = hover_power_elec_W * nom_time_s / 3600.0
    energy_margin = rng.uniform(1.2, 2.5)
    battery_energy_Wh = min_energy_Wh * energy_margin

    battery_spec_energy = rng.uniform(*LI_ION["spec_energy_range"])
    battery_mass_kg = battery_energy_Wh / battery_spec_energy

    per_rotor_peak_W = battery_max_power_W / rotor_count
    per_rotor_peak_A = per_rotor_peak_W / battery_voltage_V
    esc_current_rating_A = per_rotor_peak_A * rng.uniform(1.2, 1.8)

    max_touchdown_velocity_mps = rng.uniform(*CONFIG["MAX_TOUCHDOWN_VELOCITY_MPS_RANGE"])
    gust_rejection_gain = rng.uniform(*CONFIG["GUST_REJECTION_GAIN_RANGE"])
    frame_stiffness = rng.uniform(*CONFIG["FRAME_STIFFNESS_LONGITUDINAL_RANGE"])
    landing_gear_mass_kg = rng.uniform(*CONFIG["LANDING_GEAR_MASS_KG_RANGE"])
    tendon_cable_fraction = rng.uniform(*tendon_range)

    ratio = payload_mass_kg / empty_mass_kg

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
        battery_nominal_voltage_V=battery_voltage_V,
        battery_max_power_W=battery_max_power_W,
        supercap_mass_kg=0.0, supercap_energy_Wh=0.0, supercap_max_power_W=0.0,
        motor_type="electric_brushless_outunner",
        motor_efficiency=motor_eff, esc_efficiency=esc_eff,
        esc_current_rating_A=esc_current_rating_A,
        structural_material="carbon_composite",
        structural_material_class="light_stiff",
        structural_extra_failure_risk=0.01,
        rotor_blade_material="carbon_composite",
        landing_gear_material="composite",
        frame_stiffness_longitudinal=frame_stiffness,
        tendon_cable_fraction=tendon_cable_fraction,
        gust_rejection_gain=gust_rejection_gain,
        landing_gear_mass_kg=landing_gear_mass_kg,
        max_touchdown_velocity_mps=max_touchdown_velocity_mps,
        cruise_speed_mps=cruise_speed_mps, climb_rate_mps=climb_rate_mps, mode_count=2,
        mtow_kg=mtow_kg, payload_to_aircraft_ratio=ratio,
        rule_empty_mass_ok=empty_mass_kg <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
        rule_payload_ok=payload_mass_kg >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],
        design_qualifying=True, design_qualifying_score=ratio,
        design_summary="Monte Carlo v2 sensitivity draw (energy-capacity-corrected).",
        propulsion_architecture="pure_rotor_electric",
        primary_propulsor_type="multirotor",
        secondary_propulsor_type="none", secondary_propulsor_fraction=0.0,
        propulsion_hover_power_factor=1.0, propulsion_cruise_power_factor=1.0,
        propulsion_arch_extra_failure_risk=0.0,
        wing_foldable=False, wing_deploy_time_s=0.0, wing_deploy_failure_risk=0.0,
        image_prompt="",
    )


def rank_score_for(design, n_missions, rng):
    results = []
    for _ in range(n_missions):
        env = gen.generate_environment(rng)
        mission, _ts = gen.simulate_mission(design, env, rng)
        results.append(mission)
    n = len(results)
    success_rate = sum(1 for m in results if m.success) / n
    qualifying_rate = sum(1 for m in results if m.is_qualifying_run) / n
    rule_penalty_rate = sum(1 for m in results if m.rule_violation) / n
    w_success = CONFIG["RANK_W_SUCCESS_RATE"]
    w_qual = CONFIG["RANK_W_QUAL_RATE"]
    w_pen = CONFIG["RANK_W_RULE_PENALTY"]
    rank_score = success_rate * w_success + qualifying_rate * w_qual + (1.0 - rule_penalty_rate) * w_pen
    return success_rate, rank_score


master_rng = random.Random(20260810)
per_team_scores = {t[0]: [] for t in TEAMS}
per_team_success = {t[0]: [] for t in TEAMS}

for draw_i in range(N_DRAWS):
    draw_seed = master_rng.randint(0, 2**31 - 1)
    for (name, empty_kg, payload_kg, rotor_count, fixed_cruise, tendon_range, real_ratio, placement) in TEAMS:
        rng = random.Random(draw_seed + hash(name) % 10000)
        d = sample_design(f"DLIFT_MCv2_{name}_{draw_i}", empty_kg, payload_kg, rotor_count, fixed_cruise, tendon_range, rng)
        success_rate, rank_score = rank_score_for(d, N_MISSIONS_PER_DRAW, rng)
        per_team_scores[name].append(rank_score)
        per_team_success[name].append(success_rate)

    if (draw_i + 1) % 10 == 0:
        print(f"draw {draw_i+1}/{N_DRAWS} done")

print(f"\n=== Monte Carlo v2 (energy-capacity-corrected) results ({N_DRAWS} draws x {N_MISSIONS_PER_DRAW} missions each) ===\n")
print(f"{'Team':>15} {'real_ratio':>10} {'real_place':>18} {'mean_rank':>10} {'median_rank':>12} {'stdev':>8} {'min':>7} {'max':>7}")
for (name, empty_kg, payload_kg, rotor_count, fixed_cruise, tendon_range, real_ratio, placement) in TEAMS:
    scores = per_team_scores[name]
    print(f"{name:>15} {real_ratio:10.2f} {placement:>18} {statistics.mean(scores):10.3f} {statistics.median(scores):12.3f} {statistics.stdev(scores):8.3f} {min(scores):7.3f} {max(scores):7.3f}")

print(f"\n=== Success rate (directly interpretable: % of simulated attempts that complete the course) ===\n")
print(f"{'Team':>15} {'real_ratio':>10} {'real_place':>18} {'mean_succ':>10} {'median_succ':>12} {'stdev':>8} {'min':>7} {'max':>7}")
for (name, empty_kg, payload_kg, rotor_count, fixed_cruise, tendon_range, real_ratio, placement) in TEAMS:
    succ = per_team_success[name]
    print(f"{name:>15} {real_ratio:10.2f} {placement:>18} {statistics.mean(succ):10.3f} {statistics.median(succ):12.3f} {statistics.stdev(succ):8.3f} {min(succ):7.3f} {max(succ):7.3f}")
