"""
Controlled ablation: hold MTOW (and every other design field) fixed, sweep
ONLY the empty/payload split -- i.e. only payload_to_aircraft_ratio --
across 2.0 to 8.0. Isolates the generator's load_term/stress_index
mechanism from the confound of total mass also changing hover power
demand (which stays IDENTICAL across the whole sweep since MTOW is fixed
and battery specs are absolute, not mass-relative).

Does not touch darpa_lift_challenge_generator_v1_2.py -- only imports it.
"""
import sys
import random
from collections import Counter

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
LI_ION = gen.ENERGY_SYSTEM_PROFILES["li_ion"]

MTOW_KG = 75.0  # chosen so empty<=24.95kg and payload>=49.9kg (DARPA rules) hold for all ratios >= ~2.1
ROTOR_COUNT = 6

def make_design(ratio):
    empty_mass_kg = MTOW_KG / (1.0 + ratio)
    payload_mass_kg = MTOW_KG - empty_mass_kg

    battery_mass_kg = 4.5
    battery_spec_energy = 220.0
    battery_voltage_V = 48.0
    battery_max_power_W = 7500.0  # fixed absolute -- sized for MTOW=65kg hover with margin, held constant across sweep

    per_rotor_peak_W = battery_max_power_W / ROTOR_COUNT
    per_rotor_peak_A = per_rotor_peak_W / battery_voltage_V
    esc_current_rating_A = per_rotor_peak_A * 1.5

    return gen.AircraftDesign(
        design_id=f"DLIFT_SWEEP_{ratio:.2f}",
        animals=[], traits=[], animal_count=0, trait_count=0,
        empty_mass_kg=empty_mass_kg, payload_mass_kg=payload_mass_kg,
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
        battery_energy_Wh=battery_mass_kg * battery_spec_energy,
        battery_nominal_voltage_V=battery_voltage_V,
        battery_max_power_W=battery_max_power_W,
        supercap_mass_kg=0.0, supercap_energy_Wh=0.0, supercap_max_power_W=0.0,
        motor_type="electric_brushless_outunner",
        motor_efficiency=0.87, esc_efficiency=0.96,
        esc_current_rating_A=esc_current_rating_A,
        structural_material="carbon_composite",
        structural_material_class="light_stiff",
        structural_extra_failure_risk=0.01,
        rotor_blade_material="carbon_composite",
        landing_gear_material="composite",
        frame_stiffness_longitudinal=0.65,
        tendon_cable_fraction=0.0,
        gust_rejection_gain=1.0,
        landing_gear_mass_kg=1.0,
        max_touchdown_velocity_mps=1.5,
        cruise_speed_mps=15.0, climb_rate_mps=3.5, mode_count=2,
        mtow_kg=MTOW_KG, payload_to_aircraft_ratio=ratio,
        rule_empty_mass_ok=empty_mass_kg <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
        rule_payload_ok=payload_mass_kg >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],
        design_qualifying=True, design_qualifying_score=ratio,
        design_summary="ratio-sweep ablation point.",
        propulsion_architecture="pure_rotor_electric",
        primary_propulsor_type="multirotor",
        secondary_propulsor_type="none", secondary_propulsor_fraction=0.0,
        propulsion_hover_power_factor=1.0, propulsion_cruise_power_factor=1.0,
        propulsion_arch_extra_failure_risk=0.0,
        wing_foldable=False, wing_deploy_time_s=0.0, wing_deploy_failure_risk=0.0,
        image_prompt="",
    )


def run(design, n_missions=2500, seed=20260810):
    rng = random.Random(seed)
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
    reasons = Counter(m.failure_reason for m in results if not m.success)
    gust_rate = reasons.get("gust_induced_instability", 0) / n
    return success_rate, rank_score, gust_rate, reasons


print(f"MTOW fixed at {MTOW_KG} kg; battery/motor/structure/mission-profile fields fixed across the whole sweep.")
print(f"{'ratio':>6} {'empty_kg':>9} {'payload_kg':>11} {'load_term':>10} {'success%':>9} {'rank_score':>11} {'gust%':>7}")

ratios = [2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
for ratio in ratios:
    d = make_design(ratio)
    load_term = max(ratio - 2.0, 0.0) / 3.0
    success_rate, rank_score, gust_rate, reasons = run(d)
    print(f"{ratio:6.2f} {d.empty_mass_kg:9.2f} {d.payload_mass_kg:11.2f} {load_term:10.3f} {100*success_rate:8.1f}% {rank_score:11.3f} {100*gust_rate:6.1f}%")
