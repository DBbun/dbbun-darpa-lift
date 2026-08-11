"""
AVIDrone (Team #27) and Xtreme Aerial (Team #65) case studies: construct fixed
AircraftDesign objects matching their real-world specs and run them through the
ORIGINAL, UNMODIFIED generator's mission simulation logic (imported, not
copied/edited).

Both are real single-main-rotor + tail-rotor helicopters -- an architecture the
generator's schema cannot express. rotor_count is force-set to 4 (the schema's
floor) and primary_propulsor_type to multirotor/gas_multirotor as the closest
available category. This is flagged, not hidden.

Does not touch darpa_lift_challenge_generator_v1_2.py -- only imports it.
"""
import sys
import random
from collections import Counter

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
LI_ION = gen.ENERGY_SYSTEM_PROFILES["li_ion"]


def make_design(design_id, empty_mass_kg, payload_mass_kg, rotor_count,
                 propulsion_architecture, primary_propulsor_type,
                 energy_system_type, energy_density_class, power_class,
                 tech_maturity_class, energy_system_extra_failure_risk,
                 battery_mass_kg, battery_spec_energy, battery_voltage_V,
                 battery_max_power_W, motor_efficiency, esc_efficiency,
                 structural_material, structural_material_class,
                 landing_gear_material, landing_gear_mass_kg,
                 max_touchdown_velocity_mps, frame_stiffness_longitudinal,
                 tendon_cable_fraction, cruise_speed_mps, climb_rate_mps,
                 propulsion_hover_power_factor, propulsion_cruise_power_factor,
                 propulsion_arch_extra_failure_risk, energy_system_description):
    mtow_kg = empty_mass_kg + payload_mass_kg
    ratio = payload_mass_kg / empty_mass_kg
    per_rotor_peak_W = battery_max_power_W / rotor_count
    per_rotor_peak_A = per_rotor_peak_W / battery_voltage_V
    esc_current_rating_A = per_rotor_peak_A * 1.5

    return gen.AircraftDesign(
        design_id=design_id,
        animals=[], traits=[], animal_count=0, trait_count=0,
        empty_mass_kg=empty_mass_kg, payload_mass_kg=payload_mass_kg,
        rotor_count=rotor_count, max_twr=1.2, burst_power_factor=1.0,
        burst_duration_s=0.0, unsteady_lift_gain=0.0,
        energy_system_type=energy_system_type,
        energy_system_description=energy_system_description,
        energy_density_class=energy_density_class, power_class=power_class,
        tech_maturity_class=tech_maturity_class,
        energy_system_extra_failure_risk=energy_system_extra_failure_risk,
        battery_mass_kg=battery_mass_kg,
        battery_spec_energy_Wh_per_kg=battery_spec_energy,
        battery_energy_Wh=battery_mass_kg * battery_spec_energy,
        battery_nominal_voltage_V=battery_voltage_V,
        battery_max_power_W=battery_max_power_W,
        supercap_mass_kg=0.0, supercap_energy_Wh=0.0, supercap_max_power_W=0.0,
        motor_type="electric_brushless_outunner",
        motor_efficiency=motor_efficiency, esc_efficiency=esc_efficiency,
        esc_current_rating_A=esc_current_rating_A,
        structural_material=structural_material,
        structural_material_class=structural_material_class,
        structural_extra_failure_risk=0.01,
        rotor_blade_material="carbon_composite",
        landing_gear_material=landing_gear_material,
        frame_stiffness_longitudinal=frame_stiffness_longitudinal,
        tendon_cable_fraction=tendon_cable_fraction,
        gust_rejection_gain=1.0,
        landing_gear_mass_kg=landing_gear_mass_kg,
        max_touchdown_velocity_mps=max_touchdown_velocity_mps,
        cruise_speed_mps=cruise_speed_mps, climb_rate_mps=climb_rate_mps,
        mode_count=2,
        mtow_kg=mtow_kg, payload_to_aircraft_ratio=ratio,
        rule_empty_mass_ok=empty_mass_kg <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
        rule_payload_ok=payload_mass_kg >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],
        design_qualifying=True, design_qualifying_score=ratio,
        design_summary=f"{design_id} case study (real-world reconstruction).",
        propulsion_architecture=propulsion_architecture,
        primary_propulsor_type=primary_propulsor_type,
        secondary_propulsor_type="none", secondary_propulsor_fraction=0.0,
        propulsion_hover_power_factor=propulsion_hover_power_factor,
        propulsion_cruise_power_factor=propulsion_cruise_power_factor,
        propulsion_arch_extra_failure_risk=propulsion_arch_extra_failure_risk,
        wing_foldable=False, wing_deploy_time_s=0.0, wing_deploy_failure_risk=0.0,
        image_prompt="",
    )


avidrone = make_design(
    design_id="DLIFT_AVIDRONE_CASE",
    empty_mass_kg=13.2, payload_mass_kg=50.8, rotor_count=4,
    propulsion_architecture="pure_rotor_electric", primary_propulsor_type="multirotor",
    energy_system_type="li_ion", energy_density_class=LI_ION["energy_density_class"],
    power_class=LI_ION["power_class"], tech_maturity_class=LI_ION["tech_maturity_class"],
    energy_system_extra_failure_risk=LI_ION["extra_failure_risk"],
    battery_mass_kg=3.5, battery_spec_energy=220.0, battery_voltage_V=48.0,
    battery_max_power_W=5200.0, motor_efficiency=0.89, esc_efficiency=0.97,
    structural_material="carbon_composite", structural_material_class="light_stiff",
    landing_gear_material="composite", landing_gear_mass_kg=1.0,
    max_touchdown_velocity_mps=1.5, frame_stiffness_longitudinal=0.65,
    tendon_cable_fraction=0.0, cruise_speed_mps=15.0, climb_rate_mps=3.5,
    propulsion_hover_power_factor=1.0, propulsion_cruise_power_factor=1.0,
    propulsion_arch_extra_failure_risk=0.0,
    energy_system_description=LI_ION["description"],
)

xtreme = make_design(
    design_id="DLIFT_XTREME_CASE",
    empty_mass_kg=24.95, payload_mass_kg=85.73, rotor_count=4,
    propulsion_architecture="pure_gas_multirotor", primary_propulsor_type="gas_multirotor",
    energy_system_type="gas_or_hybrid", energy_density_class="extreme",
    power_class="high", tech_maturity_class="high",
    energy_system_extra_failure_risk=0.02,
    battery_mass_kg=3.0, battery_spec_energy=3800.0, battery_voltage_V=48.0,
    battery_max_power_W=13000.0, motor_efficiency=0.85, esc_efficiency=0.96,
    structural_material="aluminum_lithium", structural_material_class="medium_stiff",
    landing_gear_material="steel", landing_gear_mass_kg=2.5,
    max_touchdown_velocity_mps=2.0, frame_stiffness_longitudinal=0.75,
    tendon_cable_fraction=0.15, cruise_speed_mps=15.0, climb_rate_mps=3.5,
    propulsion_hover_power_factor=1.05, propulsion_cruise_power_factor=0.90,
    propulsion_arch_extra_failure_risk=0.04,
    energy_system_description="Gas/ICE-driven multirotor, effective fuel energy density modeled per pure_gas_multirotor architecture.",
)


def run_case_study(design, n_missions=3000, seed=20260810):
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

    print(f"\n=== {design.design_id} ===")
    print(f"empty={design.empty_mass_kg:.2f}kg payload={design.payload_mass_kg:.2f}kg ratio={design.payload_to_aircraft_ratio:.3f}")
    print(f"success_rate={success_rate:.3f}  qualifying_rate={qualifying_rate:.3f}  rule_penalty_rate={rule_penalty_rate:.3f}")
    print(f"design_rank_score={rank_score:.3f}")

    reasons = Counter(m.failure_reason for m in results if not m.success)
    print(f"Failure reasons ({sum(reasons.values())} failures):")
    for reason, cnt in reasons.most_common():
        print(f"  {reason}: {cnt} ({100*cnt/n:.1f}%)")

    sat_seconds = [m.power_saturation_seconds for m in results]
    thermal_peaks = [m.thermal_peak_C for m in results]
    print(f"power_saturation_seconds: max={max(sat_seconds):.1f} avg={sum(sat_seconds)/n:.2f}")
    print(f"thermal_peak_C: max={max(thermal_peaks):.1f} avg={sum(thermal_peaks)/n:.2f}")

    full_prize = sum(1 for m in results if m.prize_tier == "full")
    partial_prize = sum(1 for m in results if m.prize_tier == "partial")
    print(f"prize_tier: full={100*full_prize/n:.1f}%  partial={100*partial_prize/n:.1f}%  none={100*(n-full_prize-partial_prize)/n:.1f}%")
    return rank_score, success_rate


run_case_study(avidrone)
run_case_study(xtreme)
