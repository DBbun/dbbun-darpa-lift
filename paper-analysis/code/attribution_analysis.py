"""
Attribution analysis: for the 3 real designs whose Monte Carlo rank_score
distribution had huge variance (Xtreme Aerial, H-Squared, MacGyver),
isolate which uncertain field(s) actually drive that variance.

Two conditions per team:
  A) "touchdown_only": vary ONLY climb_rate_mps and max_touchdown_velocity_mps
     (the hypothesis: their ratio drives p_touchdown, which can spike
     sharply for unlucky draws). Every other field held at a fixed
     reasonable point estimate.
  B) "everything_else": vary every other uncertain field (battery specs,
     power margin, motor/ESC efficiency, gust rejection, frame stiffness,
     landing gear mass, tendon fraction, cruise speed where not measured)
     while holding climb_rate_mps and max_touchdown_velocity_mps FIXED at
     point estimates.

If condition A reproduces most of the full Monte Carlo's variance and B
does not, that confirms climb_rate/touchdown_velocity as the primary
driver. If B is comparably wide, the hypothesis is wrong or incomplete.

Does not touch darpa_lift_challenge_generator_v1_2.py -- only imports it.
"""
import sys
import random
import statistics

sys.path.insert(0, r"C:\DBBun\Code\DARPA Lift Challenge\v1.2\src")

import darpa_lift_challenge_generator_v1_2 as gen

CONFIG = gen.CONFIG
LI_ION = gen.ENERGY_SYSTEM_PROFILES["li_ion"]

N_DRAWS = 40
N_MISSIONS_PER_DRAW = 300

# point-estimate fixed values (same style as the original case studies)
FIXED = dict(
    battery_mass_kg=4.0, battery_spec_energy=220.0, battery_voltage_V=48.0,
    power_margin=1.5, motor_eff=0.87, esc_eff=0.96,
    cruise_speed_mps=15.0, climb_rate_mps=3.5, max_touchdown_velocity_mps=1.5,
    gust_rejection_gain=1.0, frame_stiffness=0.65, landing_gear_mass_kg=1.0,
    tendon_cable_fraction=0.05,
)

TEAMS = [
    ("Xtreme Aerial", 24.95, 85.73, 4, None),
    ("H-Squared",      24.81, 73.44, 4, 13.8),
    ("MacGyver",       24.95, 62.14, 4, None),
]


def make_design(design_id, empty_mass_kg, payload_mass_kg, rotor_count, fixed_cruise, vals):
    mtow_kg = empty_mass_kg + payload_mass_kg
    ratio = payload_mass_kg / empty_mass_kg
    hover_power_mech_W = CONFIG["HOVER_POWER_COEFF"] * (mtow_kg ** 1.5)
    hover_power_elec_W = hover_power_mech_W / max(vals["motor_eff"] * vals["esc_eff"], 1e-6)
    battery_max_power_W = hover_power_elec_W * vals["power_margin"]

    per_rotor_peak_W = battery_max_power_W / rotor_count
    per_rotor_peak_A = per_rotor_peak_W / vals["battery_voltage_V"]
    esc_current_rating_A = per_rotor_peak_A * 1.5

    cruise_speed_mps = fixed_cruise if fixed_cruise is not None else vals["cruise_speed_mps"]

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
        battery_mass_kg=vals["battery_mass_kg"],
        battery_spec_energy_Wh_per_kg=vals["battery_spec_energy"],
        battery_energy_Wh=vals["battery_mass_kg"] * vals["battery_spec_energy"],
        battery_nominal_voltage_V=vals["battery_voltage_V"],
        battery_max_power_W=battery_max_power_W,
        supercap_mass_kg=0.0, supercap_energy_Wh=0.0, supercap_max_power_W=0.0,
        motor_type="electric_brushless_outunner",
        motor_efficiency=vals["motor_eff"], esc_efficiency=vals["esc_eff"],
        esc_current_rating_A=esc_current_rating_A,
        structural_material="carbon_composite",
        structural_material_class="light_stiff",
        structural_extra_failure_risk=0.01,
        rotor_blade_material="carbon_composite",
        landing_gear_material="composite",
        frame_stiffness_longitudinal=vals["frame_stiffness"],
        tendon_cable_fraction=vals["tendon_cable_fraction"],
        gust_rejection_gain=vals["gust_rejection_gain"],
        landing_gear_mass_kg=vals["landing_gear_mass_kg"],
        max_touchdown_velocity_mps=vals["max_touchdown_velocity_mps"],
        cruise_speed_mps=cruise_speed_mps, climb_rate_mps=vals["climb_rate_mps"], mode_count=2,
        mtow_kg=mtow_kg, payload_to_aircraft_ratio=ratio,
        rule_empty_mass_ok=empty_mass_kg <= CONFIG["DARPA_MAX_EMPTY_MASS_KG"],
        rule_payload_ok=payload_mass_kg >= CONFIG["DARPA_MIN_PAYLOAD_MASS_KG"],
        design_qualifying=True, design_qualifying_score=ratio,
        design_summary="attribution analysis draw.",
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
    return rank_score


master_rng = random.Random(20260810)

print(f"{'Team':>15} {'Condition':>16} {'mean':>7} {'median':>8} {'stdev':>7} {'min':>6} {'max':>6}")

for (name, empty_kg, payload_kg, rotor_count, fixed_cruise) in TEAMS:
    for condition in ("touchdown_only", "everything_else"):
        scores = []
        for draw_i in range(N_DRAWS):
            rng = random.Random(master_rng.randint(0, 2**31 - 1))
            vals = dict(FIXED)
            if condition == "touchdown_only":
                vals["climb_rate_mps"] = rng.uniform(*CONFIG["CLIMB_RATE_RANGE_MPS"])
                vals["max_touchdown_velocity_mps"] = rng.uniform(*CONFIG["MAX_TOUCHDOWN_VELOCITY_MPS_RANGE"])
            else:
                vals["battery_mass_kg"] = rng.uniform(1.5, 6.0)
                vals["battery_spec_energy"] = rng.uniform(*LI_ION["spec_energy_range"])
                vals["battery_voltage_V"] = rng.uniform(36.0, 60.0)
                vals["power_margin"] = rng.uniform(1.1, 2.0)
                vals["motor_eff"] = rng.uniform(*CONFIG["MOTOR_EFFICIENCY_RANGE"])
                vals["esc_eff"] = rng.uniform(*CONFIG["ESC_EFFICIENCY_RANGE"])
                vals["cruise_speed_mps"] = rng.uniform(10.0, 20.0)
                vals["gust_rejection_gain"] = rng.uniform(*CONFIG["GUST_REJECTION_GAIN_RANGE"])
                vals["frame_stiffness"] = rng.uniform(*CONFIG["FRAME_STIFFNESS_LONGITUDINAL_RANGE"])
                vals["landing_gear_mass_kg"] = rng.uniform(*CONFIG["LANDING_GEAR_MASS_KG_RANGE"])
                vals["tendon_cable_fraction"] = rng.uniform(0.0, 0.2)

            d = make_design(f"DLIFT_ATTR_{name}_{condition}_{draw_i}", empty_kg, payload_kg, rotor_count, fixed_cruise, vals)
            score = rank_score_for(d, N_MISSIONS_PER_DRAW, rng)
            scores.append(score)

        print(f"{name:>15} {condition:>16} {statistics.mean(scores):7.3f} {statistics.median(scores):8.3f} {statistics.stdev(scores):7.3f} {min(scores):6.3f} {max(scores):6.3f}")
