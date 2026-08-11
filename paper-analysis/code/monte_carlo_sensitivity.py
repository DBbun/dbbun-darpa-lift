"""
Monte Carlo sensitivity analysis: for each of the 6 real teams, empty_mass_kg
and payload_mass_kg are FIXED (measured/leaderboard values -- not
uncertain). rotor_count is fixed where visually confirmed (MacGyver=4
genuine quad, MTech=6) or forced to the schema floor (AVIDrone/Xtreme
Aerial/H-Squared/DefendTex=4, real architecture unrepresentable).
cruise_speed_mps is fixed for H-Squared only (13.8 m/s, read off telemetry).

All other fields (~10) are uncertain -- sampled uniformly per draw from
the SAME plausible ranges the generator itself uses for random design
generation (CONFIG[...RANGE] / ENERGY_SYSTEM_PROFILES["li_ion"]), so the
sensitivity test uses the model's own stated plausibility bounds rather
than arbitrary ones. battery_max_power_W is derived from a sampled power
margin (1.1x-2.0x of the hover-power requirement) rather than sampled
freely, so battery sizing stays physically tied to what wattage the
airframe actually needs.

For each team: N_DRAWS random parameter draws x N_MISSIONS_PER_DRAW
simulated missions each. Reports the distribution of design_rank_score
per team, and checks how often the full real-placement ordering is
preserved across draws (paired by seed across teams, so each draw
represents one coherent "how good is my guess" scenario. Note: since
each team's params are drawn independently, "does the ranking hold"
is evaluated by comparing the mean/median and spread of each team's
distribution, and by a paired bootstrap: for each of N_DRAWS iterations,
draw one random parameter set per team, run missions, and check whether
that iteration's 6 rank_scores are ordered as MacGyver > H-Squared >
Xtreme Aerial > MTech > AVIDrone > DefendTex (the inverse-of-real-
placement order found with point estimates).

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

# (team, empty_mass_kg, payload_mass_kg, rotor_count, fixed_cruise_speed_mps or None,
#  tendon_range, real_ratio, real_placement_label)
TEAMS = [
    ("AVIDrone",      13.2,  50.8,  4, None, (0.0, 0.2), 3.85, "1st, $1.25M"),
    ("MTech",         14.51, 53.07, 6, None, (0.3, 0.7), 3.66, "2nd, $750K"),
    ("Xtreme Aerial",  24.95, 85.73, 4, None, (0.0, 0.2), 3.44, "3rd, $500K"),
    ("H-Squared",     24.81, 73.44, 4, 13.8, (0.0, 0.2), 2.96, "4th"),
    ("MacGyver",      24.95, 62.14, 4, None, (0.0, 0.2), 2.49, "5th"),
    ("DefendTex",      8.347, 50.98, 4, None, (0.3, 0.7), 6.11, "crashed, delisted"),
]


def sample_design(design_id, empty_mass_kg, payload_mass_kg, rotor_count,
                   fixed_cruise_speed, tendon_range, rng):
    battery_mass_kg = rng.uniform(1.5, 6.0)
    battery_spec_energy = rng.uniform(*LI_ION["spec_energy_range"])
    battery_voltage_V = rng.uniform(36.0, 60.0)
    motor_eff = rng.uniform(*CONFIG["MOTOR_EFFICIENCY_RANGE"])
    esc_eff = rng.uniform(*CONFIG["ESC_EFFICIENCY_RANGE"])

    mtow_kg = empty_mass_kg + payload_mass_kg
    hover_power_mech_W = CONFIG["HOVER_POWER_COEFF"] * (mtow_kg ** 1.5)
    hover_power_elec_W = hover_power_mech_W / max(motor_eff * esc_eff, 1e-6)
    power_margin = rng.uniform(1.1, 2.0)
    battery_max_power_W = hover_power_elec_W * power_margin

    per_rotor_peak_W = battery_max_power_W / rotor_count
    per_rotor_peak_A = per_rotor_peak_W / battery_voltage_V
    esc_current_rating_A = per_rotor_peak_A * rng.uniform(1.2, 1.8)

    cruise_speed_mps = fixed_cruise_speed if fixed_cruise_speed is not None else rng.uniform(10.0, 20.0)
    climb_rate_mps = rng.uniform(*CONFIG["CLIMB_RATE_RANGE_MPS"])
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
        battery_energy_Wh=battery_mass_kg * battery_spec_energy,
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
        design_summary="Monte Carlo sensitivity draw.",
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
draw_orderings_preserved = 0

for draw_i in range(N_DRAWS):
    draw_seed = master_rng.randint(0, 2**31 - 1)
    this_draw_scores = {}
    for (name, empty_kg, payload_kg, rotor_count, fixed_cruise, tendon_range, real_ratio, placement) in TEAMS:
        rng = random.Random(draw_seed + hash(name) % 10000)
        d = sample_design(f"DLIFT_MC_{name}_{draw_i}", empty_kg, payload_kg, rotor_count, fixed_cruise, tendon_range, rng)
        success_rate, rank_score = rank_score_for(d, N_MISSIONS_PER_DRAW, rng)
        per_team_scores[name].append(rank_score)
        per_team_success[name].append(success_rate)
        this_draw_scores[name] = rank_score

    order = sorted(this_draw_scores, key=lambda k: this_draw_scores[k], reverse=True)
    expected_order = ["DefendTex", "MacGyver", "H-Squared", "Xtreme Aerial", "MTech", "AVIDrone"]
    # DefendTex expected worst given its extreme ratio; among the 5 real winners,
    # expect MacGyver > H-Squared > Xtreme Aerial > MTech > AVIDrone (point-estimate finding)
    winners_order = [n for n in order if n != "DefendTex"]
    expected_winners_order = ["MacGyver", "H-Squared", "Xtreme Aerial", "MTech", "AVIDrone"]
    if winners_order == expected_winners_order:
        draw_orderings_preserved += 1

    if (draw_i + 1) % 10 == 0:
        print(f"draw {draw_i+1}/{N_DRAWS} done")

print(f"\n=== Monte Carlo sensitivity results ({N_DRAWS} draws x {N_MISSIONS_PER_DRAW} missions each) ===\n")
print(f"{'Team':>15} {'real_ratio':>10} {'real_place':>18} {'mean_rank':>10} {'median_rank':>12} {'stdev':>8} {'min':>7} {'max':>7}")
for (name, empty_kg, payload_kg, rotor_count, fixed_cruise, tendon_range, real_ratio, placement) in TEAMS:
    scores = per_team_scores[name]
    print(f"{name:>15} {real_ratio:10.2f} {placement:>18} {statistics.mean(scores):10.3f} {statistics.median(scores):12.3f} {statistics.stdev(scores):8.3f} {min(scores):7.3f} {max(scores):7.3f}")

print(f"\nFraction of draws where the 5-winner inverse-ratio ordering (MacGyver > H-Squared > Xtreme > MTech > AVIDrone) held exactly: {draw_orderings_preserved}/{N_DRAWS} ({100*draw_orderings_preserved/N_DRAWS:.1f}%)")
