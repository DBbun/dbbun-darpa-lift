from __future__ import annotations

# © 2025 DBbun LLC — All rights reserved.
# This file is part of the DBbun Synthetic Missions System for the DARPA Lift Challenge.
# Its use is strictly limited to internal research and development associated with
# DARPA Lift Challenge participation. Redistribution, publication, commercial use,
# or post-challenge retention is prohibited without written authorization from DBbun LLC.
#
# License: DBbun DARPA Lift Challenge R&D License v1.0 (see LICENSE.md)
# Contact: contact@dbbun.com
# Website: https://dbbun.com/
# CAGE: 16VU3 | UEI: QY39Y38E6WG8

"""
DBbun Synthetic Mission and Aircraft Generator (v1.1)
-------------------------------------------------------
This version fixes the "weird plots" issue by:
  - Recording vertical_rate_mps in the time-series
  - Recording speed_total_mps = sqrt(horizontal_speed^2 + vertical_rate^2)
  - Providing correct plotting utilities that use the per-timestep time-series
    (no groupby/drop_duplicates collapse of time)

Implements:
  1) Failure selection fix (event vs type selection separated)
  2) Time-varying environment per timestep (AR(1) wind and turbulence)
  3) Variability in cruise groundspeed and cruise altitude (correlated noise)
  4) Removes perfectly straight altitude slopes (correlated vertical-rate noise)
  5) Graceful failure termination (emergency descent)
  6) Trait coverage for all traits (including foldable wings, morphing membrane wings)
  7) Explicit propulsion electronics fields (motor, electronic speed controller, voltage)
  8) Peak power limiting and a simple thermal derating / failure model
  9) Design ranking and stars computed from mission outcomes
  10) Optional: generates plots (Speed vs Time, Phase Timeline) per design
"""

import os
import math
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd


# ============================================================
# 0. GLOBAL CONFIG (EDIT HERE)
# ============================================================

CONFIG: Dict[str, Any] = {
    "VERSION": "v1.1",

    "RNG_SEED": 42,
    "N_DESIGNS": 10,
    "MISSIONS_PER_DESIGN": 10,
    "TIME_STEP_S": 1,
    "OUTPUT_DIR": "output",

    # Plotting
    "MAKE_PLOTS": False,
    "PLOTS_DIR": "plots",

    # DARPA Lift rule constants
    "DARPA_MAX_EMPTY_MASS_KG": 24.95,
    "DARPA_MIN_PAYLOAD_MASS_KG": 49.9,
    "DARPA_MAX_MISSION_TIME_S": 30 * 60,  # 30 minutes

    # Mission legs
    "PAYLOAD_LEG_DISTANCE_NM": 4.0,
    "RETURN_LEG_DISTANCE_NM": 1.0,

    # Altitude
    "CRUISE_ALTITUDE_FT": 350.0,
    "CRUISE_ALTITUDE_TOL_FT": 50.0,

    # Payload drop altitude (near-ground)
    # Default: 2 inches = 0.0508 m
    "PAYLOAD_DROP_ALTITUDE_M": 0.0508,
    "DROP_HOVER_S": 1.0,

    # Environment ranges (initial conditions)
    "WIND_SPEED_KTS_RANGE": (0.0, 35.0),
    "TURBULENCE_INDEX_RANGE": (0.0, 1.5),

    # Time-varying environment (AR(1))
    # x[t+1] = rho*x[t] + (1-rho)*mu + sigma*N(0,1)
    "WIND_AR_RHO": 0.985,
    "WIND_AR_SIGMA": 0.35,   # knots per step
    "TURB_AR_RHO": 0.980,
    "TURB_AR_SIGMA": 0.03,   # turb units per step

    # Cruise variability
    "CRUISE_SPEED_AR_RHO": 0.97,
    "CRUISE_SPEED_SIGMA_MPS": 0.12,  # scaled by wind and turbulence
    "CRUISE_ALT_HOLD_RHO": 0.97,
    "CRUISE_ALT_HOLD_SIGMA_M": 0.20,  # scaled by wind and turbulence

    # Vertical-rate disturbance (removes perfectly straight slopes)
    "VZ_NOISE_RHO": 0.97,
    "VZ_NOISE_SIGMA_MPS": 0.15,

    # Emergency descent after failure
    "EMERGENCY_DESCENT_DURATION_S_RANGE": (12.0, 30.0),
    "EMERGENCY_DESCENT_POWER_FRACTION": 0.35,

    # Mass ranges
    "EMPTY_MASS_KG_RANGE": (12.0, 24.5),
    "PAYLOAD_MASS_KG_RANGE": (50.0, 80.0),

    # Rotor count range
    "ROTOR_COUNT_RANGE": (4, 24),

    # Thrust and power parameters
    "MAX_TWR_RANGE": (1.1, 2.5),
    "BURST_POWER_FACTOR_RANGE": (1.0, 3.0),
    "BURST_DURATION_S_RANGE": (3.0, 20.0),

    # Battery mass ranges
    "BATTERY_MASS_KG_RANGE": (3.0, 12.0),

    # Supercaps for burst
    "SUPERCAP_MASS_KG_RANGE": (0.0, 3.0),
    "SUPERCAP_SPEC_ENERGY_WH_PER_KG": 40.0,

    # Unsteady lift (bee-like)
    "UNSTEADY_LIFT_GAIN_RANGE": (0.0, 0.25),

    # Structure and stability
    "FRAME_STIFFNESS_LONGITUDINAL_RANGE": (0.3, 1.0),
    "TENDON_CABLE_FRACTION_RANGE": (0.0, 0.7),
    "GUST_REJECTION_GAIN_RANGE": (0.5, 2.0),

    # Landing gear
    "LANDING_GEAR_MASS_KG_RANGE": (0.5, 4.0),
    "MAX_TOUCHDOWN_VELOCITY_MPS_RANGE": (0.5, 2.5),

    # Mission and control
    "CRUISE_SPEED_RANGE_MPS": (5.0, 25.0),
    "CLIMB_RATE_RANGE_MPS": (2.0, 6.0),
    "MODE_COUNT_RANGE": (2, 6),

    # Simple physics coefficients
    "HOVER_POWER_COEFF": 8.0,
    "CRUISE_POWER_COEFF": 1.8,
    "WIND_POWER_PENALTY_FACTOR": 0.25,
    "TURB_POWER_PENALTY_FACTOR": 0.5,

    # How strongly wind slows cruise groundspeed
    "WIND_SPEED_TIME_SLOWDOWN_FACTOR": 0.5,

    # Failure model toggles
    "ENABLE_RANDOM_FAILURES": True,
    "BASE_RANDOM_FAILURE_RATE": 0.05,

    # Power and thermal modeling
    "BATTERY_MAX_C_RATE_RANGE": (3.0, 12.0),     # peak discharge in C (Wh-based approximation)
    "BATTERY_VOLTAGE_RANGE_V": (24.0, 96.0),     # nominal system voltage
    "MOTOR_EFFICIENCY_RANGE": (0.78, 0.92),
    "ESC_EFFICIENCY_RANGE": (0.94, 0.985),

    "THERMAL_TAU_S": 220.0,                      # cooling time constant
    "THERMAL_HEAT_GAIN": 0.0045,                 # temperature rise per (kW * s), synthetic
    "THERMAL_AMBIENT_C": 22.0,
    "THERMAL_DERATE_START_C": 70.0,
    "THERMAL_FAIL_C": 95.0,
    "THERMAL_DERATE_MIN_FRACTION": 0.35,         # never derate below this fraction

    "POWER_SATURATION_GRACE_S": 6.0,             # seconds of sustained saturation before failure

    # Wing deployment for foldable wings
    "WING_DEPLOY_TIME_S_RANGE": (2.0, 8.0),
    "WING_DEPLOY_FAILURE_BASE_P": 0.015,

    # Ranking weights (sum ~ 1.0). Missing signals are skipped and weights renormalize.
    "RANK_W_SUCCESS_RATE": 0.55,
    "RANK_W_QUAL_RATE": 0.25,
    "RANK_W_RULE_PENALTY": 0.20,

    # Output filenames
    "DESIGNS_CSV": "designs.csv",
    "MISSIONS_CSV": "missions.csv",
    "MISSIONS_TS_CSV": "missions_timeseries.csv",
}


# ============================================================
# Deterministic ID helper
# ============================================================

def make_id(prefix: str, rng: random.Random, nhex: int = 10) -> str:
    x = rng.getrandbits(4 * nhex)
    return f"{prefix}_{x:0{nhex}x}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def rand_uniform(rng: random.Random, a: float, b: float) -> float:
    return rng.uniform(a, b)


def rand_int(rng: random.Random, a: int, b: int) -> int:
    return rng.randint(a, b)


# ------------------------------------------------------------
# Energy system profiles
# ------------------------------------------------------------

ENERGY_SYSTEM_PROFILES: Dict[str, Dict[str, Any]] = {
    "li_ion": {
        "prob": 0.70,
        "spec_energy_range": (180.0, 260.0),
        "power_class": "high",
        "energy_density_class": "medium_high",
        "tech_maturity_class": "high",
        "extra_failure_risk": 0.01,
        "description": "Conventional high-TRL lithium-ion pack for VTOL drones, good power and proven reliability.",
    },
    "li_s": {
        "prob": 0.10,
        "spec_energy_range": (300.0, 450.0),
        "power_class": "medium",
        "energy_density_class": "very_high",
        "tech_maturity_class": "low",
        "extra_failure_risk": 0.05,
        "description": "Experimental lithium-sulfur chemistry with very high energy density but lower maturity and stability.",
    },
    "solid_state": {
        "prob": 0.10,
        "spec_energy_range": (260.0, 350.0),
        "power_class": "medium",
        "energy_density_class": "high",
        "tech_maturity_class": "medium_low",
        "extra_failure_risk": 0.03,
        "description": "Solid-state battery with improved safety and energy density but still under active development.",
    },
    "fuel_cell_li_ion_hybrid": {
        "prob": 0.10,
        "spec_energy_range": (400.0, 800.0),
        "power_class": "medium_low",
        "energy_density_class": "extreme",
        "tech_maturity_class": "medium_low",
        "extra_failure_risk": 0.06,
        "description": "Hybrid hydrogen fuel-cell plus lithium-ion buffer pack, high effective energy but complex and less mature.",
    },
}


def choose_energy_system_type(rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for key, prof in ENERGY_SYSTEM_PROFILES.items():
        cumulative += prof["prob"]
        if r <= cumulative:
            return key
    return "li_ion"


# ------------------------------------------------------------
# Structural / rotor / landing-gear materials
# ------------------------------------------------------------

STRUCTURAL_MATERIAL_PROFILES: Dict[str, Dict[str, Any]] = {
    "carbon_composite": {
        "prob": 0.30,
        "class": "light_stiff",
        "frame_stiffness_factor": 1.1,
        "landing_gear_mass_factor": 0.9,
        "structural_extra_failure_risk": 0.01,
    },
    "aluminum_lithium": {
        "prob": 0.25,
        "class": "medium_stiff",
        "frame_stiffness_factor": 1.0,
        "landing_gear_mass_factor": 1.0,
        "structural_extra_failure_risk": 0.02,
    },
    "titanium_alloy": {
        "prob": 0.15,
        "class": "heavy_robust",
        "frame_stiffness_factor": 1.15,
        "landing_gear_mass_factor": 1.2,
        "structural_extra_failure_risk": 0.005,
    },
    "magnesium_alloy": {
        "prob": 0.10,
        "class": "very_light_fragile",
        "frame_stiffness_factor": 0.9,
        "landing_gear_mass_factor": 0.85,
        "structural_extra_failure_risk": 0.04,
    },
    "steel_truss": {
        "prob": 0.20,
        "class": "heavy_robust",
        "frame_stiffness_factor": 1.05,
        "landing_gear_mass_factor": 1.3,
        "structural_extra_failure_risk": 0.01,
    },
}

ROTOR_BLADE_MATERIAL_OPTIONS = [
    ("carbon_composite", 0.6),
    ("glass_composite", 0.25),
    ("aluminum", 0.15),
]

LANDING_GEAR_MATERIAL_OPTIONS = [
    ("titanium", 0.35),
    ("steel", 0.35),
    ("composite", 0.30),
]


def weighted_choice(rng: random.Random, options: List[Tuple[str, float]]) -> str:
    r = rng.random()
    cumulative = 0.0
    for key, prob in options:
        cumulative += prob
        if r <= cumulative:
            return key
    return options[-1][0]


def choose_structural_material(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    r = rng.random()
    cumulative = 0.0
    for key, prof in STRUCTURAL_MATERIAL_PROFILES.items():
        cumulative += prof["prob"]
        if r <= cumulative:
            return key, prof
    key = "carbon_composite"
    return key, STRUCTURAL_MATERIAL_PROFILES[key]


# ------------------------------------------------------------
# Animals and traits
# ------------------------------------------------------------

ANIMAL_TRAITS: Dict[str, List[str]] = {
    "harpy_eagle": ["LIFT_BURST_POWER", "LIFT_HIGH_CONTINUOUS", "PAYLOAD_CENTRAL_TALON"],
    "golden_eagle": ["LIFT_BURST_POWER", "LIFT_HIGH_CONTINUOUS", "PAYLOAD_CENTRAL_TALON"],
    "osprey": ["PAYLOAD_SLING_LOAD", "PAYLOAD_DAMPED", "STAB_GUST_REJECTION"],
    "albatross": ["WING_HIGH_ASPECT", "LIFT_ENERGY_DENSE", "MISSION_GLIDE_SEGMENTS"],
    "dragonfly": ["STAB_DISTRIBUTED_THRUST", "STAB_GUST_REJECTION"],
    "bee": ["LIFT_UNSTEADY_AERO", "STAB_DISTRIBUTED_THRUST"],
    "bat": ["WING_MORPHING_MEMBRANE", "WING_FOLDABLE"],
    "cheetah": ["LIFT_BURST_POWER", "STRUCT_TENDON_CABLES", "STAB_COMPLIANT_SPINE", "MISSION_MULTI_GAIT"],
    "tiger": ["LIFT_BURST_POWER", "STRUCT_TENDON_CABLES", "STRUCT_ROBUST_GEAR"],
}

ALL_ANIMALS = list(ANIMAL_TRAITS.keys())

TRAIT_PHRASES: Dict[str, str] = {
    "LIFT_BURST_POWER": "short burst thrust for takeoff and climb via high-power propulsion or supercapacitors",
    "LIFT_HIGH_CONTINUOUS": "strong continuous lifting capacity for heavy payloads",
    "PAYLOAD_CENTRAL_TALON": "a central payload cradle aligned with the center of gravity",
    "PAYLOAD_SLING_LOAD": "a sling-load style payload mount hanging beneath the airframe",
    "PAYLOAD_DAMPED": "damped payload mounting to reduce oscillations during flight",
    "WING_HIGH_ASPECT": "long, high-aspect-ratio lifting surfaces for efficient cruise",
    "LIFT_ENERGY_DENSE": "an emphasis on energy-dense storage for long-range flight",
    "MISSION_GLIDE_SEGMENTS": "mission planning that includes partial-power or glide segments to save energy",
    "STAB_DISTRIBUTED_THRUST": "many smaller rotors for fine-grained control and redundancy",
    "STAB_GUST_REJECTION": "enhanced gust rejection and stability control for windy conditions",
    "LIFT_UNSTEADY_AERO": "unsteady aerodynamics for better hover and low-speed lift",
    "WING_MORPHING_MEMBRANE": "morphing lifting surface that can change camber and area for control and efficiency",
    "WING_FOLDABLE": "foldable wings that deploy after vertical takeoff for cruise efficiency",
    "STRUCT_TENDON_CABLES": "a tendon or cable-based structural system to save mass in tension members",
    "STAB_COMPLIANT_SPINE": "a compliant spine-like frame that absorbs dynamic loads",
    "MISSION_MULTI_GAIT": "multiple flight modes to optimize heavy-load and unloaded segments",
    "STRUCT_ROBUST_GEAR": "reinforced landing gear designed to tolerate higher touchdown loads",
}


# ------------------------------------------------------------
# Propulsion architectures (rotors + jets + pushers)
# ------------------------------------------------------------

PROPULSION_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "pure_rotor_electric": {
        "prob": 0.55,
        "primary_propulsor_type": "multirotor",
        "secondary_propulsor_type": "none",
        "secondary_propulsor_fraction_range": (0.0, 0.0),
        "hover_power_factor": 1.0,
        "cruise_power_factor": 1.0,
        "extra_failure_risk": 0.0,
    },
    "rotor_plus_pusher": {
        "prob": 0.20,
        "primary_propulsor_type": "multirotor",
        "secondary_propulsor_type": "pusher_propeller",
        "secondary_propulsor_fraction_range": (0.2, 0.6),
        "hover_power_factor": 1.05,
        "cruise_power_factor": 0.9,
        "extra_failure_risk": 0.01,
    },
    "rotor_plus_small_jet": {
        "prob": 0.15,
        "primary_propulsor_type": "multirotor",
        "secondary_propulsor_type": "small_jet",
        "secondary_propulsor_fraction_range": (0.1, 0.4),
        "hover_power_factor": 1.15,
        "cruise_power_factor": 0.85,
        "extra_failure_risk": 0.03,
    },
    "ducted_fans_hybrid": {
        "prob": 0.10,
        "primary_propulsor_type": "ducted_fan_array",
        "secondary_propulsor_type": "none",
        "secondary_propulsor_fraction_range": (0.0, 0.0),
        "hover_power_factor": 0.95,
        "cruise_power_factor": 1.05,
        "extra_failure_risk": 0.015,
    },
}


def choose_propulsion_architecture(rng: random.Random) -> Tuple[str, Dict[str, Any]]:
    r = rng.random()
    cumulative = 0.0
    for key, prof in PROPULSION_ARCHITECTURES.items():
        cumulative += prof["prob"]
        if r <= cumulative:
            return key, prof
    key = "pure_rotor_electric"
    return key, PROPULSION_ARCHITECTURES[key]


# ============================================================
# 1. DATA CLASSES
# ============================================================

@dataclass
class AircraftDesign:
    design_id: str
    animals: List[str]
    traits: List[str]

    animal_count: int
    trait_count: int

    empty_mass_kg: float
    payload_mass_kg: float

    rotor_count: int
    max_twr: float
    burst_power_factor: float
    burst_duration_s: float
    unsteady_lift_gain: float

    energy_system_type: str
    energy_system_description: str
    energy_density_class: str
    power_class: str
    tech_maturity_class: str
    energy_system_extra_failure_risk: float

    battery_mass_kg: float
    battery_spec_energy_Wh_per_kg: float
    battery_energy_Wh: float
    battery_nominal_voltage_V: float
    battery_max_power_W: float

    supercap_mass_kg: float
    supercap_energy_Wh: float
    supercap_max_power_W: float

    motor_type: str
    motor_efficiency: float
    esc_efficiency: float
    esc_current_rating_A: float

    structural_material: str
    structural_material_class: str
    structural_extra_failure_risk: float
    rotor_blade_material: str
    landing_gear_material: str

    frame_stiffness_longitudinal: float
    tendon_cable_fraction: float
    gust_rejection_gain: float
    landing_gear_mass_kg: float
    max_touchdown_velocity_mps: float

    cruise_speed_mps: float
    climb_rate_mps: float
    mode_count: int

    mtow_kg: float
    payload_to_aircraft_ratio: float

    rule_empty_mass_ok: bool
    rule_payload_ok: bool

    design_qualifying: bool
    design_qualifying_score: float

    design_summary: str

    propulsion_architecture: str
    primary_propulsor_type: str
    secondary_propulsor_type: str
    secondary_propulsor_fraction: float
    propulsion_hover_power_factor: float
    propulsion_cruise_power_factor: float
    propulsion_arch_extra_failure_risk: float

    wing_foldable: bool
    wing_deploy_time_s: float
    wing_deploy_failure_risk: float

    image_prompt: str

    # Ranking populated after simulation
    design_success_rate: float = 0.0
    design_qualifying_rate: float = 0.0
    design_rule_penalty_rate: float = 0.0
    design_rank_score: float = 0.0
    design_stars: int = 0


@dataclass
class Environment:
    wind_speed_kts: float
    turbulence_index: float


@dataclass
class MissionResult:
    design_id: str
    mission_id: str
    wind_speed_kts: float
    turbulence_index: float

    success: bool
    failure_phase: str
    failure_reason: str
    rule_violation: str

    total_time_s: float

    time_takeoff_climb_s: float
    time_loaded_cruise_s: float
    time_drop_descent_s: float
    time_drop_hover_s: float
    time_post_drop_climb_s: float
    time_empty_cruise_s: float
    time_descent_landing_s: float
    time_wing_deploy_s: float

    energy_used_Wh: float
    battery_energy_remaining_Wh: float

    payload_to_aircraft_ratio: float
    is_qualifying_run: bool
    qualifying_score: float

    max_power_requested_W: float
    max_power_available_W: float
    power_saturation_seconds: float
    thermal_peak_C: float


@dataclass
class TimeStep:
    design_id: str
    mission_id: str
    t_s: float
    phase: str
    altitude_m: float
    distance_m: float

    # Horizontal speed component (cruise groundspeed proxy)
    speed_mps: float

    # NEW: vertical rate and total speed magnitude
    vertical_rate_mps: float
    speed_total_mps: float

    mass_kg: float

    power_requested_W: float
    power_available_W: float
    power_used_W: float

    energy_used_Wh_cum: float
    battery_remaining_Wh: float

    payload_attached: bool
    wind_speed_kts: float
    turbulence_index: float
    temperature_C: float


# ============================================================
# 2. RANDOM HELPERS
# ============================================================

def choose_animals(rng: random.Random) -> List[str]:
    n = rand_int(rng, 1, 5)
    return rng.sample(list(ALL_ANIMALS), k=n)


def accumulate_traits(animals: List[str]) -> List[str]:
    traits = set()
    for a in animals:
        for t in ANIMAL_TRAITS.get(a, []):
            traits.add(t)
    return sorted(traits)


# ============================================================
# 3. DESIGN DESCRIPTION & VISUAL PROMPT
# ============================================================

def describe_design(design: AircraftDesign) -> str:
    if design.animals:
        animal_text = ", ".join(design.animals)
        intro = f"Bio-inspired by {animal_text}, this design is a heavy-lift VTOL rotorcraft concept. "
    else:
        intro = "This design is a heavy-lift VTOL rotorcraft concept. "

    ratio_str = f"{design.payload_to_aircraft_ratio:.2f}:1"
    config_bits = (
        f"It uses {design.rotor_count} rotors, has an empty mass of {design.empty_mass_kg:.1f} kg, "
        f"and carries a payload of {design.payload_mass_kg:.1f} kg ({ratio_str} payload-to-aircraft mass ratio). "
    )

    energy_bits = (
        f"The energy system is {design.energy_system_type.replace('_', '-')} with a {design.battery_mass_kg:.1f} kg battery "
        f"(~{design.battery_energy_Wh:.0f} Wh usable) at about {design.battery_nominal_voltage_V:.0f} V. "
        f"Estimated battery peak power is ~{design.battery_max_power_W/1000.0:.1f} kW. "
    )
    if design.supercap_mass_kg > 0:
        energy_bits += (
            f"It also includes ~{design.supercap_mass_kg:.1f} kg of supercapacitors "
            f"(~{design.supercap_energy_Wh:.0f} Wh) that can provide high peak power (up to ~{design.supercap_max_power_W/1000.0:.1f} kW). "
        )

    prop_bits = (
        f"The primary propulsion architecture is {design.propulsion_architecture.replace('_', ' ')}, "
        f"using {design.motor_type.replace('_', ' ')} motors with a nominal electronic speed controller rating of "
        f"~{design.esc_current_rating_A:.0f} A. "
    )

    material_bits = (
        f"The main frame uses {design.structural_material.replace('_', ' ')} ({design.structural_material_class}), "
        f"with {design.rotor_blade_material.replace('_', ' ')} rotor blades and {design.landing_gear_material} landing gear. "
    )

    trait_phrases = []
    for t in design.traits:
        phrase = TRAIT_PHRASES.get(t)
        if phrase:
            trait_phrases.append(phrase)
    trait_phrases = trait_phrases[:6]
    traits_text = ("Key bio-inspired features include " + "; ".join(trait_phrases) + ". ") if trait_phrases else ""

    wing_bits = ""
    if design.wing_foldable:
        wing_bits = (
            f"It uses foldable wings that deploy after takeoff. The wing deployment time is ~{design.wing_deploy_time_s:.1f} s "
            f"with a synthetic deployment failure risk of ~{100.0*design.wing_deploy_failure_risk:.1f}%. "
        )

    mission_bits = (
        f"In the synthetic mission model, it cruises at about {design.cruise_speed_mps:.1f} m/s and climbs at about {design.climb_rate_mps:.1f} m/s "
        f"to around {CONFIG['CRUISE_ALTITUDE_FT']:.0f} ft. "
    )

    if design.design_qualifying:
        align_bits = (
            "This design satisfies the basic mass and payload constraints and is treated as a qualifying candidate with a nominal score of "
            f"{design.design_qualifying_score:.2f}. "
        )
    else:
        align_bits = (
            "This design does not satisfy the baseline mass and payload thresholds and is treated as a non-qualifying concept in the synthetic universe. "
        )

    return intro + config_bits + energy_bits + prop_bits + material_bits + traits_text + wing_bits + mission_bits + align_bits


def build_image_prompt(design: AircraftDesign) -> str:
    animals = ", ".join(design.animals) if design.animals else "multiple flying animals"
    rotor_phrase = f"{design.rotor_count} evenly spaced rotors"
    struct_mat = design.structural_material.replace("_", " ")
    rotor_mat = design.rotor_blade_material.replace("_", " ")
    gear_mat = design.landing_gear_material.replace("_", " ")
    energy_type = design.energy_system_type.replace("_", "-")

    style_bits = (
        "high-resolution 3D-rendered digital image, realistic lighting, clear details, no motion blur, "
        "clean white or lightly blurred test-range background, no people, no logos, no text or numbers"
    )

    prompt = (
        f"Futuristic heavy-lift VTOL rotorcraft drone bio-inspired by {animals}, "
        f"with {rotor_phrase} and a compact central fuselage. "
        f"Sleek {struct_mat} airframe, {rotor_mat} rotor blades, and rugged {gear_mat} landing gear. "
        f"Subtle external cues of a {energy_type} energy system and integrated high-power modules, "
        f"designed to carry a heavy payload under the belly. "
        f"Aircraft shown in mid-flight at medium altitude, slightly angled to show top and side. "
        f"{style_bits}."
    )
    return prompt


# ============================================================
# 4. DESIGN GENERATION
# ============================================================

def _motor_type_for_arch(arch_key: str, primary_type: str, secondary_type: str) -> str:
    if primary_type == "ducted_fan_array":
        return "electric_ducted_fan"
    if secondary_type == "small_jet":
        return "electric_multirotor_plus_micro_turbojet"
    if secondary_type == "pusher_propeller":
        return "electric_multirotor_plus_pusher"
    return "electric_brushless_outunner"


def _power_class_factor(power_class: str) -> float:
    if power_class == "high":
        return 1.0
    if power_class == "medium":
        return 0.8
    if power_class == "medium_low":
        return 0.65
    return 0.75


def generate_design(rng: random.Random) -> AircraftDesign:
    cfg = CONFIG

    animals = choose_animals(rng)
    traits = accumulate_traits(animals)
    animal_count = len(animals)
    trait_count = len(traits)

    empty_mass_kg = rand_uniform(rng, *cfg["EMPTY_MASS_KG_RANGE"])
    payload_mass_kg = rand_uniform(rng, *cfg["PAYLOAD_MASS_KG_RANGE"])

    rotor_min, rotor_max = cfg["ROTOR_COUNT_RANGE"]
    if "STAB_DISTRIBUTED_THRUST" in traits:
        rotor_count = rand_int(rng, max(rotor_min, 8), rotor_max)
    else:
        rotor_count = rand_int(rng, rotor_min, min(rotor_max, 12))

    max_twr = rand_uniform(rng, *cfg["MAX_TWR_RANGE"])

    if "LIFT_BURST_POWER" in traits:
        burst_power_factor = rand_uniform(rng, *cfg["BURST_POWER_FACTOR_RANGE"])
        burst_duration_s = rand_uniform(rng, *cfg["BURST_DURATION_S_RANGE"])
    else:
        burst_power_factor = 1.0
        burst_duration_s = 0.0

    unsteady_lift_gain = rand_uniform(rng, *cfg["UNSTEADY_LIFT_GAIN_RANGE"]) if "LIFT_UNSTEADY_AERO" in traits else 0.0

    energy_type = choose_energy_system_type(rng)
    prof = ENERGY_SYSTEM_PROFILES[energy_type]

    battery_mass_kg = rand_uniform(rng, *cfg["BATTERY_MASS_KG_RANGE"])
    spec_min, spec_max = prof["spec_energy_range"]
    if "LIFT_ENERGY_DENSE" in traits:
        battery_spec_energy = rand_uniform(rng, (spec_min + spec_max) / 2.0, spec_max)
    else:
        battery_spec_energy = rand_uniform(rng, spec_min, spec_max)
    battery_energy_Wh = battery_mass_kg * battery_spec_energy

    battery_nominal_voltage_V = rand_uniform(rng, *cfg["BATTERY_VOLTAGE_RANGE_V"])
    c_rate = rand_uniform(rng, *cfg["BATTERY_MAX_C_RATE_RANGE"])
    c_rate *= _power_class_factor(prof["power_class"])
    battery_max_power_W = battery_energy_Wh * c_rate

    if "LIFT_BURST_POWER" in traits or energy_type == "fuel_cell_li_ion_hybrid":
        supercap_mass_kg = rand_uniform(rng, *cfg["SUPERCAP_MASS_KG_RANGE"])
    else:
        supercap_mass_kg = 0.0
    supercap_energy_Wh = supercap_mass_kg * cfg["SUPERCAP_SPEC_ENERGY_WH_PER_KG"]
    supercap_max_power_W = supercap_energy_Wh * 40.0  # synthetic

    structural_material, mat_prof = choose_structural_material(rng)
    structural_material_class = mat_prof["class"]
    structural_extra_failure_risk = mat_prof["structural_extra_failure_risk"]
    rotor_blade_material = weighted_choice(rng, ROTOR_BLADE_MATERIAL_OPTIONS)
    landing_gear_material = weighted_choice(rng, LANDING_GEAR_MATERIAL_OPTIONS)

    frame_stiffness = rand_uniform(rng, *cfg["FRAME_STIFFNESS_LONGITUDINAL_RANGE"])
    frame_stiffness *= mat_prof["frame_stiffness_factor"]
    if "STAB_COMPLIANT_SPINE" in traits:
        frame_stiffness = min(frame_stiffness, 0.7)

    tendon_cable_fraction = rand_uniform(rng, *cfg["TENDON_CABLE_FRACTION_RANGE"]) if "STRUCT_TENDON_CABLES" in traits else 0.0

    gust_gain = rand_uniform(rng, *cfg["GUST_REJECTION_GAIN_RANGE"])
    if "STAB_GUST_REJECTION" in traits:
        gust_gain = max(gust_gain, 1.2)

    landing_gear_mass_kg = rand_uniform(rng, *cfg["LANDING_GEAR_MASS_KG_RANGE"])
    landing_gear_mass_kg *= mat_prof["landing_gear_mass_factor"]
    max_touchdown_velocity_mps = rand_uniform(rng, *cfg["MAX_TOUCHDOWN_VELOCITY_MPS_RANGE"])
    if "STRUCT_ROBUST_GEAR" in traits:
        landing_gear_mass_kg *= 1.2
        max_touchdown_velocity_mps *= 1.5

    cruise_speed_mps = rand_uniform(rng, *cfg["CRUISE_SPEED_RANGE_MPS"])
    climb_rate_mps = rand_uniform(rng, *cfg["CLIMB_RATE_RANGE_MPS"])

    mode_count = rand_int(rng, *cfg["MODE_COUNT_RANGE"])
    if "MISSION_MULTI_GAIT" in traits:
        mode_count = max(mode_count, 4)

    mtow_kg = empty_mass_kg + payload_mass_kg
    payload_to_aircraft_ratio = payload_mass_kg / max(empty_mass_kg, 1e-6)

    rule_empty_mass_ok = empty_mass_kg <= cfg["DARPA_MAX_EMPTY_MASS_KG"]
    rule_payload_ok = payload_mass_kg >= cfg["DARPA_MIN_PAYLOAD_MASS_KG"]

    design_qualifying = rule_empty_mass_ok and rule_payload_ok
    design_qualifying_score = payload_to_aircraft_ratio if design_qualifying else 0.0

    arch_key, arch_prof = choose_propulsion_architecture(rng)
    secondary_fraction_range = arch_prof["secondary_propulsor_fraction_range"]
    if secondary_fraction_range[0] == secondary_fraction_range[1]:
        secondary_fraction = secondary_fraction_range[0]
    else:
        secondary_fraction = rand_uniform(rng, *secondary_fraction_range)

    motor_type = _motor_type_for_arch(arch_key, arch_prof["primary_propulsor_type"], arch_prof["secondary_propulsor_type"])
    motor_eff = rand_uniform(rng, *cfg["MOTOR_EFFICIENCY_RANGE"])
    esc_eff = rand_uniform(rng, *cfg["ESC_EFFICIENCY_RANGE"])

    primary_fraction = 1.0 - secondary_fraction
    total_peak_elec_W = max(battery_max_power_W + supercap_max_power_W, 1.0)
    per_rotor_peak_W = (total_peak_elec_W * primary_fraction) / max(rotor_count, 1)
    per_rotor_peak_A = per_rotor_peak_W / max(battery_nominal_voltage_V, 1.0)
    esc_current_rating_A = per_rotor_peak_A * rand_uniform(rng, 1.2, 1.8)

    wing_foldable = "WING_FOLDABLE" in traits
    if wing_foldable:
        wing_deploy_time_s = rand_uniform(rng, *cfg["WING_DEPLOY_TIME_S_RANGE"])
        base_p = cfg["WING_DEPLOY_FAILURE_BASE_P"]
        complexity = 0.10 * (rotor_count / 24.0) + 0.10 * clamp((mtow_kg - 60.0) / 40.0, 0.0, 1.0)
        wing_deploy_failure_risk = clamp(base_p + complexity, 0.0, 0.12)
    else:
        wing_deploy_time_s = 0.0
        wing_deploy_failure_risk = 0.0

    design_id = make_id("DLIFT", rng, 10)

    tmp_design = AircraftDesign(
        design_id=design_id,
        animals=animals,
        traits=traits,

        animal_count=animal_count,
        trait_count=trait_count,

        empty_mass_kg=empty_mass_kg,
        payload_mass_kg=payload_mass_kg,

        rotor_count=rotor_count,
        max_twr=max_twr,
        burst_power_factor=burst_power_factor,
        burst_duration_s=burst_duration_s,
        unsteady_lift_gain=unsteady_lift_gain,

        energy_system_type=energy_type,
        energy_system_description=prof["description"],
        energy_density_class=prof["energy_density_class"],
        power_class=prof["power_class"],
        tech_maturity_class=prof["tech_maturity_class"],
        energy_system_extra_failure_risk=prof["extra_failure_risk"],

        battery_mass_kg=battery_mass_kg,
        battery_spec_energy_Wh_per_kg=battery_spec_energy,
        battery_energy_Wh=battery_energy_Wh,
        battery_nominal_voltage_V=battery_nominal_voltage_V,
        battery_max_power_W=battery_max_power_W,

        supercap_mass_kg=supercap_mass_kg,
        supercap_energy_Wh=supercap_energy_Wh,
        supercap_max_power_W=supercap_max_power_W,

        motor_type=motor_type,
        motor_efficiency=motor_eff,
        esc_efficiency=esc_eff,
        esc_current_rating_A=esc_current_rating_A,

        structural_material=structural_material,
        structural_material_class=structural_material_class,
        structural_extra_failure_risk=structural_extra_failure_risk,
        rotor_blade_material=rotor_blade_material,
        landing_gear_material=landing_gear_material,

        frame_stiffness_longitudinal=frame_stiffness,
        tendon_cable_fraction=tendon_cable_fraction,
        gust_rejection_gain=gust_gain,
        landing_gear_mass_kg=landing_gear_mass_kg,
        max_touchdown_velocity_mps=max_touchdown_velocity_mps,

        cruise_speed_mps=cruise_speed_mps,
        climb_rate_mps=climb_rate_mps,
        mode_count=mode_count,

        mtow_kg=mtow_kg,
        payload_to_aircraft_ratio=payload_to_aircraft_ratio,

        rule_empty_mass_ok=rule_empty_mass_ok,
        rule_payload_ok=rule_payload_ok,

        design_qualifying=design_qualifying,
        design_qualifying_score=design_qualifying_score,

        design_summary="",

        propulsion_architecture=arch_key,
        primary_propulsor_type=arch_prof["primary_propulsor_type"],
        secondary_propulsor_type=arch_prof["secondary_propulsor_type"],
        secondary_propulsor_fraction=secondary_fraction,
        propulsion_hover_power_factor=arch_prof["hover_power_factor"],
        propulsion_cruise_power_factor=arch_prof["cruise_power_factor"],
        propulsion_arch_extra_failure_risk=arch_prof["extra_failure_risk"],

        wing_foldable=wing_foldable,
        wing_deploy_time_s=wing_deploy_time_s,
        wing_deploy_failure_risk=wing_deploy_failure_risk,

        image_prompt="",
    )

    tmp_design.design_summary = describe_design(tmp_design)
    tmp_design.image_prompt = build_image_prompt(tmp_design)
    return tmp_design


# ============================================================
# 5. MISSION PHYSICS & RULES
# ============================================================

def generate_environment(rng: random.Random) -> Environment:
    cfg = CONFIG
    wind_speed_kts = rand_uniform(rng, *cfg["WIND_SPEED_KTS_RANGE"])
    turbulence_index = rand_uniform(rng, *cfg["TURBULENCE_INDEX_RANGE"])
    return Environment(wind_speed_kts=wind_speed_kts, turbulence_index=turbulence_index)


def simulate_mission(design: AircraftDesign, env: Environment, rng: random.Random) -> Tuple[MissionResult, List[TimeStep]]:
    cfg = CONFIG
    dt = float(cfg["TIME_STEP_S"])
    ts_rows: List[TimeStep] = []

    if not (design.rule_empty_mass_ok and design.rule_payload_ok):
        mission = MissionResult(
            design_id=design.design_id,
            mission_id=make_id("MSN", rng, 10),
            wind_speed_kts=env.wind_speed_kts,
            turbulence_index=env.turbulence_index,
            success=False,
            failure_phase="precheck",
            failure_reason="mass_or_payload_rule_violation",
            rule_violation="mass_or_payload",
            total_time_s=0.0,
            time_takeoff_climb_s=0.0,
            time_loaded_cruise_s=0.0,
            time_drop_descent_s=0.0,
            time_drop_hover_s=0.0,
            time_post_drop_climb_s=0.0,
            time_empty_cruise_s=0.0,
            time_descent_landing_s=0.0,
            time_wing_deploy_s=0.0,
            energy_used_Wh=0.0,
            battery_energy_remaining_Wh=design.battery_energy_Wh,
            payload_to_aircraft_ratio=design.payload_to_aircraft_ratio,
            is_qualifying_run=False,
            qualifying_score=0.0,
            max_power_requested_W=0.0,
            max_power_available_W=0.0,
            power_saturation_seconds=0.0,
            thermal_peak_C=cfg["THERMAL_AMBIENT_C"],
        )
        return mission, ts_rows

    def nm_to_m(nm: float) -> float:
        return nm * 1852.0

    # ---- Power model helpers -------------------------------------------

    def hover_power_required_W(mass_kg: float) -> float:
        base = cfg["HOVER_POWER_COEFF"] * (mass_kg ** 1.5)
        reduction = base * design.unsteady_lift_gain * 0.5
        power = max(base - reduction, 0.0)
        if "STAB_DISTRIBUTED_THRUST" in design.traits:
            power *= 1.03
        if design.wing_foldable:
            power *= 0.985
        return power * design.propulsion_hover_power_factor

    def cruise_power_required_W(mass_kg: float, speed_mps: float, wind_frac: float, turb_frac: float, wing_deployed: bool) -> float:
        base = cfg["CRUISE_POWER_COEFF"] * mass_kg * (speed_mps / 10.0)

        if "WING_HIGH_ASPECT" in design.traits:
            base *= 0.92

        if "MISSION_GLIDE_SEGMENTS" in design.traits:
            base *= 0.95

        if design.wing_foldable and wing_deployed:
            base *= 0.94

        if "WING_MORPHING_MEMBRANE" in design.traits:
            eff_gain = 1.0 - 0.05 * clamp(0.6 * wind_frac + 0.8 * turb_frac, 0.0, 1.5)
            base *= clamp(eff_gain, 0.90, 1.0)

        return base * design.propulsion_cruise_power_factor

    def apply_env_penalties(power_W: float, wind_frac: float, turb_frac: float) -> float:
        factor = 1.0 + cfg["WIND_POWER_PENALTY_FACTOR"] * wind_frac + cfg["TURB_POWER_PENALTY_FACTOR"] * turb_frac
        if "STAB_GUST_REJECTION" in design.traits:
            factor *= 0.97
        return power_W * factor

    def available_power_W(temperature_C: float, supercap_energy_Wh: float) -> float:
        p_batt = design.battery_max_power_W
        if temperature_C >= cfg["THERMAL_DERATE_START_C"]:
            span = max(cfg["THERMAL_FAIL_C"] - cfg["THERMAL_DERATE_START_C"], 1e-6)
            x = clamp((temperature_C - cfg["THERMAL_DERATE_START_C"]) / span, 0.0, 1.0)
            derate = 1.0 - x * (1.0 - cfg["THERMAL_DERATE_MIN_FRACTION"])
            p_batt *= derate

        p_sc = design.supercap_max_power_W if supercap_energy_Wh > 0.0 else 0.0
        return max(p_batt + p_sc, 0.0)

    # ---- Dynamic environment states (AR(1)) ----------------------------

    wind_mu = clamp(env.wind_speed_kts, *cfg["WIND_SPEED_KTS_RANGE"])
    turb_mu = clamp(env.turbulence_index, *cfg["TURBULENCE_INDEX_RANGE"])
    wind_t = wind_mu
    turb_t = turb_mu

    cruise_speed_noise = 0.0
    alt_hold_noise = 0.0
    vz_noise = 0.0

    altitude_target_m = cfg["CRUISE_ALTITUDE_FT"] * 0.3048
    dist_loaded_total_m = nm_to_m(cfg["PAYLOAD_LEG_DISTANCE_NM"])
    dist_empty_total_m = nm_to_m(cfg["RETURN_LEG_DISTANCE_NM"])
    drop_alt_m = max(0.0, min(float(cfg.get("PAYLOAD_DROP_ALTITUDE_M", 0.0508)), altitude_target_m))
    drop_hover_s_target = max(0.0, float(cfg.get("DROP_HOVER_S", 1.0)))

    commanded_cruise_speed = design.cruise_speed_mps
    commanded_climb_rate = design.climb_rate_mps

    # ---- Failure pre-sampling ------------------------------------------

    max_wind = max(cfg["WIND_SPEED_KTS_RANGE"][1], 1e-6)
    wind_frac0 = wind_mu / max_wind
    slowdown0 = clamp(cfg["WIND_SPEED_TIME_SLOWDOWN_FACTOR"] * wind_frac0, 0.0, 0.7)
    effective0 = max(commanded_cruise_speed * (1.0 - slowdown0), commanded_cruise_speed * 0.3)

    time_takeoff_climb_nom = altitude_target_m / max(commanded_climb_rate, 0.5)
    time_loaded_cruise_nom = dist_loaded_total_m / max(effective0, 0.5)
    time_drop_descent_nom = (altitude_target_m - drop_alt_m) / max(commanded_climb_rate, 0.5)
    time_drop_hover_nom = drop_hover_s_target
    time_post_drop_climb_nom = (altitude_target_m - drop_alt_m) / max(commanded_climb_rate, 0.5)
    time_empty_cruise_nom = dist_empty_total_m / max(effective0, 0.5)
    time_descent_landing_nom = altitude_target_m / max(commanded_climb_rate, 0.5)

    t0 = 0.0
    t1 = t0 + time_takeoff_climb_nom
    t2 = t1 + time_loaded_cruise_nom
    t3 = t2 + time_drop_descent_nom
    t4 = t3 + time_drop_hover_nom
    t5 = t4 + time_post_drop_climb_nom
    t6 = t5 + time_empty_cruise_nom
    t7 = t6 + time_descent_landing_nom

    random_failure_time: Optional[float] = None
    random_failure_phase = ""
    random_failure_reason = ""
    random_failure_rule = ""

    if cfg["ENABLE_RANDOM_FAILURES"]:
        load_ratio = design.payload_to_aircraft_ratio
        load_term = max(load_ratio - 2.0, 0.0) / 3.0
        turb_frac0 = min(turb_mu / max(cfg["TURBULENCE_INDEX_RANGE"][1], 1e-6), 1.5)

        stress_index = 0.3 * wind_frac0 + 0.4 * turb_frac0 + 0.3 * load_term
        stress_index = clamp(stress_index, 0.0, 2.0)

        base_p = cfg["BASE_RANDOM_FAILURE_RATE"]

        p_gust = base_p * (0.5 + stress_index)
        if "STAB_GUST_REJECTION" in design.traits:
            p_gust *= 0.75

        p_control = base_p * (0.3 + 0.7 * turb_frac0)
        if "STAB_DISTRIBUTED_THRUST" in design.traits:
            p_control *= 0.80

        touchdown_ratio = design.climb_rate_mps / max(design.max_touchdown_velocity_mps, 0.1)
        p_touchdown = base_p * max(touchdown_ratio - 1.0, 0.0)
        if "STRUCT_ROBUST_GEAR" in design.traits:
            p_touchdown *= 0.70

        p_energy_fault = base_p * 0.5 + design.energy_system_extra_failure_risk
        p_structural = base_p * 0.4 + design.structural_extra_failure_risk
        p_prop_arch = base_p * 0.4 + design.propulsion_arch_extra_failure_risk

        if "PAYLOAD_SLING_LOAD" in design.traits:
            p_gust *= 1.10
            p_structural *= 1.05
        if "PAYLOAD_DAMPED" in design.traits:
            p_gust *= 0.92
            p_control *= 0.92
        if "PAYLOAD_CENTRAL_TALON" in design.traits:
            p_gust *= 0.95
            p_structural *= 0.95

        if design.wing_foldable:
            p_control *= 1.03

        total_p = min(p_gust + p_control + p_touchdown + p_energy_fault + p_structural + p_prop_arch, 0.9)

        if rng.random() < total_p:
            weights = [
                ("gust_induced_instability", "loaded_cruise", "", p_gust),
                ("control_saturation", "empty_cruise", "", p_control),
                ("hard_touchdown", "descent_landing", "landing_profile_violation", p_touchdown),
                ("energy_system_fault", "mission", "energy_architecture_risk", p_energy_fault),
                ("structural_overload", "loaded_cruise", "structural_margin_risk", p_structural),
                ("propulsion_architecture_fault", "mission", "propulsion_architecture_risk", p_prop_arch),
            ]
            wsum = sum(w for *_, w in weights)
            u_type = rng.random() * max(wsum, 1e-12)
            cum = 0.0
            chosen = weights[-1]
            for item in weights:
                cum += item[3]
                if u_type <= cum:
                    chosen = item
                    break

            random_failure_reason, random_failure_phase, random_failure_rule, _w = chosen

            if random_failure_phase == "loaded_cruise":
                phase_start, phase_end = t1, t2
            elif random_failure_phase == "empty_cruise":
                phase_start, phase_end = t5, t6
            elif random_failure_phase == "descent_landing":
                phase_start, phase_end = t6, t7
            else:
                phase_start, phase_end = t0, t7

            random_failure_time = rand_uniform(rng, phase_start, max(phase_start + 1.0, phase_end))

    # ---- Mission state --------------------------------------------------

    mission_id = make_id("MSN", rng, 10)

    altitude_m = 0.0
    distance_m = 0.0
    t = 0.0
    energy_cum_Wh = 0.0

    battery_energy_Wh = design.battery_energy_Wh
    supercap_energy_Wh = design.supercap_energy_Wh

    temperature_C = float(cfg["THERMAL_AMBIENT_C"])
    thermal_peak_C = temperature_C

    success = True
    failure_phase = ""
    failure_reason = ""
    rule_violation = ""

    emergency_active = False
    emergency_descent_rate = 0.0
    emergency_remaining_s = 0.0

    payload_attached = True
    phase = "takeoff_climb"
    hover_drop_elapsed = 0.0

    seg_start_distance_m = 0.0

    wing_deployed = not design.wing_foldable
    wing_deploy_remaining_s = design.wing_deploy_time_s if design.wing_foldable else 0.0
    time_wing_deploy_actual = 0.0

    time_takeoff_climb_actual = 0.0
    time_loaded_cruise_actual = 0.0
    time_drop_descent_actual = 0.0
    time_drop_hover_actual = 0.0
    time_post_drop_climb_actual = 0.0
    time_empty_cruise_actual = 0.0
    time_descent_landing_actual = 0.0

    max_power_req = 0.0
    max_power_avail = 0.0
    sat_seconds = 0.0

    # ---- Dynamics helpers ----------------------------------------------

    def update_environment() -> Tuple[float, float, float, float]:
        nonlocal wind_t, turb_t
        wind_t = cfg["WIND_AR_RHO"] * wind_t + (1.0 - cfg["WIND_AR_RHO"]) * wind_mu + cfg["WIND_AR_SIGMA"] * rng.gauss(0.0, 1.0)
        turb_t = cfg["TURB_AR_RHO"] * turb_t + (1.0 - cfg["TURB_AR_RHO"]) * turb_mu + cfg["TURB_AR_SIGMA"] * rng.gauss(0.0, 1.0)
        wind_t = clamp(wind_t, *cfg["WIND_SPEED_KTS_RANGE"])
        turb_t = clamp(turb_t, *cfg["TURBULENCE_INDEX_RANGE"])

        max_w = max(cfg["WIND_SPEED_KTS_RANGE"][1], 1e-6)
        wind_frac = clamp(wind_t / max_w, 0.0, 1.5)
        turb_frac = clamp(turb_t / max(cfg["TURBULENCE_INDEX_RANGE"][1], 1e-6), 0.0, 1.5)
        return wind_t, turb_t, wind_frac, turb_frac

    def update_vz_noise() -> float:
        nonlocal vz_noise
        vz_noise = cfg["VZ_NOISE_RHO"] * vz_noise + cfg["VZ_NOISE_SIGMA_MPS"] * rng.gauss(0.0, 1.0)
        return vz_noise

    def cruise_groundspeed(commanded: float, wind_frac: float, turb_frac: float) -> float:
        nonlocal cruise_speed_noise
        base_sigma = cfg["CRUISE_SPEED_SIGMA_MPS"]
        scale = 1.0 + 1.2 * wind_frac + 1.0 * turb_frac
        sigma = base_sigma * scale
        cruise_speed_noise = cfg["CRUISE_SPEED_AR_RHO"] * cruise_speed_noise + sigma * rng.gauss(0.0, 1.0)

        slowdown = clamp(cfg["WIND_SPEED_TIME_SLOWDOWN_FACTOR"] * wind_frac, 0.0, 0.7)
        mean = commanded * (1.0 - slowdown)

        v = mean + cruise_speed_noise
        if "WING_MORPHING_MEMBRANE" in design.traits:
            v = mean + 0.75 * (v - mean)

        return max(v, commanded * 0.25)

    def cruise_altitude_hold(target: float, wind_frac: float, turb_frac: float) -> float:
        nonlocal alt_hold_noise
        base_sigma = cfg["CRUISE_ALT_HOLD_SIGMA_M"]
        scale = 1.0 + 1.3 * wind_frac + 1.5 * turb_frac
        scale *= 1.0 / max(design.gust_rejection_gain, 0.5)
        if "WING_MORPHING_MEMBRANE" in design.traits:
            scale *= 0.80

        sigma = base_sigma * scale
        alt_hold_noise = cfg["CRUISE_ALT_HOLD_RHO"] * alt_hold_noise + sigma * rng.gauss(0.0, 1.0)
        return target + alt_hold_noise

    def step_thermal(power_used_W: float) -> None:
        nonlocal temperature_C, thermal_peak_C
        ambient = float(cfg["THERMAL_AMBIENT_C"])
        tau = float(cfg["THERMAL_TAU_S"])
        heat_gain = float(cfg["THERMAL_HEAT_GAIN"])
        temperature_C += heat_gain * (power_used_W / 1000.0) * dt
        temperature_C += (ambient - temperature_C) * (dt / max(tau, 1e-6))
        thermal_peak_C = max(thermal_peak_C, temperature_C)

    def record_step(
        phase_name: str,
        mass_kg: float,
        speed_mps: float,
        vertical_rate_mps: float,
        p_req_W: float,
        p_av_W: float,
        p_used_W: float,
        wind_kts: float,
        turb_idx: float,
    ) -> None:
        speed_total = math.sqrt(max(speed_mps, 0.0) ** 2 + (vertical_rate_mps ** 2))
        ts_rows.append(TimeStep(
            design_id=design.design_id,
            mission_id=mission_id,
            t_s=t,
            phase=phase_name,
            altitude_m=altitude_m,
            distance_m=distance_m,
            speed_mps=speed_mps,
            vertical_rate_mps=vertical_rate_mps,
            speed_total_mps=speed_total,
            mass_kg=mass_kg,
            power_requested_W=p_req_W,
            power_available_W=p_av_W,
            power_used_W=p_used_W,
            energy_used_Wh_cum=energy_cum_Wh,
            battery_remaining_Wh=battery_energy_Wh,
            payload_attached=payload_attached,
            wind_speed_kts=wind_kts,
            turbulence_index=turb_idx,
            temperature_C=temperature_C,
        ))

    # ---- Main loop ------------------------------------------------------

    while True:
        wind_kts, turb_idx, wind_frac, turb_frac = update_environment()

        if (t > cfg["DARPA_MAX_MISSION_TIME_S"]) and success:
            success = False
            failure_phase = "mission"
            failure_reason = "time_limit_exceeded"
            rule_violation = "time_limit"
            emergency_active = True

        if (random_failure_time is not None) and (t >= random_failure_time) and success:
            success = False
            failure_phase = random_failure_phase
            failure_reason = random_failure_reason
            rule_violation = random_failure_rule
            emergency_active = True

        if (temperature_C >= cfg["THERMAL_FAIL_C"]) and success:
            success = False
            failure_phase = phase
            failure_reason = "thermal_runaway"
            rule_violation = "thermal_limit"
            emergency_active = True

        if (battery_energy_Wh < 0.0) and success:
            success = False
            failure_phase = phase
            failure_reason = "energy_depleted"
            rule_violation = "energy_budget"
            emergency_active = True

        if emergency_active and emergency_remaining_s <= 0.0:
            dur_lo, dur_hi = cfg["EMERGENCY_DESCENT_DURATION_S_RANGE"]
            emergency_remaining_s = rand_uniform(rng, dur_lo, dur_hi)
            emergency_descent_rate = max(altitude_m / max(emergency_remaining_s, 1.0), 0.5)

        if design.wing_foldable and (not wing_deployed) and (phase in ("loaded_cruise", "empty_cruise")) and (not emergency_active):
            time_wing_deploy_actual += dt
            wing_deploy_remaining_s = max(0.0, wing_deploy_remaining_s - dt)

            p_fail_step = design.wing_deploy_failure_risk * (0.5 + 0.7 * turb_frac) * (dt / max(design.wing_deploy_time_s, 1e-6))
            if rng.random() < p_fail_step and success:
                success = False
                failure_phase = phase
                failure_reason = "wing_deploy_failure"
                rule_violation = "wing_deploy"
                emergency_active = True

            if wing_deploy_remaining_s <= 0.0:
                wing_deployed = True

        # Kinematics and power
        speed_mps = 0.0
        vertical_rate = 0.0

        mass_kg = design.empty_mass_kg + (design.payload_mass_kg if payload_attached else 0.0)

        if emergency_active:
            phase_name = "emergency_descent"
            speed_mps = max(0.2, cruise_groundspeed(commanded_cruise_speed, wind_frac, turb_frac) * 0.15)
            vertical_rate = -emergency_descent_rate + update_vz_noise()

            p_base = hover_power_required_W(mass_kg) * 0.4 + (cfg["CRUISE_POWER_COEFF"] * mass_kg * (speed_mps / 10.0)) * 0.2
            p_req = apply_env_penalties(p_base, wind_frac, turb_frac) * cfg["EMERGENCY_DESCENT_POWER_FRACTION"]
        else:
            phase_name = phase

            if phase == "takeoff_climb":
                vertical_rate = commanded_climb_rate + update_vz_noise()
                speed_mps = 0.0
                p_req = apply_env_penalties(hover_power_required_W(mass_kg), wind_frac, turb_frac)
                if design.burst_power_factor > 1.0 and time_takeoff_climb_actual < design.burst_duration_s:
                    p_req *= design.burst_power_factor

            elif phase == "loaded_cruise":
                speed_mps = cruise_groundspeed(commanded_cruise_speed, wind_frac, turb_frac)
                vertical_rate = 0.0
                p_req = apply_env_penalties(cruise_power_required_W(mass_kg, speed_mps, wind_frac, turb_frac, wing_deployed), wind_frac, turb_frac)

            elif phase == "drop_descent":
                vertical_rate = -abs(commanded_climb_rate) + update_vz_noise()
                speed_mps = 0.0
                p_req = apply_env_penalties(hover_power_required_W(mass_kg) * 0.6, wind_frac, turb_frac)

            elif phase == "drop_hover":
                vertical_rate = 0.0
                speed_mps = 0.0
                p_req = apply_env_penalties(hover_power_required_W(mass_kg) * 0.55, wind_frac, turb_frac)

            elif phase == "post_drop_climb":
                vertical_rate = commanded_climb_rate + update_vz_noise()
                speed_mps = 0.0
                p_req = apply_env_penalties(hover_power_required_W(mass_kg), wind_frac, turb_frac)

            elif phase == "empty_cruise":
                speed_mps = cruise_groundspeed(commanded_cruise_speed, wind_frac, turb_frac)
                vertical_rate = 0.0
                p_req = apply_env_penalties(cruise_power_required_W(mass_kg, speed_mps, wind_frac, turb_frac, wing_deployed), wind_frac, turb_frac)

            elif phase == "descent_landing":
                vertical_rate = -abs(commanded_climb_rate) + update_vz_noise()
                speed_mps = 0.0
                p_req = apply_env_penalties(hover_power_required_W(mass_kg) * 0.6, wind_frac, turb_frac)
            else:
                vertical_rate = 0.0
                speed_mps = 0.0
                p_req = 0.0

        # Cruise altitude hold
        if phase_name in ("loaded_cruise", "empty_cruise") and not emergency_active:
            desired_alt = cruise_altitude_hold(altitude_target_m, wind_frac, turb_frac)
            tau = 6.0
            altitude_m += (desired_alt - altitude_m) * (dt / max(tau, 1e-6))
        else:
            altitude_m += vertical_rate * dt

        altitude_m = max(0.0, altitude_m)

        distance_m += speed_mps * dt

        p_av = available_power_W(temperature_C, supercap_energy_Wh)
        p_used = min(p_req, p_av)

        max_power_req = max(max_power_req, p_req)
        max_power_avail = max(max_power_avail, p_av)

        if (p_req > p_av) and (not emergency_active):
            sat_seconds += dt
            if sat_seconds >= cfg["POWER_SATURATION_GRACE_S"] and success:
                success = False
                failure_phase = phase
                failure_reason = "power_saturation"
                rule_violation = "power_limit"
                emergency_active = True
        else:
            sat_seconds = max(0.0, sat_seconds - 0.5 * dt)

        # Energy accounting
        batt_limit = design.battery_max_power_W
        if temperature_C >= cfg["THERMAL_DERATE_START_C"]:
            span = max(cfg["THERMAL_FAIL_C"] - cfg["THERMAL_DERATE_START_C"], 1e-6)
            x = clamp((temperature_C - cfg["THERMAL_DERATE_START_C"]) / span, 0.0, 1.0)
            derate = 1.0 - x * (1.0 - cfg["THERMAL_DERATE_MIN_FRACTION"])
            batt_limit *= derate

        p_batt_used = min(p_used, batt_limit)
        p_sc_used = max(0.0, p_used - p_batt_used)

        if (phase == "takeoff_climb") and (not emergency_active) and (design.supercap_energy_Wh > 0.0) and (design.burst_power_factor > 1.0):
            shift_fraction = 0.35
            p_shift = min(p_batt_used * shift_fraction, design.supercap_max_power_W)
            p_batt_used -= p_shift
            p_sc_used += p_shift

        batt_Wh = p_batt_used * dt / 3600.0
        sc_Wh = p_sc_used * dt / 3600.0

        if supercap_energy_Wh <= 0.0:
            sc_Wh = 0.0
        else:
            if sc_Wh > supercap_energy_Wh:
                overflow = sc_Wh - supercap_energy_Wh
                sc_Wh = supercap_energy_Wh
                batt_Wh += overflow

        supercap_energy_Wh -= sc_Wh
        battery_energy_Wh -= batt_Wh
        energy_cum_Wh += (batt_Wh + sc_Wh)

        elec_losses = p_used * (1.0 - clamp(design.motor_efficiency, 0.5, 0.99)) + p_used * (1.0 - clamp(design.esc_efficiency, 0.5, 0.99))
        elec_losses *= (1.0 + 0.12 * turb_frac)
        step_thermal(elec_losses)

        # Record
        record_step(phase_name, mass_kg, speed_mps, vertical_rate, p_req, p_av, p_used, wind_kts, turb_idx)

        # Advance time
        t += dt

        # Phase time accounting
        if phase_name == "takeoff_climb":
            time_takeoff_climb_actual += dt
        elif phase_name == "loaded_cruise":
            time_loaded_cruise_actual += dt
        elif phase_name == "drop_descent":
            time_drop_descent_actual += dt
        elif phase_name == "drop_hover":
            time_drop_hover_actual += dt
        elif phase_name == "post_drop_climb":
            time_post_drop_climb_actual += dt
        elif phase_name == "empty_cruise":
            time_empty_cruise_actual += dt
        elif phase_name in ("descent_landing", "emergency_descent"):
            time_descent_landing_actual += dt
            if phase_name == "emergency_descent":
                emergency_remaining_s = max(0.0, emergency_remaining_s - dt)

        # Emergency termination
        if emergency_active:
            if altitude_m <= 0.0 or emergency_remaining_s <= 0.0:
                altitude_m = 0.0
                break
            continue

        # Mission phase transitions
        if phase == "takeoff_climb":
            if altitude_m >= altitude_target_m:
                altitude_m = altitude_target_m
                phase = "loaded_cruise"
                seg_start_distance_m = distance_m

        elif phase == "loaded_cruise":
            leg_progress = distance_m - seg_start_distance_m
            if leg_progress >= dist_loaded_total_m:
                phase = "drop_descent"

        elif phase == "drop_descent":
            if altitude_m <= drop_alt_m:
                altitude_m = drop_alt_m
                phase = "drop_hover"
                hover_drop_elapsed = 0.0

        elif phase == "drop_hover":
            hover_drop_elapsed += dt
            if hover_drop_elapsed >= drop_hover_s_target:
                payload_attached = False
                phase = "post_drop_climb"

        elif phase == "post_drop_climb":
            if altitude_m >= altitude_target_m:
                altitude_m = altitude_target_m
                phase = "empty_cruise"
                seg_start_distance_m = distance_m

        elif phase == "empty_cruise":
            leg_progress = distance_m - seg_start_distance_m
            if leg_progress >= dist_empty_total_m:
                phase = "descent_landing"

        elif phase == "descent_landing":
            if altitude_m <= 0.0:
                altitude_m = 0.0
                break

    total_time_s = t
    energy_used_Wh = energy_cum_Wh
    battery_remaining = battery_energy_Wh

    if success:
        failure_phase = ""
        failure_reason = ""
        rule_violation = ""

    is_qualifying_run = success and design.design_qualifying
    qualifying_score = design.design_qualifying_score if is_qualifying_run else 0.0

    mission = MissionResult(
        design_id=design.design_id,
        mission_id=mission_id,
        wind_speed_kts=env.wind_speed_kts,
        turbulence_index=env.turbulence_index,

        success=success,
        failure_phase=failure_phase,
        failure_reason=failure_reason,
        rule_violation=rule_violation,

        total_time_s=total_time_s,

        time_takeoff_climb_s=time_takeoff_climb_actual,
        time_loaded_cruise_s=time_loaded_cruise_actual,
        time_drop_descent_s=time_drop_descent_actual,
        time_drop_hover_s=time_drop_hover_actual,
        time_post_drop_climb_s=time_post_drop_climb_actual,
        time_empty_cruise_s=time_empty_cruise_actual,
        time_descent_landing_s=time_descent_landing_actual,
        time_wing_deploy_s=time_wing_deploy_actual,

        energy_used_Wh=energy_used_Wh,
        battery_energy_remaining_Wh=battery_remaining,

        payload_to_aircraft_ratio=design.payload_to_aircraft_ratio,
        is_qualifying_run=is_qualifying_run,
        qualifying_score=qualifying_score,

        max_power_requested_W=max_power_req,
        max_power_available_W=max_power_avail,
        power_saturation_seconds=sat_seconds,
        thermal_peak_C=thermal_peak_C,
    )

    return mission, ts_rows


# ============================================================
# 6. RANKING AND EXPORT
# ============================================================

def compute_design_rankings(designs: List[AircraftDesign], missions: List[MissionResult]) -> None:
    cfg = CONFIG

    w_success = float(cfg["RANK_W_SUCCESS_RATE"])
    w_qual = float(cfg["RANK_W_QUAL_RATE"])
    w_pen = float(cfg["RANK_W_RULE_PENALTY"])

    by_design: Dict[str, List[MissionResult]] = {}
    for m in missions:
        by_design.setdefault(m.design_id, []).append(m)

    scores: List[Tuple[str, float]] = []
    for d in designs:
        ms = by_design.get(d.design_id, [])
        n = len(ms)
        if n == 0:
            d.design_success_rate = 0.0
            d.design_qualifying_rate = 0.0
            d.design_rule_penalty_rate = 0.0
            d.design_rank_score = 0.0
            scores.append((d.design_id, 0.0))
            continue

        success_rate = sum(1 for m in ms if m.success) / n
        qualifying_rate = sum(1 for m in ms if m.is_qualifying_run) / n
        rule_penalty_rate = sum(1 for m in ms if (m.rule_violation is not None and str(m.rule_violation).strip() != "")) / n

        d.design_success_rate = success_rate
        d.design_qualifying_rate = qualifying_rate
        d.design_rule_penalty_rate = rule_penalty_rate

        parts: List[Tuple[float, float]] = [
            (success_rate, w_success),
            (qualifying_rate, w_qual),
            (1.0 - rule_penalty_rate, w_pen),
        ]

        wsum = sum(w for _, w in parts if w > 0.0)
        score = sum(val * w for val, w in parts) / wsum if wsum > 0.0 else 0.0

        d.design_rank_score = score
        scores.append((d.design_id, score))

    sorted_scores = sorted([s for _, s in scores])
    if not sorted_scores:
        return

    def percentile_threshold(p: float) -> float:
        idx = int(round(p * (len(sorted_scores) - 1)))
        idx = max(0, min(idx, len(sorted_scores) - 1))
        return sorted_scores[idx]

    t10 = percentile_threshold(0.10)
    t30 = percentile_threshold(0.30)
    t70 = percentile_threshold(0.70)
    t90 = percentile_threshold(0.90)

    for d in designs:
        s = d.design_rank_score
        if s >= t90:
            d.design_stars = 5
        elif s >= t70:
            d.design_stars = 4
        elif s >= t30:
            d.design_stars = 3
        elif s >= t10:
            d.design_stars = 2
        else:
            d.design_stars = 1


def designs_to_dataframe(designs: List[AircraftDesign]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for d in designs:
        row = asdict(d)
        row["animals"] = ",".join(d.animals)
        row["traits"] = ",".join(d.traits)
        rows.append(row)
    return pd.DataFrame(rows)


def missions_to_dataframe(missions: List[MissionResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(m) for m in missions])


def timeseries_to_dataframe(ts: List[TimeStep]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in ts])


def generate_universe(
    n_designs: int,
    missions_per_design: int,
    rng: random.Random,
) -> Tuple[List[AircraftDesign], List[MissionResult], List[TimeStep]]:
    designs: List[AircraftDesign] = []
    missions: List[MissionResult] = []
    ts_rows: List[TimeStep] = []

    for _ in range(n_designs):
        d = generate_design(rng)
        designs.append(d)

        for _ in range(missions_per_design):
            env = generate_environment(rng)
            mission, ts = simulate_mission(d, env, rng)
            missions.append(mission)
            ts_rows.extend(ts)

    compute_design_rankings(designs, missions)
    return designs, missions, ts_rows


# ============================================================
# 8. PLOTTING (OPTIONAL)
# ============================================================

def _safe_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_speed_vs_time(df_ts: pd.DataFrame, df_missions: pd.DataFrame, design_id: str, out_path: str) -> None:
    plt = _safe_import_matplotlib()

    dfts = df_ts[df_ts["design_id"] == design_id].copy()
    if dfts.empty:
        return
    dfts = dfts.sort_values(["mission_id", "t_s"])

    dfm = df_missions[df_missions["design_id"] == design_id][["mission_id", "success"]].copy()
    success_by_mid = {r["mission_id"]: bool(r["success"]) for _, r in dfm.iterrows()}

    plt.figure(figsize=(16, 9))
    for mid, g in dfts.groupby("mission_id"):
        g = g.sort_values("t_s")
        y = g["speed_total_mps"] if "speed_total_mps" in g.columns else g["speed_mps"]
        plt.plot(g["t_s"], y, linewidth=1.6)

        ok = success_by_mid.get(mid, True)
        if len(g) > 0:
            x_end = float(g["t_s"].iloc[-1])
            y_end = float(y.iloc[-1])
            plt.text(x_end, y_end, "✓" if ok else "FAIL", fontsize=12)

    plt.title(f"Speed vs Time — {design_id}")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_phase_timeline(df_ts: pd.DataFrame, df_missions: pd.DataFrame, design_id: str, out_path: str) -> None:
    plt = _safe_import_matplotlib()

    dfts = df_ts[df_ts["design_id"] == design_id].copy()
    if dfts.empty:
        return
    dfts = dfts.sort_values(["mission_id", "t_s"])

    dfm = df_missions[df_missions["design_id"] == design_id][["mission_id", "success"]].copy()
    success_by_mid = {r["mission_id"]: bool(r["success"]) for _, r in dfm.iterrows()}

    phase_order = [
        "takeoff_climb",
        "loaded_cruise",
        "drop_descent",
        "drop_hover",
        "post_drop_climb",
        "empty_cruise",
        "descent_landing",
        "emergency_descent",
    ]
    # Stable color selection via matplotlib cycle (no hard-coded colors)
    phase_to_color_idx = {p: i for i, p in enumerate(phase_order)}

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(1, 1, 1)

    missions = list(dfts["mission_id"].unique())
    bar_h = 0.8
    y_gap = 0.35

    # Build segments for each mission: contiguous runs of same phase
    for i, mid in enumerate(missions):
        g = dfts[dfts["mission_id"] == mid].sort_values("t_s")
        if g.empty:
            continue

        y = i * (bar_h + y_gap)

        # contiguous runs
        t_vals = g["t_s"].to_list()
        ph_vals = g["phase"].to_list()

        start_t = t_vals[0]
        cur_ph = ph_vals[0]

        def draw_seg(seg_start: float, seg_end: float, ph: str) -> None:
            dur = max(0.0, seg_end - seg_start + float(CONFIG["TIME_STEP_S"]))
            color = f"C{phase_to_color_idx.get(ph, 0) % 10}"
            ax.broken_barh([(seg_start, dur)], (y, bar_h), facecolors=color, edgecolors="none", alpha=0.9)

        for j in range(1, len(t_vals)):
            if ph_vals[j] != cur_ph:
                draw_seg(start_t, t_vals[j - 1], cur_ph)
                start_t = t_vals[j]
                cur_ph = ph_vals[j]

        draw_seg(start_t, t_vals[-1], cur_ph)

        ok = success_by_mid.get(mid, True)
        ax.text(t_vals[-1], y + bar_h * 0.5, "✓" if ok else "FAIL", va="center", fontsize=12)

    ax.set_title(f"Phase Timeline — {design_id} (all missions)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mission (stacked)")
    ax.grid(True, axis="x", alpha=0.25)

    # Legend
    handles = []
    labels = []
    for ph in phase_order:
        color = f"C{phase_to_color_idx.get(ph, 0) % 10}"
        h = ax.barh([-10], [0], color=color, alpha=0.9)  # dummy
        handles.append(h[0])
        labels.append(ph)
    ax.legend(handles, labels, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def generate_plots(df_designs: pd.DataFrame, df_missions: pd.DataFrame, df_ts: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for design_id in df_designs["design_id"].unique():
        speed_path = os.path.join(out_dir, f"speed_{design_id}.png")
        phase_path = os.path.join(out_dir, f"phase_{design_id}.png")
        plot_speed_vs_time(df_ts, df_missions, design_id, speed_path)
        plot_phase_timeline(df_ts, df_missions, design_id, phase_path)


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    seed = CONFIG["RNG_SEED"]
    rng = random.Random(seed if seed is not None else None)

    out_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    designs, missions, ts_rows = generate_universe(CONFIG["N_DESIGNS"], CONFIG["MISSIONS_PER_DESIGN"], rng)

    df_designs = designs_to_dataframe(designs)
    df_missions = missions_to_dataframe(missions)
    df_ts = timeseries_to_dataframe(ts_rows)

    designs_path = os.path.join(out_dir, CONFIG["DESIGNS_CSV"])
    missions_path = os.path.join(out_dir, CONFIG["MISSIONS_CSV"])
    ts_path = os.path.join(out_dir, CONFIG["MISSIONS_TS_CSV"])

    df_designs.to_csv(designs_path, index=False)
    df_missions.to_csv(missions_path, index=False)
    df_ts.to_csv(ts_path, index=False)

    print(f"[{CONFIG['VERSION']}] Generated {len(designs)} designs -> {designs_path}")
    print(f"[{CONFIG['VERSION']}] Generated {len(missions)} missions -> {missions_path}")
    print(f"[{CONFIG['VERSION']}] Generated {len(ts_rows)} time-series rows -> {ts_path}")

    if bool(CONFIG.get("MAKE_PLOTS", True)):
        plots_dir = os.path.join(out_dir, str(CONFIG.get("PLOTS_DIR", "plots")))
        generate_plots(df_designs, df_missions, df_ts, plots_dir)
        print(f"[{CONFIG['VERSION']}] Plots saved under -> {plots_dir}")

