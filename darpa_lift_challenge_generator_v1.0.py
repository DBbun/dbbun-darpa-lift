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
DBbun Synthetic Mission & Aircraft Generator
-------------------------------------------
Developed for engineering teams participating in the DARPA Lift Challenge.

Purpose
-------
To generate large-scale, fully synthetic aircraft architectures and
corresponding mission telemetry, supporting design exploration, modeling,
and optimization of heavy-lift VTOL systems.

Key Features
------------
• Bio-inspired designs:
    Translating traits from high-performance animals into mechanical
    attributes affecting lift, structural strength, and control authority.

• Multiple energy systems:
    - li_ion
    - li_s
    - solid_state
    - fuel_cell_li_ion_hybrid
    Supports varied endurance, mass distribution, burst-power strategies.

• Propulsion architectures:
    Combinations of rotors, jets, and auxiliary pusher systems enable
    trade-off analysis of thrust density vs. efficiency.

• Full time-series mission simulation:
    - Phase-based flight planning (takeoff, climb, cruise, landing)
    - Physics-informed models for mass, power, and drag
    - Environmental disturbances (wind & turbulence)

• Stochastic risk models:
    - Rule-based mission failures enforced at the moment of violation
    - Random failures sampled pre-flight and triggered during integration

• Operational constraints:
    - 30-minute DARPA mission duration limit enforced
    - Wind reduces effective cruise groundspeed, affecting success rate
    - Slow but efficient designs may exceed time limits under adverse weather

Output
------
Each generated design can produce simulated missions, exported as:
- design parameters
- mission summary statistics
- second-by-second telemetry timeseries
"""

import os
import random
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import pandas as pd

# ============================================================
# 0. GLOBAL CONFIG (EDIT HERE)
# ============================================================

CONFIG = {
    "RNG_SEED": 42,
    "N_DESIGNS": 100,
    "MISSIONS_PER_DESIGN": 10,
    "TIME_STEP_S": 1,
    "OUTPUT_DIR": "output",

    # DARPA Lift rule constants
    "DARPA_MAX_EMPTY_MASS_KG": 24.95,
    "DARPA_MIN_PAYLOAD_MASS_KG": 49.9,
    "DARPA_MAX_MISSION_TIME_S": 30 * 60,  # 30 minutes
    "PAYLOAD_LEG_DISTANCE_NM": 4.0,
    "RETURN_LEG_DISTANCE_NM": 1.0,
    "CRUISE_ALTITUDE_FT": 350.0,
    "CRUISE_ALTITUDE_TOL_FT": 50.0,

    # Environment ranges
    "WIND_SPEED_KTS_RANGE": (0.0, 35.0),
    "TURBULENCE_INDEX_RANGE": (0.0, 1.5),

    # Mass ranges
    "EMPTY_MASS_KG_RANGE": (12.0, 24.5),
    "PAYLOAD_MASS_KG_RANGE": (50.0, 80.0),

    # Rotor count range
    "ROTOR_COUNT_RANGE": (4, 24),

    # Thrust & power parameters
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

    # Structure / stability
    "FRAME_STIFFNESS_LONGITUDINAL_RANGE": (0.3, 1.0),
    "TENDON_CABLE_FRACTION_RANGE": (0.0, 0.7),
    "GUST_REJECTION_GAIN_RANGE": (0.5, 2.0),

    # Landing gear
    "LANDING_GEAR_MASS_KG_RANGE": (0.5, 4.0),
    "MAX_TOUCHDOWN_VELOCITY_MPS_RANGE": (0.5, 2.5),

    # Mission / control
    # Allow slower designs so that some missions can approach/exceed 30 minutes
    "CRUISE_SPEED_RANGE_MPS": (5.0, 25.0),
    "CLIMB_RATE_RANGE_MPS": (2.0, 6.0),
    "MODE_COUNT_RANGE": (2, 6),

    # Simple physics fudge factors
    "HOVER_POWER_COEFF": 8.0,
    "CRUISE_POWER_COEFF": 1.8,
    "WIND_POWER_PENALTY_FACTOR": 0.25,
    "TURB_POWER_PENALTY_FACTOR": 0.5,

    # How strongly wind slows cruise groundspeed (0=no effect, 0.5=up to 50% slowdown)
    "WIND_SPEED_TIME_SLOWDOWN_FACTOR": 0.5,

    # Failure model toggles
    "ENABLE_RANDOM_FAILURES": True,
    "BASE_RANDOM_FAILURE_RATE": 0.05,

    # Output filenames
    "DESIGNS_CSV": "designs.csv",
    "MISSIONS_CSV": "missions.csv",
    "MISSIONS_TS_CSV": "missions_timeseries.csv",
}

# ------------------------------------------------------------
# Energy system profiles
# ------------------------------------------------------------

ENERGY_SYSTEM_PROFILES = {
    "li_ion": {
        "prob": 0.70,
        "spec_energy_range": (180.0, 260.0),
        "power_class": "high",
        "energy_density_class": "medium_high",
        "tech_maturity_class": "high",
        "extra_failure_risk": 0.01,
        "description": "Conventional high-TRL Li-ion pack for VTOL drones, good power and proven reliability.",
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
        "description": "Hybrid hydrogen fuel-cell plus Li-ion buffer pack, high effective energy but complex and less mature.",
    },
}


def choose_energy_system_type() -> str:
    r = random.random()
    cumulative = 0.0
    for key, prof in ENERGY_SYSTEM_PROFILES.items():
        cumulative += prof["prob"]
        if r <= cumulative:
            return key
    return "li_ion"


# ------------------------------------------------------------
# Structural / rotor / landing-gear materials
# ------------------------------------------------------------

STRUCTURAL_MATERIAL_PROFILES = {
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


def weighted_choice(options: List[Tuple[str, float]]) -> str:
    r = random.random()
    cumulative = 0.0
    for key, prob in options:
        cumulative += prob
        if r <= cumulative:
            return key
    return options[-1][0]


def choose_structural_material() -> Tuple[str, Dict[str, Any]]:
    r = random.random()
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
    "PAYLOAD_CENTRAL_TALON": "a central, talon-like payload cradle aligned with the center of gravity",
    "PAYLOAD_SLING_LOAD": "a sling-load style payload mount hanging beneath the airframe",
    "PAYLOAD_DAMPED": "damped payload mounting to reduce oscillations during flight",
    "WING_HIGH_ASPECT": "long, high-aspect-ratio lifting surfaces for efficient cruise",
    "LIFT_ENERGY_DENSE": "an emphasis on energy-dense storage for long-range flight",
    "MISSION_GLIDE_SEGMENTS": "mission planning that includes partial-power or glide segments to save energy",
    "STAB_DISTRIBUTED_THRUST": "many smaller rotors for fine-grained control and redundancy",
    "STAB_GUST_REJECTION": "enhanced gust rejection and stability control for windy conditions",
    "LIFT_UNSTEADY_AERO": "unsteady aerodynamics (insect-like vortex lift) for better hover and low-speed lift",
    "WING_MORPHING_MEMBRANE": "morphing membrane-like lifting surfaces that can change area and camber",
    "WING_FOLDABLE": "foldable wings that deploy after vertical takeoff for cruise efficiency",
    "STRUCT_TENDON_CABLES": "a tendon/cable-based structural system to save mass in tension members",
    "STAB_COMPLIANT_SPINE": "a compliant spine-like frame that absorbs dynamic loads",
    "MISSION_MULTI_GAIT": "multiple flight modes (hover, heavy-load cruise, light-load return, etc.)",
    "STRUCT_ROBUST_GEAR": "reinforced landing gear designed to tolerate higher touchdown loads",
}

# ------------------------------------------------------------
# Propulsion architectures (rotors + jets + pushers)
# ------------------------------------------------------------

PROPULSION_ARCHITECTURES = {
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


def choose_propulsion_architecture() -> Tuple[str, Dict[str, Any]]:
    r = random.random()
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
    supercap_mass_kg: float
    supercap_energy_Wh: float

    # Materials
    structural_material: str
    structural_material_class: str
    structural_extra_failure_risk: float
    rotor_blade_material: str
    landing_gear_material: str

    # Structure & stability
    frame_stiffness_longitudinal: float
    tendon_cable_fraction: float
    gust_rejection_gain: float
    landing_gear_mass_kg: float
    max_touchdown_velocity_mps: float

    # Mission / control
    cruise_speed_mps: float
    climb_rate_mps: float
    mode_count: int

    # Derived metrics
    mtow_kg: float
    payload_to_aircraft_ratio: float

    rule_empty_mass_ok: bool
    rule_payload_ok: bool

    design_qualifying: bool
    design_qualifying_score: float

    design_summary: str

    # Propulsion architecture (rotors + jets + pushers)
    propulsion_architecture: str
    primary_propulsor_type: str
    secondary_propulsor_type: str
    secondary_propulsor_fraction: float
    propulsion_hover_power_factor: float
    propulsion_cruise_power_factor: float
    propulsion_arch_extra_failure_risk: float

    # DALL-E / image prompt (optional, can be ignored by downstream tools)
    image_prompt: str


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
    time_empty_cruise_s: float
    time_descent_landing_s: float
    energy_used_Wh: float
    battery_energy_remaining_Wh: float
    payload_to_aircraft_ratio: float
    is_qualifying_run: bool
    qualifying_score: float


@dataclass
class TimeStep:
    design_id: str
    mission_id: str
    t_s: float
    phase: str
    altitude_m: float
    distance_m: float
    speed_mps: float
    mass_kg: float
    power_W: float
    energy_used_Wh_cum: float
    battery_remaining_Wh: float
    payload_attached: bool
    wind_speed_kts: float
    turbulence_index: float


# ============================================================
# 2. RANDOM HELPERS
# ============================================================

def rand_uniform(a: float, b: float) -> float:
    return random.uniform(a, b)


def rand_int(a: int, b: int) -> int:
    return random.randint(a, b)


def choose_animals() -> List[str]:
    n = rand_int(1, 5)
    return random.sample(list(ALL_ANIMALS), k=n)


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

    ratio = design.payload_to_aircraft_ratio
    ratio_str = f"{ratio:.2f}:1"
    config_bits = (
        f"It uses {design.rotor_count} rotors, has an empty mass of {design.empty_mass_kg:.1f} kg, "
        f"and is configured to carry a payload of {design.payload_mass_kg:.1f} kg "
        f"({ratio_str} payload-to-aircraft mass ratio). "
    )

    energy_bits = (
        f"The primary energy system is {design.energy_system_type.replace('_', '-')} "
        f"with a battery pack of {design.battery_mass_kg:.1f} kg "
        f"(~{design.battery_energy_Wh:.0f} Wh usable). "
    )
    if design.supercap_mass_kg > 0:
        energy_bits += (
            f"It also incorporates approximately {design.supercap_mass_kg:.1f} kg of high-power supercapacitors "
            f"to support short burst power for takeoff and climb. "
        )

    material_bits = (
        f"The main load-bearing frame uses {design.structural_material.replace('_', ' ')} "
        f"({design.structural_material_class}), with {design.rotor_blade_material} rotor blades and "
        f"{design.landing_gear_material} landing gear. "
    )

    trait_phrases = []
    for t in design.traits:
        phrase = TRAIT_PHRASES.get(t)
        if phrase:
            trait_phrases.append(phrase)
    max_phrases = 5
    trait_phrases = trait_phrases[:max_phrases]
    if trait_phrases:
        traits_text = "Key bio-inspired features include " + "; ".join(trait_phrases) + ". "
    else:
        traits_text = ""

    mission_bits = (
        f"In the synthetic DARPA Lift mission model, it cruises at about {design.cruise_speed_mps:.1f} m/s "
        f"and climbs at roughly {design.climb_rate_mps:.1f} m/s to around {CONFIG['CRUISE_ALTITUDE_FT']:.0f} ft. "
    )

    if design.design_qualifying:
        align_bits = (
            "This design satisfies the basic DARPA Lift mass and payload constraints and is treated as a "
            f"qualifying candidate with a nominal score of {design.design_qualifying_score:.2f}. "
        )
    else:
        align_bits = (
            "This design does not fully satisfy the baseline DARPA Lift empty-mass and payload thresholds and is "
            "treated as a non-qualifying concept in the synthetic universe. "
        )

    return intro + config_bits + energy_bits + material_bits + traits_text + mission_bits + align_bits


def build_image_prompt(design: AircraftDesign, summary: str) -> str:
    """
    Construct a compact, DALL-E–ready visual prompt.

    Downstream consumers can ignore this if they do not need images.
    """
    animals = ", ".join(design.animals) if design.animals else "multiple flying animals"
    rotor_phrase = f"{design.rotor_count} evenly spaced rotors"
    struct_mat = design.structural_material.replace("_", " ")
    rotor_mat = design.rotor_blade_material.replace("_", " ")
    gear_mat = design.landing_gear_material.replace("_", " ")
    energy_type = design.energy_system_type.replace("_", "-")

    style_bits = (
        "high-resolution 3D-rendered digital image, "
        "realistic lighting, clear details, no motion blur, "
        "clean white or lightly blurred test-range background, "
        "no people, no logos, no text or numbers"
    )

    prompt = (
        f"Futuristic heavy-lift VTOL rotorcraft drone bio-inspired by {animals}, "
        f"with {rotor_phrase} and a compact central fuselage. "
        f"Sleek {struct_mat} airframe, {rotor_mat} rotor blades, "
        f"and rugged {gear_mat} landing gear. "
        f"Subtle external cues of a {energy_type} energy system and integrated high-power modules, "
        f"designed to carry a heavy payload under the belly. "
        f"Aircraft shown in mid-flight at medium altitude, slightly angled to show top and side. "
        f"{style_bits}."
    )

    return prompt


# ============================================================
# 4. DESIGN GENERATION
# ============================================================

def generate_design() -> AircraftDesign:
    cfg = CONFIG

    animals = choose_animals()
    traits = accumulate_traits(animals)
    animal_count = len(animals)
    trait_count = len(traits)

    empty_mass_kg = rand_uniform(*cfg["EMPTY_MASS_KG_RANGE"])
    payload_mass_kg = rand_uniform(*cfg["PAYLOAD_MASS_KG_RANGE"])

    rotor_count = rand_int(*cfg["ROTOR_COUNT_RANGE"])
    max_twr = rand_uniform(*cfg["MAX_TWR_RANGE"])

    if "LIFT_BURST_POWER" in traits:
        burst_power_factor = rand_uniform(*cfg["BURST_POWER_FACTOR_RANGE"])
        burst_duration_s = rand_uniform(*cfg["BURST_DURATION_S_RANGE"])
    else:
        burst_power_factor = 1.0
        burst_duration_s = 0.0

    if "LIFT_UNSTEADY_AERO" in traits:
        unsteady_lift_gain = rand_uniform(*cfg["UNSTEADY_LIFT_GAIN_RANGE"])
    else:
        unsteady_lift_gain = 0.0

    energy_type = choose_energy_system_type()
    prof = ENERGY_SYSTEM_PROFILES[energy_type]

    battery_mass_kg = rand_uniform(*cfg["BATTERY_MASS_KG_RANGE"])
    spec_min, spec_max = prof["spec_energy_range"]
    battery_spec_energy = rand_uniform(spec_min, spec_max)
    battery_energy_Wh = battery_mass_kg * battery_spec_energy

    if "LIFT_BURST_POWER" in traits or energy_type == "fuel_cell_li_ion_hybrid":
        supercap_mass_kg = rand_uniform(*cfg["SUPERCAP_MASS_KG_RANGE"])
    else:
        supercap_mass_kg = 0.0
    supercap_energy_Wh = supercap_mass_kg * cfg["SUPERCAP_SPEC_ENERGY_WH_PER_KG"]

    structural_material, mat_prof = choose_structural_material()
    structural_material_class = mat_prof["class"]
    structural_extra_failure_risk = mat_prof["structural_extra_failure_risk"]
    rotor_blade_material = weighted_choice(ROTOR_BLADE_MATERIAL_OPTIONS)
    landing_gear_material = weighted_choice(LANDING_GEAR_MATERIAL_OPTIONS)

    frame_stiffness = rand_uniform(*cfg["FRAME_STIFFNESS_LONGITUDINAL_RANGE"])
    frame_stiffness *= mat_prof["frame_stiffness_factor"]
    if "STAB_COMPLIANT_SPINE" in traits:
        frame_stiffness = min(frame_stiffness, 0.7)

    tendon_cable_fraction = 0.0
    if "STRUCT_TENDON_CABLES" in traits:
        tendon_cable_fraction = rand_uniform(*cfg["TENDON_CABLE_FRACTION_RANGE"])

    gust_gain = rand_uniform(*cfg["GUST_REJECTION_GAIN_RANGE"])
    if "STAB_GUST_REJECTION" in traits:
        gust_gain = max(gust_gain, 1.2)

    landing_gear_mass_kg = rand_uniform(*cfg["LANDING_GEAR_MASS_KG_RANGE"])
    landing_gear_mass_kg *= mat_prof["landing_gear_mass_factor"]
    max_touchdown_velocity_mps = rand_uniform(*cfg["MAX_TOUCHDOWN_VELOCITY_MPS_RANGE"])
    if "STRUCT_ROBUST_GEAR" in traits:
        landing_gear_mass_kg *= 1.2
        max_touchdown_velocity_mps *= 1.5

    cruise_speed_mps = rand_uniform(*cfg["CRUISE_SPEED_RANGE_MPS"])
    climb_rate_mps = rand_uniform(*cfg["CLIMB_RATE_RANGE_MPS"])
    mode_count = rand_int(*cfg["MODE_COUNT_RANGE"])
    if "MISSION_MULTI_GAIT" in traits:
        mode_count = max(mode_count, 4)

    mtow_kg = empty_mass_kg + payload_mass_kg
    payload_to_aircraft_ratio = payload_mass_kg / empty_mass_kg

    rule_empty_mass_ok = empty_mass_kg <= cfg["DARPA_MAX_EMPTY_MASS_KG"]
    rule_payload_ok = payload_mass_kg >= cfg["DARPA_MIN_PAYLOAD_MASS_KG"]

    design_qualifying = rule_empty_mass_ok and rule_payload_ok
    design_qualifying_score = payload_to_aircraft_ratio if design_qualifying else 0.0

    # Propulsion architecture
    arch_key, arch_prof = choose_propulsion_architecture()
    secondary_fraction_range = arch_prof["secondary_propulsor_fraction_range"]
    if secondary_fraction_range[0] == secondary_fraction_range[1]:
        secondary_fraction = secondary_fraction_range[0]
    else:
        secondary_fraction = rand_uniform(*secondary_fraction_range)

    design_id = f"DLIFT_{uuid.uuid4().hex[:10]}"

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
        supercap_mass_kg=supercap_mass_kg,
        supercap_energy_Wh=supercap_energy_Wh,
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
        design_summary="",   # filled below
        propulsion_architecture=arch_key,
        primary_propulsor_type=arch_prof["primary_propulsor_type"],
        secondary_propulsor_type=arch_prof["secondary_propulsor_type"],
        secondary_propulsor_fraction=secondary_fraction,
        propulsion_hover_power_factor=arch_prof["hover_power_factor"],
        propulsion_cruise_power_factor=arch_prof["cruise_power_factor"],
        propulsion_arch_extra_failure_risk=arch_prof["extra_failure_risk"],
        image_prompt="",     # filled below
    )

    summary = describe_design(tmp_design)
    tmp_design.design_summary = summary
    tmp_design.image_prompt = build_image_prompt(tmp_design, summary)
    return tmp_design


# ============================================================
# 5. MISSION PHYSICS & RULES
# ============================================================

def generate_environment() -> Environment:
    cfg = CONFIG
    wind_speed_kts = rand_uniform(*cfg["WIND_SPEED_KTS_RANGE"])
    turbulence_index = rand_uniform(*cfg["TURBULENCE_INDEX_RANGE"])
    return Environment(wind_speed_kts=wind_speed_kts, turbulence_index=turbulence_index)


def simulate_mission(design: AircraftDesign, env: Environment) -> Tuple[MissionResult, List[TimeStep]]:
    """
    Simulate one mission for a given design and environment.
    
    - Wind affects effective cruise groundspeed via WIND_SPEED_TIME_SLOWDOWN_FACTOR.
    - Missions are truncated as soon as:
        * time exceeds DARPA_MAX_MISSION_TIME_S, or
        * battery energy drops below zero, or
        * a random failure triggers at a sampled time.
    """

    cfg = CONFIG
    dt = cfg["TIME_STEP_S"]
    ts_rows: List[TimeStep] = []

    # Immediate rejection if design doesn't meet mass/payload rules
    if not (design.rule_empty_mass_ok and design.rule_payload_ok):
        mission = MissionResult(
            design_id=design.design_id,
            mission_id=f"MSN_{uuid.uuid4().hex[:10]}",
            wind_speed_kts=env.wind_speed_kts,
            turbulence_index=env.turbulence_index,
            success=False,
            failure_phase="precheck",
            failure_reason="mass_or_payload_rule_violation",
            rule_violation="mass_or_payload",
            total_time_s=0.0,
            time_takeoff_climb_s=0.0,
            time_loaded_cruise_s=0.0,
            time_empty_cruise_s=0.0,
            time_descent_landing_s=0.0,
            energy_used_Wh=0.0,
            battery_energy_remaining_Wh=design.battery_energy_Wh,
            payload_to_aircraft_ratio=design.payload_to_aircraft_ratio,
            is_qualifying_run=False,
            qualifying_score=0.0,
        )
        return mission, ts_rows

    # --- Helpers ---------------------------------------------------------

    def hover_power(mass_kg: float) -> float:
        base = cfg["HOVER_POWER_COEFF"] * (mass_kg ** 1.5)
        reduction = base * design.unsteady_lift_gain * 0.5
        power = max(base - reduction, 0.0)
        return power * design.propulsion_hover_power_factor

    def cruise_power(mass_kg: float, speed_mps: float) -> float:
        base = cfg["CRUISE_POWER_COEFF"] * mass_kg * (speed_mps / 10.0)
        return base * design.propulsion_cruise_power_factor

    # Environment fractions
    max_wind = max(cfg["WIND_SPEED_KTS_RANGE"][1], 1e-6)
    wind_frac = env.wind_speed_kts / max_wind
    turb_frac = min(env.turbulence_index / max(cfg["TURBULENCE_INDEX_RANGE"][1], 1e-6), 1.5)

    def apply_env_penalties(power_W: float) -> float:
        factor = 1.0 + cfg["WIND_POWER_PENALTY_FACTOR"] * wind_frac \
                 + cfg["TURB_POWER_PENALTY_FACTOR"] * turb_frac
        return power_W * factor

    # Mission geometry and nominal speeds
    altitude_m = 0.0
    distance_m = 0.0
    t = 0.0
    energy_cum_Wh = 0.0
    battery_energy = design.battery_energy_Wh
    supercap_energy = design.supercap_energy_Wh

    altitude_target_m = CONFIG["CRUISE_ALTITUDE_FT"] * 0.3048
    cruise_speed = design.cruise_speed_mps
    climb_rate = design.climb_rate_mps

    def nm_to_m(nm: float) -> float:
        return nm * 1852.0

    dist_loaded_total_m = nm_to_m(cfg["PAYLOAD_LEG_DISTANCE_NM"])
    dist_empty_total_m = nm_to_m(cfg["RETURN_LEG_DISTANCE_NM"])

    # Effective cruise groundspeeds include wind slowdown
    slowdown_factor = cfg["WIND_SPEED_TIME_SLOWDOWN_FACTOR"]
    slowdown = slowdown_factor * wind_frac
    slowdown = max(0.0, min(slowdown, 0.7))  # cap at 70% slowdown
    effective_cruise_speed = max(cruise_speed * (1.0 - slowdown), cruise_speed * 0.3)

    # Pre-compute phase durations based on effective speeds
    time_takeoff_climb = altitude_target_m / max(climb_rate, 0.5)
    time_loaded_cruise = dist_loaded_total_m / max(effective_cruise_speed, 0.5)
    time_empty_cruise = dist_empty_total_m / max(effective_cruise_speed, 0.5)
    time_descent_landing = altitude_target_m / max(climb_rate, 0.5) if altitude_target_m > 0 else 0.0

    # Phase time boundaries (for random failure timing)
    t0 = 0.0
    t1 = t0 + time_takeoff_climb
    t2 = t1 + time_loaded_cruise
    t3 = t2 + time_empty_cruise
    t4 = t3 + time_descent_landing

    # --- Random failure pre-sampling ------------------------------

    random_failure_time = None
    random_failure_phase = ""
    random_failure_reason = ""
    random_failure_rule = ""

    if cfg["ENABLE_RANDOM_FAILURES"]:
        load_ratio = design.payload_to_aircraft_ratio
        load_term = max(load_ratio - 2.0, 0.0) / 3.0
        stress_index = 0.3 * wind_frac + 0.4 * turb_frac + 0.3 * load_term
        stress_index = max(0.0, min(stress_index, 2.0))

        base_p = cfg["BASE_RANDOM_FAILURE_RATE"]

        p_gust = base_p * (0.5 + stress_index)
        p_control = base_p * (0.3 + 0.7 * turb_frac)
        touchdown_ratio = design.climb_rate_mps / max(design.max_touchdown_velocity_mps, 0.1)
        p_touchdown = base_p * max(touchdown_ratio - 1.0, 0.0)
        p_energy_fault = base_p * 0.5 + design.energy_system_extra_failure_risk
        p_structural = base_p * 0.4 + design.structural_extra_failure_risk
        p_prop_arch = base_p * 0.4 + design.propulsion_arch_extra_failure_risk

        total_p = p_gust + p_control + p_touchdown + p_energy_fault + p_structural + p_prop_arch
        total_p = min(total_p, 0.9)

        r = random.random()
        if r < total_p:
            # A random failure will occur; choose type & phase
            threshold1 = p_gust
            threshold2 = threshold1 + p_control
            threshold3 = threshold2 + p_touchdown
            threshold4 = threshold3 + p_energy_fault
            threshold5 = threshold4 + p_structural

            if r < threshold1:
                random_failure_phase = "loaded_cruise"
                random_failure_reason = "gust_induced_instability"
                random_failure_rule = ""
                phase_start, phase_end = t1, t2
            elif r < threshold2:
                random_failure_phase = "empty_cruise"
                random_failure_reason = "control_saturation"
                random_failure_rule = ""
                phase_start, phase_end = t2, t3
            elif r < threshold3:
                random_failure_phase = "descent_landing"
                random_failure_reason = "hard_touchdown"
                random_failure_rule = "landing_profile_violation"
                phase_start, phase_end = t3, t4
            elif r < threshold4:
                random_failure_phase = "mission"
                random_failure_reason = "energy_system_fault"
                random_failure_rule = "energy_architecture_risk"
                phase_start, phase_end = t0, t4
            elif r < threshold5:
                random_failure_phase = "loaded_cruise"
                random_failure_reason = "structural_overload"
                random_failure_rule = "structural_margin_risk"
                phase_start, phase_end = t1, t2
            else:
                random_failure_phase = "mission"
                random_failure_reason = "propulsion_architecture_fault"
                random_failure_rule = "propulsion_architecture_risk"
                phase_start, phase_end = t0, t4

            random_failure_time = rand_uniform(phase_start, phase_end)

    mission_id = f"MSN_{uuid.uuid4().hex[:10]}"

    # Failure state (will be updated during integration)
    success = True
    failure_phase = ""
    failure_reason = ""
    rule_violation = ""

    # --- Core integration loop per phase --------------------------------

    def run_phase(
        phase_name: str,
        duration_s: float,
        mass_kg: float,
        payload_attached: bool,
        horizontal_speed_mps: float,
        vertical_rate_mps: float,
    ) -> bool:
        """
        Run a mission phase step-by-step.
        Returns False if we should stop the mission early due to failure.
        """
        nonlocal t, altitude_m, distance_m, energy_cum_Wh, battery_energy, supercap_energy
        nonlocal success, failure_phase, failure_reason, rule_violation
        nonlocal random_failure_time, random_failure_phase, random_failure_reason, random_failure_rule

        phase_start_t = t
        steps = max(1, int(duration_s / dt))

        for _ in range(steps):
            if not success:
                return False

            remaining_s = phase_start_t + duration_s - t
            step_dt = min(dt, max(remaining_s, 0.0))
            if step_dt <= 0:
                break

            # Advance kinematics
            altitude_m += vertical_rate_mps * step_dt
            altitude_m = max(0.0, altitude_m)
            distance_m += horizontal_speed_mps * step_dt

            # Determine power regime
            if abs(vertical_rate_mps) > 0.0 and horizontal_speed_mps < 1e-3:
                base_power_W = hover_power(mass_kg)
            elif abs(vertical_rate_mps) > 0.0 and horizontal_speed_mps > 0.0:
                base_power_W = hover_power(mass_kg) * 0.5 + cruise_power(mass_kg, horizontal_speed_mps) * 0.5
            else:
                base_power_W = cruise_power(mass_kg, horizontal_speed_mps)

            power_W = apply_env_penalties(base_power_W)
            energy_step_Wh = power_W * step_dt / 3600.0

            # Use supercaps first for takeoff/climb bursts
            energy_from_supercap = 0.0
            if phase_name == "takeoff_climb" and design.burst_power_factor > 1.0 and supercap_energy > 0.0:
                use_Wh = min(energy_step_Wh, supercap_energy)
                energy_from_supercap = use_Wh
                supercap_energy -= use_Wh

            energy_from_battery = energy_step_Wh - energy_from_supercap
            battery_energy -= energy_from_battery
            energy_cum_Wh += energy_step_Wh

            # Advance time
            t += step_dt

            # Record timestep
            ts_rows.append(TimeStep(
                design_id=design.design_id,
                mission_id=mission_id,
                t_s=t,
                phase=phase_name,
                altitude_m=altitude_m,
                distance_m=distance_m,
                speed_mps=horizontal_speed_mps,
                mass_kg=mass_kg,
                power_W=power_W,
                energy_used_Wh_cum=energy_cum_Wh,
                battery_remaining_Wh=battery_energy,
                payload_attached=payload_attached,
                wind_speed_kts=env.wind_speed_kts,
                turbulence_index=env.turbulence_index,
            ))

            # Rule-based failures: time and energy limits
            if t > cfg["DARPA_MAX_MISSION_TIME_S"]:
                success = False
                failure_phase = "mission"
                failure_reason = "time_limit_exceeded"
                rule_violation = "time_limit"
                return False

            if battery_energy < 0.0:
                success = False
                failure_phase = phase_name
                failure_reason = "energy_depleted"
                rule_violation = "energy_budget"
                return False

            # Random failure trigger (if one is scheduled)
            if random_failure_time is not None and t >= random_failure_time and success:
                success = False
                # Use pre-sampled failure descriptors
                failure_phase = random_failure_phase
                failure_reason = random_failure_reason
                rule_violation = random_failure_rule
                return False

        return True

    # Run phases in order, stopping early if needed
    mass_with_payload = design.mtow_kg
    mass_empty = design.empty_mass_kg

    # 1) Takeoff & climb
    cont = run_phase(
        "takeoff_climb",
        time_takeoff_climb,
        mass_with_payload,
        True,
        0.0,
        climb_rate,
    )
    if not cont:
        time_takeoff_climb_actual = t
        time_loaded_cruise_actual = 0.0
        time_empty_cruise_actual = 0.0
        time_descent_landing_actual = 0.0
    else:
        time_takeoff_climb_actual = time_takeoff_climb

        # 2) Loaded cruise (payload leg, with wind-affected groundspeed)
        start_t_loaded = t
        cont = run_phase(
            "loaded_cruise",
            time_loaded_cruise,
            mass_with_payload,
            True,
            effective_cruise_speed,
            0.0,
        )
        time_loaded_cruise_actual = t - start_t_loaded if t > start_t_loaded else 0.0

        if not cont:
            time_empty_cruise_actual = 0.0
            time_descent_landing_actual = 0.0
        else:
            # 3) Empty cruise (return leg)
            start_t_empty = t
            cont = run_phase(
                "empty_cruise",
                time_empty_cruise,
                mass_empty,
                False,
                effective_cruise_speed,
                0.0,
            )
            time_empty_cruise_actual = t - start_t_empty if t > start_t_empty else 0.0

            if not cont:
                time_descent_landing_actual = 0.0
            else:
                # 4) Descent & landing
                start_t_descent = t
                descent_rate = -climb_rate
                cont = run_phase(
                    "descent_landing",
                    time_descent_landing,
                    mass_empty,
                    False,
                    0.0,
                    descent_rate,
                )
                time_descent_landing_actual = t - start_t_descent if t > start_t_descent else 0.0

    total_time_s = t
    energy_used_Wh = energy_cum_Wh
    battery_remaining = battery_energy

    # If no failure was triggered, success remains True
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
        time_empty_cruise_s=time_empty_cruise_actual,
        time_descent_landing_s=time_descent_landing_actual,
        energy_used_Wh=energy_used_Wh,
        battery_energy_remaining_Wh=battery_remaining,
        payload_to_aircraft_ratio=design.payload_to_aircraft_ratio,
        is_qualifying_run=is_qualifying_run,
        qualifying_score=qualifying_score,
    )

    return mission, ts_rows


# ============================================================
# 6. DATASET GENERATION
# ============================================================

def designs_to_dataframe(designs: List[AircraftDesign]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for d in designs:
        row = asdict(d)
        row["animals"] = ",".join(d.animals)
        row["traits"] = ",".join(d.traits)
        rows.append(row)
    return pd.DataFrame(rows)


def missions_to_dataframe(missions: List[MissionResult]) -> pd.DataFrame:
    rows = [asdict(m) for m in missions]
    return pd.DataFrame(rows)


def timeseries_to_dataframe(ts: List[TimeStep]) -> pd.DataFrame:
    rows = [asdict(x) for x in ts]
    return pd.DataFrame(rows)


def generate_universe(
    n_designs: int,
    missions_per_design: int,
) -> Tuple[List[AircraftDesign], List[MissionResult], List[TimeStep]]:
    designs: List[AircraftDesign] = []
    missions: List[MissionResult] = []
    ts_rows: List[TimeStep] = []

    for _ in range(n_designs):
        d = generate_design()
        designs.append(d)
        for _ in range(missions_per_design):
            env = generate_environment()
            mission, ts = simulate_mission(d, env)
            missions.append(mission)
            ts_rows.extend(ts)

    return designs, missions, ts_rows


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    if CONFIG["RNG_SEED"] is not None:
        random.seed(CONFIG["RNG_SEED"])

    out_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    n_designs = CONFIG["N_DESIGNS"]
    missions_per_design = CONFIG["MISSIONS_PER_DESIGN"]

    designs, missions, ts_rows = generate_universe(n_designs, missions_per_design)

    df_designs = designs_to_dataframe(designs)
    df_missions = missions_to_dataframe(missions)
    df_ts = timeseries_to_dataframe(ts_rows)

    designs_path = os.path.join(out_dir, CONFIG["DESIGNS_CSV"])
    missions_path = os.path.join(out_dir, CONFIG["MISSIONS_CSV"])
    ts_path = os.path.join(out_dir, CONFIG["MISSIONS_TS_CSV"])

    df_designs.to_csv(designs_path, index=False)
    df_missions.to_csv(missions_path, index=False)
    df_ts.to_csv(ts_path, index=False)

    print(f"Generated {len(designs)} designs -> {designs_path}")
    print(f"Generated {len(missions)} missions -> {missions_path}")
    print(f"Generated {len(ts_rows)} time-series rows -> {ts_path}")
