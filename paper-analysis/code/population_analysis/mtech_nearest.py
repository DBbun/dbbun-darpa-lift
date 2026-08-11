import csv

path = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 1000) v1.2\designs.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

MTECH_EMPTY = 14.51   # 32 lb
MTECH_PAYLOAD = 53.07 # 117 lb
MTECH_ROTOR = 6

candidates = [
    r for r in rows
    if r["primary_propulsor_type"] == "multirotor"
    and int(r["rotor_count"]) == MTECH_ROTOR
    and r["energy_system_type"] in ("li_ion", "li_s", "solid_state")
]
print(f"exact rotor_count=6, multirotor, electric: {len(candidates)} candidates")

def dist(r):
    de = float(r["empty_mass_kg"]) - MTECH_EMPTY
    dp = float(r["payload_mass_kg"]) - MTECH_PAYLOAD
    return (de**2 + dp**2) ** 0.5

candidates.sort(key=dist)
print("\n=== 8 nearest neighbors to MTech (empty=14.51kg, payload=53.07kg, 6 rotors) ===")
for r in candidates[:8]:
    print(
        r["design_id"],
        "dist=%.2f" % dist(r),
        "empty=%.1f" % float(r["empty_mass_kg"]),
        "payload=%.1f" % float(r["payload_mass_kg"]),
        "ratio=%.2f" % float(r["payload_to_aircraft_ratio"]),
        "tendon_frac=%.2f" % float(r["tendon_cable_fraction"]),
        "arch=", r["propulsion_architecture"],
        "energy=", r["energy_system_type"],
        "material=", r["structural_material"],
        "success=", r["design_success_rate"],
        "rank=", r["design_rank_score"],
        "stars=", r["design_stars"],
    )
