import csv

path = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 1000) v1.2\designs.csv"

FIELDS_OF_INTEREST = [
    "design_id", "empty_mass_kg", "payload_mass_kg", "payload_to_aircraft_ratio",
    "rotor_count", "propulsion_architecture", "primary_propulsor_type",
    "energy_system_type", "structural_material", "wing_foldable",
    "design_qualifying", "design_qualifying_score",
    "design_success_rate", "design_qualifying_rate", "design_rank_score", "design_stars",
]

rows = []
with open(path, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

qualifying = [r for r in rows if r["design_qualifying"] == "True"]
print(f"total designs: {len(rows)}, qualifying: {len(qualifying)}")

ratios = sorted((float(r["payload_to_aircraft_ratio"]) for r in qualifying), reverse=True)
print(f"max ratio: {ratios[0]:.3f}, top10 ratios: {[round(x,3) for x in ratios[:10]]}")

real_band = [r for r in qualifying if 3.45 <= float(r["payload_to_aircraft_ratio"]) <= 3.84]
print(f"qualifying designs with ratio in real-world band [3.45, 3.84]: {len(real_band)} of {len(qualifying)} ({100*len(real_band)/len(qualifying):.1f}%)")

print("\n=== TOP 10 BY payload_to_aircraft_ratio (qualifying only) ===")
top_by_ratio = sorted(qualifying, key=lambda r: float(r["payload_to_aircraft_ratio"]), reverse=True)[:10]
for r in top_by_ratio:
    print({k: r[k] for k in FIELDS_OF_INTEREST})

print("\n=== TOP 10 BY design_rank_score ===")
top_by_rank = sorted(rows, key=lambda r: float(r["design_rank_score"]), reverse=True)[:10]
for r in top_by_rank:
    print({k: r[k] for k in FIELDS_OF_INTEREST})
