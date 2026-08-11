import csv

path = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 1000) v1.2\designs.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

matches = [
    r for r in rows
    if r["primary_propulsor_type"] == "multirotor"
    and 5 <= int(r["rotor_count"]) <= 7
    and float(r["tendon_cable_fraction"]) > 0
    and r["energy_system_type"] in ("li_ion", "li_s", "solid_state")
]
print(f"near-neighbor matches (rotor 5-7, multirotor, tendon cables, electric): {len(matches)} of {len(rows)}")
for r in sorted(matches, key=lambda r: float(r["payload_to_aircraft_ratio"]), reverse=True):
    print(
        r["design_id"],
        "rotor=", r["rotor_count"],
        "ratio=%.2f" % float(r["payload_to_aircraft_ratio"]),
        "empty=%.1f" % float(r["empty_mass_kg"]),
        "payload=%.1f" % float(r["payload_mass_kg"]),
        "success=", r["design_success_rate"],
        "rank=", r["design_rank_score"],
        "stars=", r["design_stars"],
        "tendon_frac=%.2f" % float(r["tendon_cable_fraction"]),
    )
