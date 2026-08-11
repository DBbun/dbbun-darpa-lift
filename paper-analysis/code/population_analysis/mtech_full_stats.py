import csv

path = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 1000) v1.2\designs.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

candidates = [
    r for r in rows
    if r["primary_propulsor_type"] == "multirotor"
    and int(r["rotor_count"]) == 6
    and r["energy_system_type"] in ("li_ion", "li_s", "solid_state")
]
print(f"Total candidates (rotor_count=6, multirotor, electric): {len(candidates)}\n")

success = [float(r["design_success_rate"]) for r in candidates]
ratio = [float(r["payload_to_aircraft_ratio"]) for r in candidates]
rank = [float(r["design_rank_score"]) for r in candidates]

print(f"success_rate: min={min(success):.2f} max={max(success):.2f} avg={sum(success)/len(success):.2f}")
print(f"ratio: min={min(ratio):.2f} max={max(ratio):.2f} avg={sum(ratio)/len(ratio):.2f}")
print(f"rank_score: min={min(rank):.2f} max={max(rank):.2f} avg={sum(rank)/len(rank):.2f}")

# narrow to designs whose mass profile is close-ish to MTech's real numbers (within 8kg empty, 20kg payload)
close = [r for r in candidates if abs(float(r["empty_mass_kg"])-14.51) < 8 and abs(float(r["payload_mass_kg"])-53.07) < 20]
print(f"\nWithin +/-8kg empty, +/-20kg payload of MTech's real numbers: {len(close)} designs")
for r in sorted(close, key=lambda r: float(r["payload_to_aircraft_ratio"])):
    print(
        r["design_id"],
        "empty=%.1f" % float(r["empty_mass_kg"]),
        "payload=%.1f" % float(r["payload_mass_kg"]),
        "ratio=%.2f" % float(r["payload_to_aircraft_ratio"]),
        "success=", r["design_success_rate"],
        "rank=", r["design_rank_score"],
        "stars=", r["design_stars"],
        "tendon=%.2f" % float(r["tendon_cable_fraction"]),
    )
