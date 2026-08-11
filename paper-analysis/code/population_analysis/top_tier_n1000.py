import csv
from collections import Counter

path = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 1000) v1.2\designs.csv"

with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["design_rank_score"] = float(r["design_rank_score"])
    r["payload_to_aircraft_ratio"] = float(r["payload_to_aircraft_ratio"])
    r["rotor_count"] = int(r["rotor_count"])

max_score = max(r["design_rank_score"] for r in rows)
top_tier = [r for r in rows if r["design_rank_score"] == max_score]
print(f"max design_rank_score = {max_score}, {len(top_tier)} designs tied at that score (of {len(rows)})")

ratios = [r["payload_to_aircraft_ratio"] for r in top_tier]
ratios.sort()
print(f"top-tier ratio range: min={ratios[0]:.2f} max={ratios[-1]:.2f} median={ratios[len(ratios)//2]:.2f}")

cols = ["primary_propulsor_type", "propulsion_architecture", "energy_system_type", "structural_material", "wing_foldable"]
for c in cols:
    cnt = Counter(r[c] for r in top_tier)
    total = len(top_tier)
    print(f"\n{c} distribution in top tier (n={total}):")
    for val, n in cnt.most_common():
        print(f"  {val}: {n} ({100*n/total:.1f}%)")

rc = Counter(r["rotor_count"] for r in top_tier)
print(f"\nrotor_count distribution in top tier:")
for val, n in sorted(rc.items()):
    print(f"  {val}: {n}")

# For comparison: same distributions across ALL 1000 designs (baseline)
print("\n\n=== BASELINE: same distributions across all 1000 designs ===")
for c in cols:
    cnt = Counter(r[c] for r in rows)
    total = len(rows)
    print(f"\n{c} distribution overall (n={total}):")
    for val, n in cnt.most_common():
        print(f"  {val}: {n} ({100*n/total:.1f}%)")
