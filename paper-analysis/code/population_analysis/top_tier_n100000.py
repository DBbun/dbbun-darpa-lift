import csv
from collections import Counter

path = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 100000) v1.2\designs.csv"

with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for r in rows:
    r["design_rank_score"] = float(r["design_rank_score"])
    r["payload_to_aircraft_ratio"] = float(r["payload_to_aircraft_ratio"])
    r["rotor_count"] = int(r["rotor_count"])

max_score = max(r["design_rank_score"] for r in rows)
top_tier = [r for r in rows if r["design_rank_score"] == max_score]
print(f"n={len(rows)}  max design_rank_score = {max_score}  top-tier count = {len(top_tier)}")

ratios = sorted(r["payload_to_aircraft_ratio"] for r in top_tier)
print(f"top-tier ratio range: min={ratios[0]:.2f} max={ratios[-1]:.2f} median={ratios[len(ratios)//2]:.2f} mean={sum(ratios)/len(ratios):.2f}")

real_band = [r for r in top_tier if 2.49 <= r["payload_to_aircraft_ratio"] <= 6.11]
print(f"top-tier designs with ratio in real-world observed band [2.49, 6.11]: {len(real_band)} of {len(top_tier)} ({100*len(real_band)/len(top_tier):.1f}%)")
narrow_band = [r for r in top_tier if 2.49 <= r["payload_to_aircraft_ratio"] <= 3.85]
print(f"top-tier designs with ratio in real WINNERS band [2.49, 3.85]: {len(narrow_band)} of {len(top_tier)} ({100*len(narrow_band)/len(top_tier):.1f}%)")

cols = ["primary_propulsor_type", "propulsion_architecture", "energy_system_type", "structural_material", "wing_foldable"]
for c in cols:
    cnt = Counter(r[c] for r in top_tier)
    total = len(top_tier)
    print(f"\n{c} distribution in top tier (n={total}):")
    for val, n in cnt.most_common():
        print(f"  {val}: {n} ({100*n/total:.1f}%)")

# raw ratio max in the whole population, and its success rate
top_ratio_design = max(rows, key=lambda r: r["payload_to_aircraft_ratio"])
print(f"\nmax raw ratio in population: {top_ratio_design['payload_to_aircraft_ratio']:.2f} (design_id={top_ratio_design['design_id']}, success_rate={top_ratio_design['design_success_rate']}, rank_score={top_ratio_design['design_rank_score']})")
