import csv

mpath = r"C:\DBBun\Code\DARPA Lift Challenge\Sample dataset (n = 1000) v1.2\missions.csv"
target_ids = {"DLIFT_1cb08b6b0b", "DLIFT_8124662365"}

with open(mpath, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["design_id"] in target_ids]

for did in target_ids:
    print(f"\n=== {did} missions ===")
    for r in rows:
        if r["design_id"] != did:
            continue
        print(
            r["mission_id"],
            "wind=%.1fkts" % float(r["wind_speed_kts"]),
            "turb=%.2f" % float(r["turbulence_index"]),
            "success=", r["success"],
            "failure_phase=", r["failure_phase"],
            "failure_reason=", r["failure_reason"],
            "rule_violation=", r["rule_violation"],
            "power_sat_s=", r["power_saturation_seconds"],
            "thermal_peak_C=", r["thermal_peak_C"],
        )
