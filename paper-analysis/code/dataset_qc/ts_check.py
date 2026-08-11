import csv, sys
csv.field_size_limit(10_000_000)

def load_ids(path):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)
        return set(row[0] for row in reader)

def check_ts(path, design_ids, label):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        ncols = len(header)
        bad_rows = 0
        orphans = 0
        neg_altitude = 0
        power_exceeds = 0
        neg_energy = 0
        neg_battery = 0
        total = 0
        for row in reader:
            total += 1
            if len(row) != ncols:
                bad_rows += 1
                continue
            if row[idx["design_id"]] not in design_ids:
                orphans += 1
            try:
                if float(row[idx["altitude_m"]]) < -0.01:
                    neg_altitude += 1
                if "power_requested_W" in idx and "power_available_W" in idx:
                    if float(row[idx["power_requested_W"]]) > float(row[idx["power_available_W"]]) + 1e-6:
                        power_exceeds += 1
                if float(row[idx["energy_used_Wh_cum"]]) < -0.01:
                    neg_energy += 1
                if float(row[idx["battery_remaining_Wh"]]) < -0.01:
                    neg_battery += 1
            except (ValueError, KeyError):
                pass
        print(f"=== {label} ===")
        print(f"  cols={ncols} total_rows={total} malformed={bad_rows} orphan_design_id_refs={orphans}")
        print(f"  neg_altitude={neg_altitude} power_requested>available={power_exceeds} neg_energy_cum={neg_energy} neg_battery_remaining={neg_battery}")

if __name__ == "__main__":
    designs_path = sys.argv[1]
    ts_path = sys.argv[2]
    label = sys.argv[3]
    ids = load_ids(designs_path)
    check_ts(ts_path, ids, label)
