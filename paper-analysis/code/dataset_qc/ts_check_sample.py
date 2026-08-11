import csv, sys
csv.field_size_limit(10_000_000)

def load_ids(path):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        next(reader)
        return set(row[0] for row in reader)

def check_ts(path, design_ids, label, limit):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        ncols = len(header)
        bad_rows = 0
        orphans = 0
        neg_altitude = 0
        power_exceeds = 0
        total = 0
        for row in reader:
            total += 1
            if total > limit:
                break
            if len(row) != ncols:
                bad_rows += 1
                continue
            if row[idx["design_id"]] not in design_ids:
                orphans += 1
            try:
                if float(row[idx["altitude_m"]]) < -0.01:
                    neg_altitude += 1
                if float(row[idx["power_requested_W"]]) > float(row[idx["power_available_W"]]) + 1e-6:
                    power_exceeds += 1
            except (ValueError, KeyError):
                pass
        print(f"=== {label} (sampled first {total} rows) ===")
        print(f"  malformed={bad_rows} orphan_design_id_refs={orphans} neg_altitude={neg_altitude} power_exceeds={power_exceeds}")

if __name__ == "__main__":
    ids = load_ids(sys.argv[1])
    check_ts(sys.argv[2], ids, sys.argv[3], int(sys.argv[4]))
