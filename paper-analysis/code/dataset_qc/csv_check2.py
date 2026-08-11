import csv, sys
csv.field_size_limit(10_000_000)

def load_designs(path):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        ids = set()
        dup = 0
        bad_rows = 0
        empty_cols = {}
        for row in reader:
            if len(row) != ncols:
                bad_rows += 1
                continue
            if row[0] in ids:
                dup += 1
            ids.add(row[0])
            for i, v in enumerate(row):
                if v == "":
                    empty_cols[header[i]] = empty_cols.get(header[i], 0) + 1
        return header, ids, dup, bad_rows, empty_cols

def check_missions(path, design_ids):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        bad_rows = 0
        orphans = 0
        per_design = {}
        empty_key_cols = {}
        key_idx = {name: i for i, name in enumerate(header) if name in ("success","total_time_s","failure_phase","design_id","mission_id")}
        for row in reader:
            if len(row) != ncols:
                bad_rows += 1
                continue
            did = row[key_idx["design_id"]]
            if did not in design_ids:
                orphans += 1
            per_design[did] = per_design.get(did, 0) + 1
            for name in ("success","total_time_s"):
                if row[key_idx[name]] == "":
                    empty_key_cols[name] = empty_key_cols.get(name, 0) + 1
        counts = set(per_design.values())
        return header, bad_rows, orphans, counts, empty_key_cols

if __name__ == "__main__":
    ddir = sys.argv[1]
    label = sys.argv[2]
    dh, ids, dup, bad, empty_cols = load_designs(ddir + "/designs.csv")
    print(f"=== {label} designs.csv ===")
    print(f"  cols={len(dh)} rows={len(ids)+dup+bad} unique_ids={len(ids)} dup_ids={dup} malformed_rows={bad}")
    if empty_cols:
        print(f"  empty-value columns: {empty_cols}")
    mh, mbad, orphans, counts, mempty = check_missions(ddir + "/missions.csv", ids)
    print(f"=== {label} missions.csv ===")
    print(f"  cols={len(mh)} malformed_rows={mbad} orphan_design_id_refs={orphans} distinct_missions_per_design_counts={counts}")
    if mempty:
        print(f"  empty success/total_time_s counts: {mempty}")
