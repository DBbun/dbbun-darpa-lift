import csv, sys, os

def check(path, label):
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)
        bad_field_count = 0
        empty_field_rows = 0
        ids = {}
        dup_ids = 0
        rownum = 1
        empty_cols = {i: 0 for i in range(ncols)}
        for row in reader:
            rownum += 1
            if len(row) != ncols:
                bad_field_count += 1
                continue
            has_empty = False
            for i, v in enumerate(row):
                if v == "":
                    empty_cols[i] += 1
                    has_empty = True
            if has_empty:
                empty_field_rows += 1
            rid = row[0]
            if rid in ids:
                dup_ids += 1
            ids[rid] = ids.get(rid, 0) + 1
        print(f"=== {label} ===")
        print(f"  header cols: {ncols}, data rows: {rownum-1}")
        print(f"  rows with wrong field count: {bad_field_count}")
        print(f"  rows with >=1 empty field: {empty_field_rows}")
        print(f"  duplicate id rows: {dup_ids}")
        nonzero_empty = {header[i]: c for i, c in empty_cols.items() if c > 0}
        if nonzero_empty:
            print(f"  columns with empty values: {nonzero_empty}")
        return set(ids.keys())

if __name__ == "__main__":
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    check(path, label)
