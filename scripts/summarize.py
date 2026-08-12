import csv
from collections import defaultdict


def main():
    path = "experiments/results/results.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    last = {}
    for r in rows:
        last[r["name"]] = r
    cols = ["name"]
    for r in last.values():
        for k in r:
            if k is not None and k not in cols:
                cols.append(k)
    print(",".join(cols))
    for name in sorted(last):
        r = last[name]
        print(",".join(str(r.get(c) or "") for c in cols))


if __name__ == "__main__":
    main()
