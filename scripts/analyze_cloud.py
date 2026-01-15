import csv
import sys
from collections import Counter

if len(sys.argv) < 2:
    print('Usage: python analyze_cloud.py <csv_path>')
    sys.exit(1)

path = sys.argv[1]
rows = []
with open(path, newline='', encoding='utf-8') as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        if not row:
            continue
        node = row[0]
        v = row[1] if len(row) > 1 else ''
        pid = row[2] if len(row) > 2 else ''
        rows.append((node, v, pid))

n_rows = len(rows)
unique_nodes = set(r[0] for r in rows)
counts = Counter(r[0] for r in rows)

# count NaN-like values
import math
nan_count = sum(1 for _, v, _ in rows if v.strip().lower() in ('nan', '') or not math.isfinite(float(v))) if rows else 0

print(f'File: {path}')
print(f'Rows (excluding header): {n_rows}')
print(f'Unique nodes: {len(unique_nodes)}')
print(f'Rows with non-finite V (NaN/inf/empty): {nan_count}')

# show distribution
most_common = counts.most_common(10)
print('\nTop 10 nodes by count:')
for node, cnt in most_common:
    print(f'  node={node} count={cnt}')

# show range of nodes
nodes_sorted = sorted(float(n) for n in unique_nodes)
print(f'Node min/max: {nodes_sorted[0]} / {nodes_sorted[-1]}')

# basic per-node counts stats
vals = list(counts.values())
import statistics
print('\nPer-node counts: min={0}, max={1}, mean={2:.2f}, median={3}'.format(min(vals), max(vals), statistics.mean(vals), statistics.median(vals)))

# print first 20 rows
print('\nFirst 20 rows:')
for row in rows[:20]:
    print(row)

# print last 20 rows
print('\nLast 20 rows:')
for row in rows[-20:]:
    print(row)
