import pandas as pd
import math

input_file = "labeled.txt"
output_prefix = "labeled"
n_parts = 10

#Get data
df = pd.read_csv(input_file, sep="\t", dtype=str)
df['gse'] = df['gse'].str.strip()

#Sort by experiment ID
groups = [(gse, group_df) for gse, group_df in df.groupby('gse', sort=False)]

#Sort groups with smallest first
groups.sort(key=lambda x: len(x[1]))

#Round robin assignment
parts = [[] for _ in range(n_parts)]
for idx, group in enumerate(groups):
    part_index = idx % n_parts
    parts[part_index].append(group)

#Break into 10 smaller files with shorter experiments first
for i, part_groups in enumerate(parts, start=1):
    part_groups_sorted = sorted(part_groups, key=lambda x: len(x[1]))
    part_df = pd.concat([gdf for _, gdf in part_groups_sorted])
    output_file = f"{output_prefix}{i:02d}"
    part_df.to_csv(output_file, sep="\t", index=False)
    print(f"Wrote {len(part_df)} rows to {output_file}")

