import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

#Loading mash distance
df = pd.read_csv("mash_distances.tsv", sep='\t', header=None)
df.columns = ["g1", "g2", "dist", "pval", "shared"]

#Making matrix of distances
dists = defaultdict(dict)
for _, row in df.iterrows():
    dists[row.g1][row.g2] = row.dist
    dists[row.g2][row.g1] = row.dist
all_genomes = sorted(set(df["g1"]).union(df["g2"]))

#Setting the reference to K-12
reference = [g for g in all_genomes if "GCF_000005845" in g or "K-12" in g or "CP000948" in g]
if not reference:
    raise ValueError("Could not find a valid starting genome like K-12.")
selected = [reference[0]]
remaining = set(all_genomes) - set(selected)

#Selecting 50 genomes that are the most different from K-12 (51 gecause an outlier was discovered)
while len(selected) < 51:
    max_dist = -1
    next_genome = None
    for g in remaining:
        min_d = min([dists[g].get(s, 1.0) for s in selected])
        if min_d > max_dist:
            max_dist = min_d
            next_genome = g
    selected.append(next_genome)
    remaining.remove(next_genome)

#Build matrix for the 50 genomes
dist_matrix = np.zeros((len(selected), len(selected)))
for i, g1 in enumerate(selected):
    for j, g2 in enumerate(selected):
        if i == j:
            dist = 0.0
        else:
            dist = dists[g1].get(g2, 1.0)
        dist_matrix[i, j] = dist

dist_df = pd.DataFrame(dist_matrix, index=selected, columns=selected)

#Outlier to remove
outlier = "genomes/GCF_021307345.1_ASM2130734v1_genomic.fna"
if outlier in dist_df.columns:
    df_clean = dist_df.drop(index=outlier, columns=outlier)
    print(f"Removed outlier: {outlier}")
else:
    print(f"{outlier} not found in matrix. No changes made.")
    df_clean = dist_df.copy()

#Saving cleaned matrix
df_clean.to_csv("distance_matrix_50.csv")
with open("selected_genomes.txt", "w") as f:
    for genome in df_clean.index:
        f.write(f"{genome}\n")

#Make heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Mash Distance Heatmap (49 Genomes)")
plt.tight_layout()
plt.savefig("heatmap_49_filtered.png", dpi=300)
plt.close()

#Make dendrogram
linked = linkage(squareform(df_clean), method='average')
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=[g.split("/")[-1].replace("_genomic.fna", "") for g in df_clean.index],
           leaf_rotation=90, leaf_font_size=6)
plt.title("Dendrogram of 49 Representative Genomes (Outlier Removed)")
plt.tight_layout()
plt.savefig("dendrogram_49_filtered.png", dpi=300)
plt.close()

#Make pca plot
pca = PCA(n_components=2)
coords = pca.fit_transform(df_clean.values)

plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1])
for i, g in enumerate(df_clean.index):
    label = g.split("/")[-1].replace("_genomic.fna", "")
    plt.text(coords[i, 0], coords[i, 1], label, fontsize=6)
plt.title("PCA of Mash Distances (49 Genomes)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("pca_49_filtered.png", dpi=300)
plt.close()
