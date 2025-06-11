import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
import seaborn as sns
from matplotlib.patches import Patch
from scipy.spatial.distance import squareform


#####MAKING MATRIX######

#Loading mash distances
df = pd.read_csv("mash_distances.tsv", sep='\t', header=None)
df.columns = ["genome1", "genome2", "distance", "pvalue", "shared"]

#Removing known outlier
outlier = "genomes/GCF_021307345.1_ASM2130734v1_genomic.fna"
df = df[~df["genome1"].eq(outlier) & ~df["genome2"].eq(outlier)]

#Creating list of genomes
genomes = sorted(set(df["genome1"]).union(set(df["genome2"])))
genome_to_idx = {g: i for i, g in enumerate(genomes)}

print("building matrix", flush=True)
#Building distance matrix
n = len(genomes)
dist_matrix = np.zeros((n, n))
for _, row in df.iterrows():
    i = genome_to_idx[row["genome1"]]
    j = genome_to_idx[row["genome2"]]
    dist = row["distance"]
    dist_matrix[i, j] = dist
    dist_matrix[j, i] = dist

#Clustering matrix
condensed = squareform(dist_matrix)
Z = linkage(condensed, method="average")

#Saving matrix (for making figures etc so it doesnt have to be recalculated)
with open("linkage_matrix.pkl", "wb") as f:
    pickle.dump(Z, f)

#Loading matrix
with open("linkage_matrix.pkl", "rb") as f:
    Z = pickle.load(f)

#Getting genome names
df = pd.read_csv("mash_distances.tsv", sep='\t', header=None)
df.columns = ["genome1", "genome2", "distance", "pvalue", "shared"]
outlier = "genomes/GCF_021307345.1_ASM2130734v1_genomic.fna"  #Removing outlier from names
df = df[~df["genome1"].eq(outlier) & ~df["genome2"].eq(outlier)]
genomes = sorted(set(df["genome1"]).union(set(df["genome2"])))

#Setting clustering threshold
threshold = 0.015
clusters = fcluster(Z, t=threshold, criterion='distance')

#Force K-12 to be the selected genome for its cluster
forced_genome_match = [g for g in genomes if "GCF_000005845.2" in g]
if not forced_genome_match:
    raise ValueError("Genome GCF_000005845.2 not found in genome list.")
forced_genome = forced_genome_match[0]
forced_index = genomes.index(forced_genome)
original_cluster_id = clusters[forced_index]
forced_cluster_id = max(clusters) + 1
new_cluster_id_for_others = forced_cluster_id + 1
new_clusters = clusters.copy()
for i, c in enumerate(clusters):
    if c == original_cluster_id and i != forced_index:
        new_clusters[i] = new_cluster_id_for_others
new_clusters[forced_index] = forced_cluster_id
clusters = new_clusters
if len(genomes) != len(clusters):
    raise ValueError(f"Mismatch: {len(genomes)} genomes vs {len(clusters)} cluster assignments")

#Create cluster dataframe
cluster_df = pd.DataFrame({
    "genome": genomes,
    "cluster": clusters
})
cluster_df.to_csv("genome_clusters.csv", index=False)

#Select one representative genome, just alphabetical
cluster_df_sorted = cluster_df.sort_values("genome")
representatives = cluster_df_sorted.groupby("cluster").first().reset_index()
representatives.to_csv("representative_genomes.csv", index=False)




######CREATING FIGURES######

##ALL GENOMES##

unique_clusters = sorted(cluster_df["cluster"].unique())
palette = sns.color_palette("hsv", len(unique_clusters))
cluster_colors = {cluster: palette[i] for i, cluster in enumerate(unique_clusters)}
def link_color_func(node_id):
    if node_id < len(genomes):
        return "#000000"  # leaf node
    left = int(Z[node_id - len(genomes), 0])
    right = int(Z[node_id - len(genomes), 1])
    left_cluster = clusters[left] if left < len(clusters) else None
    right_cluster = clusters[right] if right < len(clusters) else None
    if left_cluster == right_cluster and left_cluster in cluster_colors:
        r, g, b = cluster_colors[left_cluster]
        return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    return "#999999"  # mixed cluster branches
visible_clusters = set()
for i in range(len(Z)):
    left = int(Z[i, 0])
    right = int(Z[i, 1])
    left_cluster = clusters[left] if left < len(clusters) else None
    right_cluster = clusters[right] if right < len(clusters) else None
    if left_cluster == right_cluster and left_cluster in cluster_colors:
        visible_clusters.add(left_cluster)
color_map = []
for cluster in sorted(visible_clusters):
    rgb = tuple(int(c * 255) for c in cluster_colors[cluster])
    color_str = f"rgb{rgb}"
    members = cluster_df[cluster_df["cluster"] == cluster]["genome"].tolist()
    for genome in members:
        color_map.append({
            "cluster": cluster,
            "color": color_str,
            "genome": genome
        })
color_map_df = pd.DataFrame(color_map)
color_map_df.to_csv("cluster_color_key.csv", index=False)
legend_patches = []
for cluster in sorted(visible_clusters):
    rgb = cluster_colors[cluster]
    hex_color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
    legend_patches.append(Patch(color=hex_color, label=f'Cluster {cluster}'))
plt.figure(figsize=(20, 10))
dendrogram(
    Z,
    no_labels=True,
    color_threshold=threshold,
    link_color_func=link_color_func
)
plt.axhline(y=threshold, color='red', linestyle='--')
plt.title(f"Colored Dendrogram (cutoff = {threshold})\nGCF_000005845.2 isolated")

plt.legend(
    handles=legend_patches,
    title="Cluster Color Key",
    loc="upper right",
    bbox_to_anchor=(1.15, 1.0),
    fontsize='small'
)

plt.tight_layout()
plt.savefig("colored_dendrogram_with_legend.png")
plt.close()



##REPRESENTATIVE GENOMES##

#Load representative genomes
reps_df = pd.read_csv("representative_genomes.csv")
rep_genomes = set(reps_df["genome"])

#Load mash distances
df = pd.read_csv("mash_distances.tsv", sep='\t', header=None)
df.columns = ["genome1", "genome2", "distance", "pvalue", "shared"]

#Filter for representatives
rep_df = df[df["genome1"].isin(rep_genomes) & df["genome2"].isin(rep_genomes)]

#Build matrix
rep_list = sorted(rep_genomes)
rep_index = {g: i for i, g in enumerate(rep_list)}
n = len(rep_list)
dist_matrix = np.zeros((n, n))

for _, row in rep_df.iterrows():
    i = rep_index[row["genome1"]]
    j = rep_index[row["genome2"]]
    dist_matrix[i, j] = row["distance"]
    dist_matrix[j, i] = row["distance"]

#Condense and cluster
condensed = squareform(dist_matrix)
Z = linkage(condensed, method="average")

#Plotting dendrogram
plt.figure(figsize=(max(12, n * 0.15), 6))
dendrogram(Z, labels=[g.split("/")[-1] for g in rep_list], leaf_rotation=90, leaf_font_size=8)
plt.title("Dendrogram of Representative Genomes (1 per Cluster)")
plt.tight_layout()
plt.savefig("representative_dendrogram.png")
plt.close()


