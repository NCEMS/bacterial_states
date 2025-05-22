import pandas as pd

# Snakemake inputs/outputs
quant = snakemake.input["quant"]
output = snakemake.output["tsv"]

# Load quant.sf
df = pd.read_csv(quant, sep="\t")

# Only keep transcripts from W3110
df = df[df["Name"].str.startswith("NC_004431|")].copy()

# Extract gene name and orthogroup
df[["Genome", "Gene", "Orthogroup"]] = df["Name"].str.split("|", expand=True)

# Compute scaled NumReads / EffectiveLength
df["scaled"] = df["NumReads"] / df["EffectiveLength"]
scaling_factor = df["scaled"].sum()

# Compute TPM per orthogroup
grouped = df.groupby(["Orthogroup", "Gene"]).agg({
    "NumReads": "sum",
    "EffectiveLength": "mean"
}).reset_index()

grouped["scaled"] = grouped["NumReads"] / grouped["EffectiveLength"]
grouped["TPM"] = grouped["scaled"] / scaling_factor * 1e6

# Output: one gene per orthogroup (W3110-based)
grouped[["Gene", "TPM"]].to_csv(output, sep="\t", index=False)

