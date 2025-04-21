import pandas as pd
from collections import defaultdict

quant = snakemake.input["quant"]
orthomap = snakemake.input["map"]
output = snakemake.output["tsv"]

#Load quant.sf
df = pd.read_csv(quant, sep="\t")
tpm_dict = dict(zip(df["Name"], df["TPM"]))

#Load ortholog map
ortho_map = pd.read_csv(orthomap, sep="\t", header=None, names=["transcript", "orthogroup"])
orthogroup_tpm = defaultdict(float)

for _, row in ortho_map.iterrows():
    orthogroup_tpm[row["orthogroup"]] += tpm_dict.get(row["transcript"], 0.0)

#Write final summed TPMs
pd.DataFrame(orthogroup_tpm.items(), columns=["Orthogroup", "TPM"]).to_csv(output, sep="\t", index=False)

