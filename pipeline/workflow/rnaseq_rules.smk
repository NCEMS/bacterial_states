#Trimming
rule fastp:
    input:
        r1=config["r1"],
        r2=config.get("r2", None)
    output:
        r1=temp("results/clean_R1.fastq"),
        r2=temp("results/clean_R2.fastq"),
        html=temp("results/fastp.html"),
        json=temp("results/fastp.json")
    shell:
        """
        fastp -i {input.r1} -o {output.r1} \
        {("-I " + input.r2 + " -O " + output.r2) if input.r2 else ""} \
        -h {output.html} -j {output.json}
        """
    conda: "envs/rnaseq.yaml"

#Getting contamination levels
rule kraken2:
    input:
        r1="results/clean_R1.fastq",
        r2="results/clean_R2.fastq"
    output:
        report=temp("results/kraken_report.txt")
    params:
        db=config["kraken_db"]
    shell:
        "kraken2 --db {params.db} --paired {input.r1} {input.r2} --report {output.report} > /dev/null"
    conda: "envs/rnaseq.yaml"

#Alignment
rule salmon_quant:
    input:
        r1="results/clean_R1.fastq",
        r2="results/clean_R2.fastq",
        index=config["salmon_index"]
    output:
        quant="results/salmon/quant.sf"
    shell:
        "salmon quant -i {input.index} -l A -1 {input.r1} -2 {input.r2} -o results/salmon --validateMappings"
    conda: "envs/rnaseq.yaml"

#Summing together counts for orthologs
rule sum_orthologs:
    input:
        quant="results/salmon/quant.sf",
        map=config["ortholog_map"]
    output:
        tsv="results/counts_salmon_orthogroups.tsv"
    script:
        "scripts/sum_orthologs.py"
    conda: "envs/rnaseq.yaml"

#Cleaning up log files into one summarized stats file
rule parse_stats:
    input:
        fastp_json="results/fastp.json",
        kraken_report="results/kraken_report.txt"
    output:
        stats="results/stats.txt"
    script:
        "scripts/parse_stats.py"
    conda: "envs/rnaseq.yaml"

