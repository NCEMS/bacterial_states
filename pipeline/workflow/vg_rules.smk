rule vg_map:
    input:
        r1="results/clean_R1.fastq",
        r2="results/clean_R2.fastq",
        xg=config["vg_graph_xg"],
        gcsa=config["vg_graph_gcsa"]
    output:
        gam="results/sample.gam"
    shell:
        "vg map -x {input.xg} -g {input.gcsa} -f {input.r1} -f {input.r2} > {output.gam}"

rule surject_to_k12:
    input:
        gam="results/sample.gam",
        xg=config["vg_graph_xg"]
    output:
        bam="results/sample_k12.bam"
    shell:
        "vg surject -x {input.xg} -b {input.gam} -p K12 > {output.bam}"

rule featurecounts:
    input:
        bam="results/sample_k12.bam",
        gtf=config["k12_gtf"]
    output:
        counts="results/counts_vg_k12.tsv"
    shell:
        "featureCounts -a {input.gtf} -o {output.counts} {input.bam}"

