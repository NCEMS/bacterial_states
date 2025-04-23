rule fastp:
    input:
        r1=config["r1"],
        r2=config.get("r2", None)
    output:
        r1="results/clean_R1.fastq",
        r2="results/clean_R2.fastq",
        html="results/fastp.html",
        json="results/fastp.json"
    shell:
        """
        fastp -i {input.r1} -o {output.r1} \
        {("-I " + input.r2 + " -O " + output.r2) if input.r2 else ""} \
        -h {output.html} -j {output.json}
        """
    conda: "envs/rnaseq.yaml"


rule fastqc:
    input:
        r1="results/clean_R1.fastq",
        r2="results/clean_R2.fastq"
    output:
        r1_html="results/fastqc/clean_R1_fastqc.html",
        r1_zip="results/fastqc/clean_R1_fastqc.zip",
        r2_html="results/fastqc/clean_R2_fastqc.html",
        r2_zip="results/fastqc/clean_R2_fastqc.zip"
    shell:
        """
        fastqc {input.r1} {input.r2 if input.r2 else ''} -o results/fastqc
        """
    conda: "envs/rnaseq.yaml"


rule centrifuge:
    input:
        r1="results/clean_R1.fastq",
        r2="results/clean_R2.fastq"
    output:
        report="results/centrifuge_report.tsv",
        summary="results/centrifuge_summary.tsv"
    params:
        db=config["centrifuge_index"]
    shell:
        """
        centrifuge -x {params.db} \
        {("-1 " + input.r1 + " -2 " + input.r2) if input.r2 else ("-U " + input.r1)} \
        -S {output.report} --report-file {output.summary}
        """
    conda: "envs/rnaseq.yaml"


rule salmon_quant:
    input:
        r1="results/clean_R1.fastq",
        r2="results/clean_R2.fastq",
        index=config["salmon_index"]
    output:
        quant="results/salmon/quant.sf"
    shell:
        """
        salmon quant -i {input.index} -l ISR \
        {("-1 " + input.r1 + " -2 " + input.r2) if input.r2 else ("-r " + input.r1)} \
        -o results/salmon --validateMappings
        """
    conda: "envs/rnaseq.yaml"
    
    
rule sum_orthologs:
    input:
        quant="results/salmon/quant.sf",
        map=config["ortholog_map"]  # TSV: transcript_id \t orthogroup
    output:
        tsv="results/counts_salmon_orthogroups.tsv"
    script:
        "scripts/sum_orthologs.py"
    conda: "envs/rnaseq.yaml"
    

rule multiqc:
    input:
        fastp="results/fastp.json",
        fastqc_r1="results/fastqc/clean_R1_fastqc.zip",
        fastqc_r2="results/fastqc/clean_R2_fastqc.zip",
        salmon="results/salmon/quant.sf",
        centrifuge="results/centrifuge_summary.tsv"
    output:
        html="results/multiqc_report.html"
    shell:
        """
        multiqc results/ -o results
        """
    conda: "envs/rnaseq.yaml"
