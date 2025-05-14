SE = config.get("r2") is None 

########################## Single-end Rules
if SE:
    rule fastp_se:
        input:
            r1=config["r1"]
        output:
            r1="results/clean_R1.fastq",
            html="results/fastp.html",
            json="results/fastp.json"
        conda: "../envs/rnaseq.yaml"
        shell:
            """
            fastp -i {input.r1} -o {output.r1} \
            -h {output.html} -j {output.json}
            """

    rule fastqc_se:
        input:
            r1="results/clean_R1.fastq"
        output:
            r1_html="results/fastqc/clean_R1_fastqc.html",
            r1_zip="results/fastqc/clean_R1_fastqc.zip"
        conda: "../envs/rnaseq.yaml"
        shell:
            "fastqc {input.r1} -o results/fastqc"

    rule kraken_se:
        input:
            r1="results/clean_R1.fastq"
        output:
            report="results/kraken/clean_R1_kraken_report.txt",
            classified="results/kraken/clean_R1_kraken_output.txt"
        params:
            db="../resources/kraken/"
        conda: "../envs/rnaseq.yaml"
        shell:
            """
            mkdir -p results/kraken
            kraken2 --db {params.db} --report {output.report} --output {output.classified} {input.r1}
            """

    rule salmon_quant_se:
        input:
            r1="results/clean_R1.fastq",
            index=config["salmon_index"]
        output:
            quant="results/salmon/quant.sf"
        conda: "../envs/rnaseq.yaml"
        shell:
            """
            salmon quant -i {input.index} -l ISR \
            -r {input.r1} \
            -o results/salmon --validateMappings
            """

########################## Paired-end Rules
else:
    rule fastp_pe:
        input:
            r1=config["r1"],
            r2=config["r2"]
        output:
            r1="results/clean_R1.fastq",
            r2="results/clean_R2.fastq",
            html="results/fastp.html",
            json="results/fastp.json"
        conda: "../envs/rnaseq.yaml"
        shell:
            """
            fastp -i {input.r1} -I {input.r2} \
            -o {output.r1} -O {output.r2} \
            -h {output.html} -j {output.json}
            """

    rule fastqc_pe:
        input:
            r1="results/clean_R1.fastq",
            r2="results/clean_R2.fastq"
        output:
            r1_html="results/fastqc/clean_R1_fastqc.html",
            r1_zip="results/fastqc/clean_R1_fastqc.zip",
            r2_html="results/fastqc/clean_R2_fastqc.html",
            r2_zip="results/fastqc/clean_R2_fastqc.zip"
        conda: "../envs/rnaseq.yaml"
        shell:
            "fastqc {input.r1} {input.r2} -o results/fastqc"

    rule kraken_pe:
        input:
            r1="results/clean_R1.fastq",
            r2="results/clean_R2.fastq"
        output:
            report="results/kraken/clean_PE_kraken_report.txt",
            classified="results/kraken/clean_PE_kraken_output.txt"
        params:
            db="resources/kraken2-db"
        conda: "../envs/rnaseq.yaml"
        shell:
            """
            mkdir -p results/kraken
            kraken2 --db {params.db} --report {output.report} --output {output.classified} \
                --paired {input.r1} {input.r2}
            """

    rule salmon_quant_pe:
        input:
            r1="results/clean_R1.fastq",
            r2="results/clean_R2.fastq",
            index=config["salmon_index"]
        output:
            quant="results/salmon/quant.sf"
        conda: "../envs/rnaseq.yaml"
        shell:
            """
            salmon quant -i {input.index} -l ISR \
            -1 {input.r1} -2 {input.r2} \
            -o results/salmon --validateMappings
            """

########################## Shared Rules
rule sum_orthologs:
    input:
        quant="results/salmon/quant.sf",
        map=config["ortholog_map"]
    output:
        tsv="results/counts_salmon_orthogroups.tsv"
    conda: "../envs/rnaseq.yaml"
    script:
        "../scripts/sum_orthologs.py"
#MultiQC inputs
multiqc_inputs = {
    "fastp": "results/fastp.json",
    "salmon": "results/salmon/quant.sf",
    "fastqc_r1": "results/fastqc/clean_R1_fastqc.zip",
    "kraken_report": "results/kraken/clean_R1_kraken_report.txt",
    "kraken_output": "results/kraken/clean_R1_kraken_output.txt"
}
if config.get("r2"):
    multiqc_inputs.update({
        "fastqc_r2": "results/fastqc/clean_R2_fastqc.zip",
        "kraken_report": "results/kraken/clean_PE_kraken_report.txt",
        "kraken_output": "results/kraken/clean_PE_kraken_output.txt"
    })

rule multiqc:
    input:
        **multiqc_inputs
    output:
        html="results/multiqc_report.html"
    conda: "../envs/rnaseq.yaml"
    shell:
        "multiqc results/ -o results"



