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
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            fastp -i {input.r1} -o {output.r1} \
            -h {output.html} -j {output.json} -w {threads}
            """

    rule fastqc_se:
        input:
            r1="results/clean_R1.fastq"
        output:
            r1_html="results/fastqc/clean_R1_fastqc.html",
            r1_zip="results/fastqc/clean_R1_fastqc.zip"
        conda: "../envs/rnaseq_vg.yaml"
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
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            mkdir -p results/kraken
            kraken2 --db {params.db} --report {output.report} \
                    --output {output.classified} --threads {threads} {input.r1}
            """

    rule vg_giraffe_se:
        input:
            r1="results/clean_R1.fastq"
        output:
            gam="results/aligned.gam"
        params:
            graph="../resources/vg/ecoli_graph_test.d2.gbz",
            dist="../resources/vg/ecoli_graph_test.d2.dist",
            min="../resources/vg/ecoli_graph_test.d2.min"
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            vg giraffe -Z {params.graph} -f {input.r1} \
                       -d {params.dist} -m {params.min} -t {threads} > {output.gam}
            """

    rule vg_surject_se:
        input:
            gam="results/aligned.gam"
        output:
            bam="results/aligned.bam"
        params:
            graph="../resources/vg/ecoli_graph_test.d2.gbz",
            path="GCF_000005845_2_ASM584v2_genomic#0#NC_000913.3"
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            vg surject -x {params.graph} -b -p {params.path} -t {threads} {input.gam} > {output.bam}
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
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            fastp -i {input.r1} -I {input.r2} \
                  -o {output.r1} -O {output.r2} \
                  -h {output.html} -j {output.json} -w {threads}
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
        conda: "../envs/rnaseq_vg.yaml"
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
            db="../resources/kraken2-db"
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            mkdir -p results/kraken
            kraken2 --db {params.db} --report {output.report} \
                    --output {output.classified} --threads {threads} \
                    --paired {input.r1} {input.r2}
            """

    rule vg_giraffe_pe:
        input:
            r1="results/clean_R1.fastq",
            r2="results/clean_R2.fastq"
        output:
            gam="results/aligned.gam"
        params:
            graph="../resources/vg/ecoli_graph_test.d2.gbz",
            dist="../resources/vg/ecoli_graph_test.d2.dist",
            min="../resources/vg/ecoli_graph_test.d2.min"
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            vg giraffe -Z {params.graph} -f {input.r1} -f {input.r2} \
                       -d {params.dist} -m {params.min} -t {threads} > {output.gam}
            """

    rule vg_surject_pe:
        input:
            gam="results/aligned.gam"
        output:
            bam="results/aligned.bam"
        params:
            graph="../resources/vg/ecoli_graph_test.d2.gbz",
            path="GCF_000005845_2_ASM584v2_genomic#0#NC_000913.3"
        threads: 12
        conda: "../envs/rnaseq_vg.yaml"
        shell:
            """
            vg surject -x {params.graph} -b -p {params.path} -t {threads} {input.gam} > {output.bam}
            """

########################## VG Stats Rule
rule vg_stats:
    input:
        gam="results/aligned.gam"
    output:
        txt="results/vg/stats.txt"
    conda: "../envs/rnaseq_vg.yaml"
    shell:
        """
        mkdir -p results/vg
        vg stats -a {input.gam} > {output.txt}
        """

########################## Shared Rules
multiqc_inputs = {
    "fastp": "results/fastp.json",
    "fastqc_r1": "results/fastqc/clean_R1_fastqc.zip",
    "kraken_report": "results/kraken/clean_R1_kraken_report.txt",
    "kraken_output": "results/kraken/clean_R1_kraken_output.txt",
    "vg_bam": "results/aligned.bam",
    "vg_stats": "results/vg/stats.txt"
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
    conda: "../envs/rnaseq_vg.yaml"
    shell:
        "multiqc results/ -o results"

