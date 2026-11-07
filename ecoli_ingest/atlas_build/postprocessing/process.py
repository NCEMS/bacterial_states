import os
import subprocess
import pandas as pd
import sys
import shutil
import time


#GSEs to ignore on next run
def get_finished_gses(file_path="../postprocessing_scripts/finished_null.txt"):
    finished_gses = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                finished_gses.add(line.strip().upper())
    return finished_gses

#Added wait to avoid issues with NCBI connection
def wait_for_files_to_appear(expected_fastq_paths, timeout=300, interval=5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        all_files_ready = True
        for fpath in expected_fastq_paths:
            if not os.path.exists(fpath):
                print(f"Waiting for: {os.path.basename(fpath)} (not found yet)", flush=True)
                all_files_ready = False
                break
            if os.path.getsize(fpath) == 0:
                print(f"Waiting for: {os.path.basename(fpath)} (0 bytes, flushing)", flush=True)
                all_files_ready = False
                break

        if all_files_ready:
            print(
                f"All {len(expected_fastq_paths)} FASTQ files for GSE are ready after "
                f"{int(time.time() - start_time)} seconds.",
                flush=True,
            )
            return True

        time.sleep(interval)

    print(f"TIMEOUT: Not all FASTQ files became ready within {timeout} seconds.", flush=True)
    return False


#Downloading FASTQ files using GSM ID
def download(gsm, data_dir_absolute):
    #Have to get SRR value from GSM value in order to use fasterq-dump
    cmd = f"esearch -db sra -query {gsm} | efetch -format runinfo | cut -d',' -f1 | grep SRR"
    try:
        srr_output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: E-utility search failed for {gsm} with exit code {e.returncode}. ",
            flush=True,
        )
        print(f"Stderr: {e.stderr}", flush=True)
        return [], []

    srr_ids = srr_output.strip().split("\n")

    sample_names = []
    downloaded_fastq_paths = []

    for srr in srr_ids:
        current_sample_name = f"{gsm}_{srr}"

        #Avoiding redownloading files if they already exist (for restarting runs)
        fq_se_path = os.path.join(data_dir_absolute, f"{current_sample_name}_1.fastq")
        fq_pe_r1_path = os.path.join(data_dir_absolute, f"{current_sample_name}_1.fastq")
        fq_pe_r2_path = os.path.join(data_dir_absolute, f"{current_sample_name}_2.fastq")

        if os.path.exists(fq_se_path) or (os.path.exists(fq_pe_r1_path) and os.path.exists(fq_pe_r2_path)):
            print(f"FASTQ file for {srr} already exists. Skipping download.", flush=True)
            sample_names.append(current_sample_name)
            if os.path.exists(fq_se_path):
                downloaded_fastq_paths.append(fq_se_path)
            if os.path.exists(fq_pe_r1_path):
                downloaded_fastq_paths.append(fq_pe_r1_path)
            if os.path.exists(fq_pe_r2_path):
                downloaded_fastq_paths.append(fq_pe_r2_path)
            continue

        print(f"Downloading and extracting {srr} to {current_sample_name}...", flush=True)
        
        #Downloading fastqs using SRR ID
        fasterq_cmd = f"fasterq-dump --split-files --skip-technical {srr} -O {data_dir_absolute} --threads 10"
        try:
            subprocess.run(fasterq_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"WARNING: fasterq-dump failed for {srr}. Skipping this sample.", flush=True)
            continue

        fq_base = os.path.join(data_dir_absolute, srr)
        fq1_raw = f"{fq_base}_1.fastq"
        fq2_raw = f"{fq_base}_2.fastq"
        fq_se_raw = f"{fq_base}.fastq"

        if os.path.exists(fq_se_raw):
            new_fq1 = os.path.join(data_dir_absolute, f"{current_sample_name}_1.fastq")
            os.rename(fq_se_raw, new_fq1)
            downloaded_fastq_paths.append(new_fq1)
        elif os.path.exists(fq1_raw):
            new_fq1 = os.path.join(data_dir_absolute, f"{current_sample_name}_1.fastq")
            os.rename(fq1_raw, new_fq1)
            downloaded_fastq_paths.append(new_fq1)

            if os.path.exists(fq2_raw):
                new_fq2 = os.path.join(data_dir_absolute, f"{current_sample_name}_2.fastq")
                os.rename(fq2_raw, new_fq2)
                downloaded_fastq_paths.append(new_fq2)
        else:
            print(f"WARNING: No expected FASTQ files found for {srr} after fasterq-dump.", flush=True)
            continue

        sample_names.append(current_sample_name)

        #Adding a pause for rate limits
        print("Pausing for 5 seconds", flush=True)
        time.sleep(5)

    return sample_names, downloaded_fastq_paths

#Used to make temporary directories for resources as we are running many slurm jobs at once
def prepare_resources(resource_dir, top_level_project_root):
    #Skipping if already exists
    if os.path.exists(resource_dir):
        print(f"[INFO] Reusing existing resource directory: {resource_dir}", flush=True)
        return

    print(f"[INFO] Creating resource directory: {resource_dir}", flush=True)
    os.makedirs(resource_dir, exist_ok=True)

    vg_index_base_host = os.path.join(top_level_project_root, "resources", "vg", "ecoli_graph_test")
    linear_base_host = os.path.join(top_level_project_root, "resources", "linear", "ecoli_linear")
    annotation_gff_host = os.path.join(
        top_level_project_root, "resources", "annotation", "GCF_000005845.2_ASM584v2_genomic.gff"
    )
    centrifuge_index_host = os.path.join(top_level_project_root, "resources", "centrifuge", "p_compressed+h+v")

    shutil.copytree(os.path.dirname(vg_index_base_host), os.path.join(resource_dir, "vg"))
    shutil.copytree(os.path.dirname(linear_base_host), os.path.join(resource_dir, "linear"))
    shutil.copytree(os.path.dirname(annotation_gff_host), os.path.join(resource_dir, "annotation"))
    shutil.copytree(os.path.dirname(centrifuge_index_host), os.path.join(resource_dir, "centrifuge"))

    print(f"[INFO] Resource directory populated successfully: {resource_dir}", flush=True)


#Running the snakemake workflow
def run_pipeline(df: pd.DataFrame, finished_gses: set, resource_dir: str):

    ##Actions per metadata file
    df["gse"] = df["gse"].astype(str).str.strip().str.upper()
    df = df.sort_values(by="gse")

    #Getting directories
    pipeline_root_dir = os.getcwd()
    top_level_project_root = os.path.dirname(pipeline_root_dir)
    sif_absolute_path_on_host = os.path.join(top_level_project_root, "resources", "build", "rnaseq_vg.sif")
    gse_runs_base_dir = os.path.join(pipeline_root_dir, "gse_runs2")
    os.makedirs(gse_runs_base_dir, exist_ok=True)

    #Making sure the resource files are available
    prepare_resources(resource_dir, top_level_project_root)
    vg_index_final = os.path.join(resource_dir, "vg", "ecoli_graph_test")
    linear_final = os.path.join(resource_dir, "linear", "ecoli_linear")
    annotation_gff_final = os.path.join(resource_dir, "annotation", "GCF_000005845.2_ASM584v2_genomic.gff")
    annotation_bed_final = os.path.join(resource_dir, "annotation", "GCF_000005845.2_ASM584v2_genomic.bed")
    centrifuge_index_final = os.path.join(resource_dir, "centrifuge", "p_compressed+h+v")

    ##Actions per experiment
    for gse_value, group_df in df.groupby("gse"):
        if gse_value in finished_gses:
            print(f"{gse_value} is in the finished list. Skipping.", flush=True)
            continue
        
        #Each experiment gets its own directory and snakemake run folder
        gse_run_specific_output_dir = os.path.join(gse_runs_base_dir, gse_value)
        if os.path.exists(gse_run_specific_output_dir):
            print(f"Directory for {gse_value} exists. Resuming...", flush=True)

        print(f"\nProcessing GSE: {gse_value}", flush=True)
        os.makedirs(gse_run_specific_output_dir, exist_ok=True)

        #Downloading data
        data_dir_absolute = os.path.join(top_level_project_root, "data", gse_value)
        os.makedirs(data_dir_absolute, exist_ok=True)
        gsm_srr_to_char_map = {}
        all_gse_fastq_paths = []
        for _, row in group_df.iterrows():
            gsm = row["gsm"]
            sample_names_for_gsm, downloaded_paths = download(gsm, data_dir_absolute)
            for sample_name in sample_names_for_gsm:
                gsm_srr_to_char_map[sample_name] = row["characteristics_ch1"]
            all_gse_fastq_paths.extend(downloaded_paths)
        #Issues with download
        if not gsm_srr_to_char_map:
            print(f"WARNING: No valid samples found for {gse_value}. Skipping.", flush=True)
            continue
        
        ##Making config file for this experiment
        #Used for labeling the conditions of samples for DEseq2 in config file
        unique_characteristics = sorted(list(set(gsm_srr_to_char_map.values())))
        char_to_condition_name = {char: f"group{i+1}" for i, char in enumerate(unique_characteristics)}
        #Creating config file using conditions from metadata
        sample_entries = []
        for sample_name, char_val in gsm_srr_to_char_map.items():
            condition_name = char_to_condition_name[char_val]
            entry = f"  {sample_name}:\n"

            r1_host_path = os.path.join(data_dir_absolute, f"{sample_name}_1.fastq")
            r1_rel_to_project_root = os.path.relpath(r1_host_path, top_level_project_root)
            r1_path_in_container = os.path.join("/mnt/project_root", r1_rel_to_project_root)
            entry += f"    r1: \"{r1_path_in_container}\"\n"

            r2_host_path = os.path.join(data_dir_absolute, f"{sample_name}_2.fastq")
            if os.path.exists(r2_host_path):
                r2_rel_to_project_root = os.path.relpath(r2_host_path, top_level_project_root)
                r2_path_in_container = os.path.join("/mnt/project_root", r2_rel_to_project_root)
                entry += f"    r2: \"{r2_path_in_container}\"\n"

            entry += f"    condition: \"{condition_name}\"\n"
            sample_entries.append(entry)

        samples_str = "\n".join(sample_entries)
        
        #Necessary paths for config file
        vg_index_in_container = os.path.join("/mnt/project_root", os.path.relpath(vg_index_final, top_level_project_root))
        linear_in_container = os.path.join("/mnt/project_root", os.path.relpath(linear_final, top_level_project_root))
        annotation_gff_in_container = os.path.join(
            "/mnt/project_root", os.path.relpath(annotation_gff_final, top_level_project_root)
        )
        annotation_bed_in_container = os.path.join(
            "/mnt/project_root", os.path.relpath(annotation_bed_final, top_level_project_root)
        )
        centrifuge_index_in_container = os.path.join(
            "/mnt/project_root", os.path.relpath(centrifuge_index_final, top_level_project_root)
        )
        scripts_dir_host_path = os.path.join(pipeline_root_dir, "scripts")
        scripts_dir_in_container = os.path.join("/mnt/project_root", os.path.relpath(scripts_dir_host_path, top_level_project_root))

        config_content = f"""experiment: "{gse_value}"

samples:
{samples_str}

deseq2:
  contrasts: []

vg_index: "{vg_index_in_container}"
ref: "GCF_000005845_2_ASM584v2_genomic#0#NC_000913.3"

annotation_gff: "{annotation_gff_in_container}"
annotation_bed: "{annotation_bed_in_container}"
centrifuge_index_path: "{centrifuge_index_in_container}"
scripts_dir: "{scripts_dir_in_container}"
centrifuge_organism: "Escherichia coli"
centrifuge_min_percentage: 30.0
fastp_min_kept_percentage: 75.0
"""
        config_file_path_for_storage = os.path.join(gse_run_specific_output_dir, f"{gse_value}_config.yaml")
        with open(config_file_path_for_storage, "w") as f:
            f.write(config_content)
            f.flush()
            os.fsync(f.fileno())

        ##Running snakemake using constructed config file
        print(f"Running Snakemake for {gse_value}...", flush=True)
        #Setting up apptainer information before running subprocess
        apptainer_project_bind_mount = f"{top_level_project_root}:/mnt/project_root"
        apptainer_bind_mounts = f"--bind \"{apptainer_project_bind_mount}\" --bind \"/tmp:/tmp\""
        snakefile_path_in_container = "/mnt/project_root/pipeline/Snakefile"
        config_file_path_in_container_mount = os.path.join(
            "/mnt/project_root",
            os.path.relpath(gse_run_specific_output_dir, top_level_project_root),
            f"{gse_value}_config.yaml",
        )
        #Snakemake commands
        unlock_cmd = (
            f"apptainer exec {apptainer_bind_mounts} "
            f"{sif_absolute_path_on_host} "
            f"snakemake --unlock "
            f"--snakefile {snakefile_path_in_container} "
            f"--configfile {config_file_path_in_container_mount} "
            f"--directory {gse_run_specific_output_dir} "
        )

        snakemake_run_cmd = (
            f"apptainer exec {apptainer_bind_mounts} "
            f"{sif_absolute_path_on_host} "
            f"snakemake --cores 10 --resources mem_mb=102400 "
            f"--rerun-incomplete " #Rerun any rules that were not completed previously in the last run due to walltime/memory issues
            f"--notemp " #Keep temporary files
            f"--rerun-triggers input " #Only trigger a re-run of a rule if the input is missing (helpful for restarting runs where they left off)
            f"--snakefile {snakefile_path_in_container} "
            f"--configfile {config_file_path_in_container_mount} "
            f"--directory {gse_run_specific_output_dir} "
            f"all"
        )

        if not wait_for_files_to_appear(all_gse_fastq_paths):
            print(f"ERROR: Not all FASTQ files for GSE {gse_value} were ready. Skipping.", flush=True)
            continue
        #Unlocking directory for when runs have to be restarted due to wall time
        print(f"Attempting to unlock directory for {gse_value}...", flush=True)
        try:
            subprocess.run(unlock_cmd, shell=True, check=True, cwd=top_level_project_root)
            print("Successfully unlocked working directory.", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"WARNING: Unlock command failed with exit code {e.returncode}.", flush=True)

        print(f"Executing main Snakemake command for {gse_value}...", flush=True)
        try:
            subprocess.run(snakemake_run_cmd, shell=True, check=True, cwd=top_level_project_root)
        except subprocess.CalledProcessError as e:
            print(f"[WARN] Snakemake failed for {gse_value} with exit code {e.returncode}", flush=True)

    print("\nAll GSEs processed. No run directories deleted.")

##Main function
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py <input_tsv_file> <resource_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    resource_dir = sys.argv[2]

    df = pd.read_csv(input_file, sep="\t")
    required_cols = {"gse", "gsm", "characteristics_ch1"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    finished_gses = get_finished_gses()
    run_pipeline(df, finished_gses, resource_dir)

