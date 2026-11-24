# Running Ecoli_inGEST on CyVerse (draft)

> This guide is a draft kept from the first release. The general install and run instructions are in
> [ecoli_ingest/README.md](../ecoli_ingest/README.md); this page only covers what is specific to a
> CyVerse virtual machine without conda: installing Apptainer without root and running through the
> container.

## Install Apptainer without root

```bash
sudo apt update
sudo apt install -y rpm cpio
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | bash -s - ./build/apptainer_local
./build/apptainer_local/bin/apptainer --version
export ECOLI_INGEST_APPTAINER="$PWD/build/apptainer_local/bin/apptainer"   # tells ecoli_ingest where it is
```

## Get the code and the launcher

```bash
git clone https://github.com/NCEMS/bacterial_states.git
cd bacterial_states
pip install ./ecoli_ingest          # or use bin/ecoli_ingest without installing
```

## Download the resources

```bash
ecoli_ingest download --with-apptainer      # core + centrifuge + the image of this version, checksums verified
```

The image (an asset of the code release of your version) is downloaded into
`resources/container/ecoli_ingest.sif`; run it with the Apptainer installed above (`ECOLI_INGEST_APPTAINER`).

## Run

```bash
mkdir ~/tutorial && cd ~/tutorial
ecoli_ingest init                    # then edit config.yaml with nano (ctrl+X, Y, Enter to save)
ecoli_ingest check --container always
ecoli_ingest run --container always --cores 12
```

`ecoli_ingest` binds the run directory, the resources and the directories holding your reads into
the container automatically. Samples given by `accession` are downloaded into `~/tutorial/sra/`
(inside the run directory, so nothing extra needs binding). Results appear in
`~/tutorial/<experiment>_results/`.
