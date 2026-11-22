import os
from pathlib import Path

import pytest

from ecoli_ingest import checks, container, paths, runconfig
from ecoli_ingest.cli import build_parser, _extra_snakemake_args


def test_packaged_workflow_present():
    assert paths.snakefile().is_file()
    assert (paths.scripts_dir() / "annotate_counts.py").is_file()
    assert paths.config_template().is_file()


def test_template_parses_and_is_single_end():
    cfg = runconfig.load_config(paths.config_template())
    ss = runconfig.samples(cfg)
    assert len(ss) == 4 and all(s.r2 is None for s in ss)
    paired, mixed = runconfig.is_paired(cfg)
    assert (paired, mixed) == (False, False)
    assert runconfig.centrifuge_enabled(cfg) and cfg["centrifuge"] is True
    assert runconfig.centrifuge_gate_enabled(cfg)
    # the template must not pin resource paths; those come from the resources directory
    assert not any(k in cfg for k in paths.RESOURCE_FILES)


def test_mixed_layout_detected():
    cfg = {"samples": {"a": {"r1": "a_1.fq", "r2": "a_2.fq"}, "b": {"r1": "b.fq"}}}
    assert runconfig.is_paired(cfg) == (False, True)


def test_samples_accept_lists_and_accessions():
    cfg = {"samples": {
        "one": {"r1": "a.fq.gz", "condition": "x"},
        "two": {"r1": ["l1.fq.gz", "l2.fq.gz"], "r2": ["l1_2.fq.gz", "l2_2.fq.gz"]},
        "acc": {"accession": "GSM1646283"},
        "accs": {"accession": ["SRR1", "SRR2"]},
    }}
    ss = {s.name: s for s in runconfig.samples(cfg)}
    assert ss["one"].r1 == ["a.fq.gz"] and ss["one"].r2 is None and ss["one"].accessions == []
    assert ss["two"].r1 == ["l1.fq.gz", "l2.fq.gz"] and ss["two"].r2 == ["l1_2.fq.gz", "l2_2.fq.gz"]
    assert ss["acc"].accessions == ["GSM1646283"] and ss["acc"].r1 == [] and ss["acc"].r2 is None
    assert ss["accs"].accessions == ["SRR1", "SRR2"]
    assert runconfig.uses_accessions(cfg) and not runconfig.uses_accessions({"samples": {"a": {"r1": "a.fq"}}})
    # accession samples are unknown until resolved, then they take part in the layout decision
    assert runconfig.is_paired(cfg) == (False, True)
    assert runconfig.is_paired(cfg, {"acc": True, "accs": True}) == (False, True)
    assert runconfig.is_paired({"samples": {"acc": {"accession": "SRR1"}}}, {"acc": True}) == (True, False)
    assert runconfig.is_paired({"samples": {"acc": {"accession": "SRR1"}}}) == (False, False)
    assert runconfig.sra_dir({}, Path("/run")) == Path("/run/sra")
    assert runconfig.sra_dir({"sra_dir": "/shared/sra"}, Path("/run")) == Path("/shared/sra")


@pytest.mark.parametrize("info, msg", [
    ({"r1": "a.fq", "accession": "SRR1"}, "not both"),
    ({"condition": "x"}, "needs an 'r1'"),
    ({"r1": ["a.fq", "b.fq"], "r2": "c.fq"}, "r1 has 2"),
    ("a.fq", "must be a mapping"),
])
def test_samples_reject_bad_entries(info, msg):
    with pytest.raises(ValueError, match=msg):
        runconfig.samples({"samples": {"s": info}})


def test_compute_binds_dedupes_and_drops_nested(tmp_path):
    top = tmp_path / "top"
    nested = top / "inner" / "deep"
    other = tmp_path / "other"
    nested.mkdir(parents=True)
    other.mkdir()
    f = other / "reads.fq"
    f.write_text("x")
    binds = container.compute_binds([nested, top, f, other, top, tmp_path / "missing"])
    assert binds == sorted([str(top.resolve()), str(other.resolve())], key=lambda p: (len(Path(p).parts), p)) or \
        set(binds) == {str(top.resolve()), str(other.resolve())}


def test_resolve_mode(monkeypatch):
    monkeypatch.setattr(container.shutil, "which", lambda name: "/bin/x")
    assert container.resolve_mode("auto") == "never"
    monkeypatch.setattr(container.shutil, "which", lambda name: None)
    assert container.resolve_mode("auto") == "always"
    assert container.resolve_mode("never") == "never"
    with pytest.raises(ValueError):
        container.resolve_mode("sometimes")


def test_snakemake_and_container_commands():
    inner = container.snakemake_command(snakefile=Path("/pkg/Snakefile"), rundir=Path("/run"),
                                        configfile=Path("/run/config.yaml"), resources=Path("/res"),
                                        cores="8", targets=["all"], dry_run=True, extra=["--unlock"])
    assert inner[:3] == ["snakemake", "--snakefile", "/pkg/Snakefile"]
    assert "--config" in inner and "resources_dir=/res" in inner
    assert "--resources" not in inner
    limited = container.snakemake_command(snakefile=Path("/pkg/Snakefile"), rundir=Path("/run"),
                                          configfile=Path("/run/config.yaml"), resources=Path("/res"),
                                          resource_limits={"download_slots": 4})
    assert limited[limited.index("--resources") + 1] == "download_slots=4"
    assert limited.index("--resources") < limited.index("--cores") < limited.index("all")
    assert inner[-2:] == ["all", "--unlock"] and "--dry-run" in inner  # targets precede pass-through args
    # finished steps are not redone because the workflow code or a parameter changed ...
    assert inner[inner.index("--rerun-triggers") + 1] == "mtime" and inner.index("--rerun-triggers") < inner.index("--cores")
    # ... unless the user names the triggers
    own = container.snakemake_command(snakefile=Path("/pkg/Snakefile"), rundir=Path("/run"),
                                      configfile=Path("/run/config.yaml"), resources=Path("/res"),
                                      extra=["--rerun-triggers", "mtime", "params"])
    assert own.count("--rerun-triggers") == 1 and own[-3:] == ["--rerun-triggers", "mtime", "params"]
    outer = container.container_command("apptainer", Path("/res/container/x.sif"), ["/pkg", "/run"], Path("/run"), inner)
    assert outer[:2] == ["apptainer", "exec"]
    assert outer.count("--bind") == 2
    assert outer[outer.index("--pwd") + 1] == "/run"
    assert outer[outer.index("/res/container/x.sif") + 1:] == inner


def test_cli_run_parses_remainder():
    args = build_parser().parse_args(["run", "-n", "--cores", "4", "--", "--unlock", "--rerun-triggers", "mtime"])
    assert args.dry_run and args.cores == "4"
    assert _extra_snakemake_args(args.snakemake_args) == ["--unlock", "--rerun-triggers", "mtime"]


def test_checks_report_missing_everything(tmp_path):
    cfg = {"samples": {"a": {"r1": "nope.fq"}}, "centrifuge_min_percentage": 30}
    result = checks.check_resources(cfg, None, tmp_path) + checks.check_run_dir(cfg, tmp_path, tmp_path / "config.yaml")
    statuses = {c.name: c.status for c in result}
    assert statuses["resources directory"] == checks.FAIL
    assert statuses["sample reads"] == checks.FAIL
    assert statuses["run directory writable"] == checks.OK


def test_checks_multi_file_and_accession_samples(tmp_path, monkeypatch):
    (tmp_path / "a.fq.gz").write_bytes(b"")
    (tmp_path / "b.fq.gz").write_bytes(b"")
    cfg = {"samples": {"two": {"r1": ["a.fq.gz", "b.fq.gz"]}, "acc": {"accession": "GSM1646283"}}}
    sra = paths.load_script("sra_reads")

    def fake_fetch(url, timeout=60, attempts=3):
        if "runinfo" in url:
            return "Run,LibraryLayout\nSRR1945181,PAIRED\n"
        return ("run_accession\tlibrary_layout\tfastq_ftp\tfastq_md5\tfastq_bytes\n"
                "SRR1945181\tPAIRED\tx/SRR1945181_1.fastq.gz;x/SRR1945181_2.fastq.gz\ta;b\t10;20\n")

    monkeypatch.setattr(paths, "load_script", lambda name: sra)
    monkeypatch.setattr(sra, "_fetch", fake_fetch)
    result = {c.name: c for c in checks.check_run_dir(cfg, tmp_path, tmp_path / "config.yaml")}
    assert result["sample reads"].status == checks.OK and "2 files" in result["sample reads"].detail
    assert result["sra accessions"].status == checks.OK
    assert "1 run(s), 2 file(s)" in result["sra accessions"].detail and "0.00 GB" in result["sra accessions"].detail
    assert (tmp_path / "sra" / "manifest" / "GSM1646283.json").exists()      # cache written in the run dir
    method = result["read download method"]                                    # files still to fetch: say how
    assert method.status in (checks.OK, checks.WARN) and ("sra-tools" in method.detail or "ENA" in method.detail)
    cfg["sra_method"] = "aspera"
    assert {c.name: c for c in checks.check_run_dir(cfg, tmp_path, tmp_path / "config.yaml")}["read download method"].status == checks.FAIL
    del cfg["sra_method"]
    assert result["read layout"].status == checks.WARN                         # two=SE, acc=PE -> mixed

    cfg["samples"]["acc"]["accession"] = "GSE67218"
    result = {c.name: c for c in checks.check_run_dir(cfg, tmp_path, tmp_path / "config.yaml")}
    assert result["sra accessions"].status == checks.FAIL and "study-level" in result["sra accessions"].detail

    cfg = {"samples": {"mix": {"r1": ["a.fq.gz", "c.fq"]}}}
    (tmp_path / "c.fq").write_bytes(b"")
    result = checks.check_run_dir(cfg, tmp_path, tmp_path / "config.yaml")
    assert any(c.name == "sample reads" and c.status == checks.FAIL and "gzipped and plain" in c.detail for c in result)


def test_cli_run_max_downloads_default():
    assert build_parser().parse_args(["run"]).max_downloads == 4
    assert build_parser().parse_args(["run", "--max-downloads", "2"]).max_downloads == 2


def test_container_image_legacy_layout(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "rnaseq_vg.sif").write_bytes(b"")
    img, legacy = paths.container_image(tmp_path)
    assert legacy and img.name == "rnaseq_vg.sif"
    (tmp_path / "container").mkdir()
    (tmp_path / "container" / "rnaseq_vg.sif").write_bytes(b"")
    img, legacy = paths.container_image(tmp_path)
    assert not legacy and img.parent.name == "container"


def test_centrifuge_on_unless_switched_off():
    assert runconfig.centrifuge_enabled({})                                  # old configs keep running it
    assert not runconfig.centrifuge_enabled({"centrifuge": False})
    assert not runconfig.centrifuge_gate_enabled({"centrifuge": False, "species_gate": True})
    assert runconfig.centrifuge_gate_enabled({})                              # the species gate is on with Centrifuge
    assert not runconfig.centrifuge_gate_enabled({"species_gate": False})


def _centrifuge_checks(cfg, tmp_path):
    res = tmp_path / "res"
    res.mkdir(exist_ok=True)
    return {c.name: c for c in checks.check_resources(cfg, res, tmp_path) if c.name.startswith("centrifuge")}


def test_checks_centrifuge_index(tmp_path):
    # enabled (even without a gate) needs the index
    assert _centrifuge_checks({}, tmp_path)["centrifuge index"].status == checks.FAIL
    cf = tmp_path / "res" / paths.RESOURCE_FILES["centrifuge_index_path"]
    cf.parent.mkdir(parents=True)
    for suffix in paths.CENTRIFUGE_SUFFIXES:
        Path(str(cf) + suffix).write_bytes(b"")
    assert _centrifuge_checks({}, tmp_path)["centrifuge index"].status == checks.OK
    # switched off: not needed; a leftover threshold is flagged
    off = _centrifuge_checks({"centrifuge": False}, tmp_path)
    assert off["centrifuge index"].status == checks.OK and "not needed" in off["centrifuge index"].detail
    assert "centrifuge gate" not in off
    gated = _centrifuge_checks({"centrifuge": False, "centrifuge_min_percentage": 30}, tmp_path)
    assert gated["centrifuge gate"].status == checks.WARN and "species gate" in gated["centrifuge gate"].detail


def test_init_without_centrifuge(tmp_path):
    from ecoli_ingest.cli import main
    assert main(["init", "-d", str(tmp_path)]) == 0
    assert runconfig.load_config(tmp_path / "config.yaml")["centrifuge"] is True
    assert main(["init", "-d", str(tmp_path), "--force", "--without-centrifuge"]) == 0
    cfg = runconfig.load_config(tmp_path / "config.yaml")
    assert cfg["centrifuge"] is False and not runconfig.centrifuge_enabled(cfg)
