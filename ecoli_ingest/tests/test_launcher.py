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
    assert runconfig.centrifuge_enabled(cfg)
    # the template must not pin resource paths; those come from the resources directory
    assert not any(k in cfg for k in paths.RESOURCE_FILES)


def test_mixed_layout_detected():
    cfg = {"samples": {"a": {"r1": "a_1.fq", "r2": "a_2.fq"}, "b": {"r1": "b.fq"}}}
    assert runconfig.is_paired(cfg) == (False, True)


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
    assert inner[-2:] == ["all", "--unlock"] and "--dry-run" in inner  # targets precede pass-through args
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


def test_container_image_legacy_layout(tmp_path):
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "rnaseq_vg.sif").write_bytes(b"")
    img, legacy = paths.container_image(tmp_path)
    assert legacy and img.name == "rnaseq_vg.sif"
    (tmp_path / "container").mkdir()
    (tmp_path / "container" / "rnaseq_vg.sif").write_bytes(b"")
    img, legacy = paths.container_image(tmp_path)
    assert not legacy and img.parent.name == "container"
