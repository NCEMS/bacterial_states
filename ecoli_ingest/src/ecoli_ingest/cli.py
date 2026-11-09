"""Command-line entry point: `ecoli_ingest <subcommand>`.

The run directory is always the current directory (or --directory). The workflow files ship
inside this package, the resources live in a separate directory, and nothing has to be run from
inside the git checkout.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, checks, container, paths, resources, runconfig


def _err(msg: str) -> None:
    print(f"ecoli_ingest: {msg}", file=sys.stderr)


def _rundir(args) -> Path:
    d = Path(getattr(args, "directory", None) or ".").expanduser()
    return Path(os.path.realpath(d))


def _configfile(args, rundir: Path) -> Path:
    c = Path(args.config).expanduser() if getattr(args, "config", None) else rundir / "config.yaml"
    return Path(os.path.realpath(c)) if c.exists() else c


def _load(rundir: Path, configfile: Path) -> dict | None:
    if not configfile.exists():
        _err(f"no config file at {configfile}. Run `ecoli_ingest init` in your run directory first.")
        return None
    try:
        return runconfig.load_config(configfile)
    except Exception as e:  # yaml errors, structure errors
        _err(f"could not read {configfile}: {e}")
        return None


def cmd_init(args) -> int:
    rundir = _rundir(args)
    target = rundir / "config.yaml"
    if target.exists() and not args.force:
        _err(f"{target} already exists (use --force to overwrite)")
        return 1
    text = paths.config_template().read_text(encoding="utf-8")
    if args.without_centrifuge:
        assert text.count("\ncentrifuge: true\n") == 1, "config template must hold one `centrifuge: true` line"
        text = text.replace("\ncentrifuge: true\n", "\ncentrifuge: false\n")
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}\nEdit the samples and conditions, then run `ecoli_ingest check`.", file=sys.stderr)
    return 0


def cmd_check(args) -> int:
    rundir = _rundir(args)
    configfile = _configfile(args, rundir)
    cfg = _load(rundir, configfile)
    if cfg is None:
        return 1
    res = paths.find_resources(args.resources)
    mode = container.resolve_mode(args.container)
    print(f"execution mode: {mode} ({'conda environment' if mode == 'never' else 'container'})", file=sys.stderr)
    result = checks.run_all(cfg=cfg, rundir=rundir, configfile=configfile, resources=res, mode=mode)
    ok = checks.report(result)
    if ok and args.verify and res is not None:
        manifest = paths.packaged_manifest()
        if not manifest.exists():
            _err("no packaged manifest to verify against")
            return 1
        bad = [(r, s) for r, s in resources.verify(res, resources.read_manifest(manifest), log=sys.stderr) if s != "ok"]
        for rel, status in bad:
            print(f"[FAIL] {status:8s} {rel}", file=sys.stderr)
        ok = not bad
        print("checksums ok" if ok else f"{len(bad)} files failed verification", file=sys.stderr)
    return 0 if ok else 1


def _extra_snakemake_args(raw: list[str]) -> list[str]:
    return raw[1:] if raw and raw[0] == "--" else list(raw)


def cmd_run(args) -> int:
    rundir = _rundir(args)
    configfile = _configfile(args, rundir)
    cfg = _load(rundir, configfile)
    if cfg is None:
        return 1
    res = paths.find_resources(args.resources)
    mode = container.resolve_mode(args.container)
    if not args.skip_checks:
        result = checks.run_all(cfg=cfg, rundir=rundir, configfile=configfile, resources=res, mode=mode)
        if not checks.report(result):
            _err("checks failed; fix the problems above or pass --skip-checks")
            return 1
    assert res is not None

    extra = _extra_snakemake_args(args.snakemake_args)
    inner = container.snakemake_command(
        snakefile=paths.snakefile(), rundir=rundir, configfile=configfile, resources=Path(os.path.realpath(res)),
        cores=args.cores, targets=args.target, dry_run=args.dry_run, extra=extra,
        resource_limits={"download_slots": args.max_downloads})
    uses_sra = runconfig.uses_accessions(cfg)
    if uses_sra:
        # downloads land here; it must exist before the bind list is computed (missing paths are dropped)
        runconfig.sra_dir(cfg, rundir).mkdir(parents=True, exist_ok=True)

    if mode == "never":
        cmd = inner
    else:
        runtime = container.find_runtime()
        image, _legacy = paths.container_image(res)
        bind_sources: list = [paths.package_root(), rundir, res, configfile]
        for s in runconfig.samples(cfg):
            for f in s.r1 + (s.r2 or []):
                bind_sources.append(runconfig.resolve_path(f, rundir))
        if uses_sra:
            bind_sources.append(runconfig.sra_dir(cfg, rundir))
        bind_sources.extend(args.bind or [])
        binds = container.compute_binds(bind_sources)
        cmd = container.container_command(runtime, Path(os.path.realpath(image)), binds, rundir, inner)

    print("+ " + " ".join(_quote(c) for c in cmd), file=sys.stderr, flush=True)
    try:
        os.chdir(rundir)
        os.execvp(cmd[0], cmd)
    except OSError as e:
        _err(f"could not start {cmd[0]}: {e}")
        return 127


def cmd_shell(args) -> int:
    rundir = _rundir(args)
    res = paths.find_resources(args.resources)
    if res is None:
        _err("no resources directory found")
        return 1
    runtime = container.find_runtime()
    if runtime is None:
        _err("no container runtime found")
        return 1
    image, _ = paths.container_image(res)
    if not image.exists():
        _err(f"container image not found: {image}")
        return 1
    binds = container.compute_binds([paths.package_root(), rundir, res] + list(args.bind or []))
    cmd = container.container_command(runtime, Path(os.path.realpath(image)), binds, rundir, ["bash"])
    os.chdir(rundir)
    os.execvp(cmd[0], cmd)


def cmd_download(args) -> int:
    # an explicit --resources is the target even if it does not exist yet
    dest = Path(args.resources) if args.resources else (paths.find_resources() or paths.default_resources_target())
    unknown = [b for b in args.bundles if b not in resources.BUNDLES]
    if unknown:
        _err(f"unknown bundle(s) {', '.join(unknown)}; choose from {', '.join(sorted(resources.BUNDLES))}")
        return 2
    bundles = list(args.bundles or resources.DEFAULT_BUNDLES)
    if args.without_centrifuge and "centrifuge" in bundles:
        bundles.remove("centrifuge")
    if args.with_apptainer and "container" not in bundles:
        bundles.append("container")
    rc = 0
    for bundle in bundles:
        rc = max(rc, resources.download(bundle, Path(dest), dry_run=args.dry_run))
    return rc


def cmd_manifest(args) -> int:
    res = paths.find_resources(args.resources)
    if res is None:
        _err("no resources directory found")
        return 1
    rows = resources.generate_manifest(Path(res), log=sys.stderr)
    out = Path(args.out) if args.out else paths.packaged_manifest()
    resources.write_manifest(rows, out)
    print(f"wrote {len(rows)} entries to {out}", file=sys.stderr)
    return 0


def cmd_version(args) -> int:
    res = paths.find_resources(args.resources)
    print(f"ecoli_ingest {__version__}")
    print(f"workflow   : {paths.snakefile()}")
    print(f"resources  : {res if res else '(not found)'}")
    print(f"runtime    : {container.find_runtime() or '(none)'}")
    print(f"mode (auto): {container.resolve_mode('auto')}")
    return 0


def _quote(s: str) -> str:
    return s if all(ch.isalnum() or ch in "-_./=:+,@" for ch in s) else repr(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ecoli_ingest",
                                description="Ecoli_inGEST: run the E. coli RNA-seq workflow in the current directory.")
    p.add_argument("--resources", metavar="DIR",
                   help=f"resources directory (default: ${paths.ENV_RESOURCES}, ${paths.ENV_DATA_ROOT}/resources, "
                        "./resources, ~/.local/share/ecoli_ingest/resources)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="write a config.yaml template into the run directory")
    sp.add_argument("--directory", "-d", help="run directory (default: current directory)")
    sp.add_argument("--force", action="store_true", help="overwrite an existing config.yaml")
    sp.add_argument("--without-centrifuge", action="store_true",
                    help="write `centrifuge: false`: skip the contamination screen (no index needed)")
    sp.set_defaults(func=cmd_init)

    def run_common(sp):
        sp.add_argument("--directory", "-d", help="run directory (default: current directory)")
        sp.add_argument("--config", "-c", help="config file (default: <run dir>/config.yaml)")
        sp.add_argument("--container", choices=container.MODES, default="auto",
                        help="never = tools from the active environment; always = Apptainer image; "
                             "auto (default) = never if snakemake and vg are on PATH, else always")

    sp = sub.add_parser("check", help="check tools, resources, and the run configuration")
    run_common(sp)
    sp.add_argument("--verify", action="store_true", help="also verify resource checksums (slow)")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("run", help="run the workflow (extra arguments after -- go to snakemake)")
    run_common(sp)
    sp.add_argument("--cores", "-j", default="all", help="snakemake --cores (default: all)")
    sp.add_argument("--dry-run", "-n", action="store_true", help="show what would run")
    sp.add_argument("--target", action="append", default=None,
                    help="snakemake target (repeatable; default: all)")
    sp.add_argument("--bind", action="append", metavar="DIR", help="extra directory to bind into the container")
    sp.add_argument("--skip-checks", action="store_true", help="do not run `check` first")
    sp.add_argument("--max-downloads", type=int, default=4, metavar="N",
                    help="concurrent FASTQ downloads for samples given by accession (default: 4)")
    sp.add_argument("snakemake_args", nargs=argparse.REMAINDER,
                    help="additional snakemake arguments, e.g. -- --unlock")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("shell", help="open a shell inside the container with the run directory mounted")
    sp.add_argument("--directory", "-d", help="run directory (default: current directory)")
    sp.add_argument("--bind", action="append", metavar="DIR")
    sp.set_defaults(func=cmd_shell)

    sp = sub.add_parser("download", help="download resource bundles (default: core + centrifuge)")
    sp.add_argument("bundles", nargs="*", metavar="BUNDLE",
                    help="core = vg index + annotation; centrifuge = contamination index (from the Centrifuge "
                         "authors); container = Apptainer image. Default: " + " ".join(resources.DEFAULT_BUNDLES))
    sp.add_argument("--without-centrifuge", action="store_true",
                    help="skip the Centrifuge index (5.4 GB); for runs with `centrifuge: false`")
    sp.add_argument("--with-apptainer", action="store_true",
                    help="also download the Apptainer image (~5.3 GB); only needed for --container always/auto "
                         "without the conda environment")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("manifest", help="(maintainers) regenerate the resource checksum manifest")
    sp.add_argument("--out", help="output path (default: the packaged manifest)")
    sp.set_defaults(func=cmd_manifest)

    sp = sub.add_parser("version", help="show version, workflow location, resources and execution mode")
    sp.set_defaults(func=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "target", None) is None and hasattr(args, "target"):
        args.target = ["all"]
    return args.func(args)
