#!/usr/bin/env bash
# Build the Apptainer image of ecoli_ingest or ecoli_atlas from the files git tracks in this checkout.
#
#   release/build_container.sh ecoli_ingest|ecoli_atlas [OUTDIR]        # OUTDIR default: work/containers
#
# Writes OUTDIR/<package>_<version>.sif and .sif.sha256 and prints the upload command. The image installs the
# package's locks/linux-64.txt (no solve) on a digest-pinned base, so the same commit gives the same package set.
# A tree with uncommitted changes in the package is refused unless ALLOW_DIRTY=1 (the label then says +dirty).
# Every cache and temporary directory of the build goes under OUTDIR, never into $HOME.
#
# Publishing (the maintainer): attach the image to the code release v<version> (release assets: at most 2 GiB),
#   gh release upload v<version> OUTDIR/<package>_<version>.sif     (or the release's web page)
# then pin the printed SHA-256 in ecoli_ingest/src/ecoli_ingest/resources.py (CONTAINER_SHA256) or
# ecoli_atlas/src/ecoli_atlas/download.py (IMAGE_SHA256).
set -euo pipefail

pkg=${1:-}
case "$pkg" in
  ecoli_ingest|ecoli_atlas) ;;
  *) echo "usage: $0 ecoli_ingest|ecoli_atlas [OUTDIR]" >&2; exit 2 ;;
esac
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=$(mkdir -p "${2:-$root/work/containers}" && cd "${2:-$root/work/containers}" && pwd)
runtime=${ECOLI_INGEST_APPTAINER:-$(command -v apptainer || command -v singularity || true)}
[ -n "$runtime" ] || { echo "no apptainer/singularity on PATH" >&2; exit 1; }

version=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' "$root/$pkg/src/$pkg/__init__.py")
revision=$(git -C "$root" rev-parse --short HEAD)
if [ -n "$(git -C "$root" status --porcelain -- "$pkg")" ]; then
  if [ "${ALLOW_DIRTY:-0}" != 1 ]; then
    echo "$pkg has uncommitted changes; commit them or set ALLOW_DIRTY=1 for a test build" >&2
    exit 1
  fi
  revision="$revision+dirty"
fi

# the package as git sees it (tracked and new files, nothing ignored: no data, caches or run outputs)
stage="$out/stage_$pkg"
rm -rf "$stage"
mkdir -p "$stage"
(cd "$root" && git ls-files -z --cached --others --exclude-standard -- "$pkg" | tar --null -T - -cf -) | tar -xf - -C "$stage"
# the image has no git: the package records its commit from this file (ecoli_ingest: tool_versions.json, QC records)
echo "$revision" > "$stage/$pkg/src/$pkg/REVISION"

export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$out/cache} SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-$out/cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$out/tmp} SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-$out/tmp}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

sif="$out/${pkg}_${version}.sif"
# ecoli_atlas/build/ is pip's output folder (ignored), so that package keeps its definition in container/
def="$pkg/container/Singularity.def"; [ -f "$stage/$def" ] || def="$pkg/build/Singularity.def"
# unprivileged builds need fakeroot (a /etc/subuid entry); SUDO_BUILD=1 runs only the build step with sudo instead
# (a CI runner); as root neither is needed
if [ "$(id -u)" = 0 ]; then build=("$runtime" build)
elif [ "${SUDO_BUILD:-0}" = 1 ]; then build=(sudo -E "$runtime" build)
else build=("$runtime" build --fakeroot)
fi
echo "building $sif ($pkg $version, $revision) with ${build[*]}" >&2
(cd "$stage" && "${build[@]}" --force --build-arg VERSION="$version" --build-arg REVISION="$revision" \
   "$sif" "$def")
[ "${SUDO_BUILD:-0}" = 1 ] && sudo chown "$(id -u):$(id -g)" "$sif"
rm -rf "$stage"

sha=$(sha256sum "$sif" | cut -d' ' -f1)
echo "$sha  $(basename "$sif")" > "$sif.sha256"
echo "image  : $sif ($(du -h "$sif" | cut -f1))"
echo "sha256 : $sha"
bytes=$(stat -c %s "$sif")
[ "$bytes" -lt 2147483648 ] || echo "WARNING: $bytes bytes, over the 2 GiB limit of a GitHub release asset" >&2
echo "upload : gh release upload v$version $sif"
