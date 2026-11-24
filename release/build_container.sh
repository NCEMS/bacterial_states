#!/usr/bin/env bash
# Build the Apptainer image of ecoli_ingest or ecoli_atlas from the files git tracks in this checkout.
#
#   release/build_container.sh ecoli_ingest|ecoli_atlas [OUTDIR]        # OUTDIR default: work/containers
#
# The image is the environment of a major version, not of one code release: the package's locks/linux-64.txt
# (no solve) on a digest-pinned base, named by its environment id (CONTAINER_ENV in ecoli_ingest/resources.py,
# IMAGE_ENV in ecoli_atlas/download.py): OUTDIR/<package>_env<id>.sif and .sif.sha256. The lock must have the
# SHA-256 pinned beside the id (CONTAINER_LOCK_SHA256 / IMAGE_LOCK_SHA256), otherwise the build is refused: a changed
# lock needs a new environment id and a new image. The image is labelled with both (org.ncems.environment,
# org.ncems.lock-sha256), which `ecoli_ingest check` compares. It also holds the package at this commit
# (org.opencontainers.image.version / .revision), for `apptainer run` without a host install.
# A tree with uncommitted changes in the package is refused unless ALLOW_DIRTY=1 (the label then says +dirty).
# Every cache and temporary directory of the build goes under OUTDIR, never into $HOME. Needs git, GNU coreutils
# and apptainer (or singularity); root, fakeroot (an /etc/subuid entry) or SUDO_BUILD=1. No GPU is needed.
#
# Publishing (the maintainer): attach the image to the release containers-v<major> (assets: at most 2 GiB),
#   gh release create containers-v<major> --title "Apptainer images, environment <major>" --notes ...   (once)
#   gh release upload containers-v<major> OUTDIR/<package>_env<id>.sif
# then pin the printed SHA-256 in ecoli_ingest/src/ecoli_ingest/resources.py (CONTAINER_SHA256) or
# ecoli_atlas/src/ecoli_atlas/download.py (IMAGE_SHA256). Never replace a published image (--clobber): releases pin it.
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
# the environment id and lock checksum pinned in the package
case "$pkg" in
  ecoli_ingest) pins="$root/ecoli_ingest/src/ecoli_ingest/resources.py"; envvar=CONTAINER_ENV; lockvar=CONTAINER_LOCK_SHA256 ;;
  ecoli_atlas) pins="$root/ecoli_atlas/src/ecoli_atlas/download.py"; envvar=IMAGE_ENV; lockvar=IMAGE_LOCK_SHA256 ;;
esac
env_id=$(sed -n "s/^$envvar = \"\(.*\)\"\$/\1/p" "$pins")
lock_pin=$(sed -n "s/^$lockvar = \"\([0-9a-f]*\)\".*/\1/p" "$pins")
lock_sha=$(sha256sum "$root/$pkg/locks/linux-64.txt" | cut -d' ' -f1)
[ -n "$env_id" ] && [ -n "$lock_pin" ] || { echo "no $envvar / $lockvar in $pins" >&2; exit 1; }
if [ "$lock_sha" != "$lock_pin" ]; then
  echo "$pkg/locks/linux-64.txt has SHA-256 $lock_sha, but environment $env_id pins $lock_pin ($lockvar in $pins)." >&2
  echo "A changed lock is a new environment: set a new $envvar and $lockvar = \"$lock_sha\", then build." >&2
  exit 1
fi
major=${env_id%%.*}
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

sif="$out/${pkg}_env${env_id}.sif"
# ecoli_atlas/build/ is pip's output folder (ignored), so that package keeps its definition in container/
def="$pkg/container/Singularity.def"; [ -f "$stage/$def" ] || def="$pkg/build/Singularity.def"
# unprivileged builds need fakeroot (a /etc/subuid entry); SUDO_BUILD=1 runs only the build step with sudo instead
# (a CI runner); as root neither is needed
if [ "$(id -u)" = 0 ]; then build=("$runtime" build)
elif [ "${SUDO_BUILD:-0}" = 1 ]; then build=(sudo -E "$runtime" build)
else build=("$runtime" build --fakeroot)
fi
echo "building $sif (environment $env_id, lock ${lock_sha:0:12}; $pkg $version, $revision) with ${build[*]}" >&2
(cd "$stage" && "${build[@]}" --force --build-arg VERSION="$version" --build-arg REVISION="$revision" \
   --build-arg ENVIRONMENT="$env_id" --build-arg LOCK_SHA256="$lock_sha" "$sif" "$def")
[ "${SUDO_BUILD:-0}" = 1 ] && sudo chown "$(id -u):$(id -g)" "$sif"
rm -rf "$stage"

sha=$(sha256sum "$sif" | cut -d' ' -f1)
echo "$sha  $(basename "$sif")" > "$sif.sha256"
echo "image  : $sif ($(du -h "$sif" | cut -f1))"
echo "sha256 : $sha"
bytes=$(stat -c %s "$sif")
[ "$bytes" -lt 2147483648 ] || echo "WARNING: $bytes bytes, over the 2 GiB limit of a GitHub release asset" >&2
echo "upload : gh release upload containers-v$major $sif      (then pin the sha256 in $(basename "$pins"))"
