"""Pure-Python build stage definitions.

Each stage is a ``Stage`` whose ``steps`` list is the authoritative sequence
of commands to run inside the container.  Shell globs and ``||`` operators are
expressed as ``["bash", "-c", "..."]`` steps; everything else is a plain argv.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field

from .config import Config


@dataclass
class Step:
    argv: list[str]
    workdir: str | None = None
    allow_failure: bool = False


@dataclass
class Stage:
    name: str
    steps: list[Step] = field(default_factory=list)


def install_dependencies_stage(cfg: Config) -> Stage:
    return Stage(
        name="install_dependencies",
        steps=[
            Step(["mkdir", "-p", cfg.container_log_dir]),
            Step(
                [
                    "bash",
                    "-c",
                    "if grep -q '^MIRRORSITE=' /etc/pbuilderrc; then "
                    "sudo sed -i "
                    "'s|^MIRRORSITE=.*|MIRRORSITE=http://archive.ubuntu.com/ubuntu|' "
                    "/etc/pbuilderrc; "
                    "else echo 'MIRRORSITE=http://archive.ubuntu.com/ubuntu' "
                    "| sudo tee -a /etc/pbuilderrc; fi",
                ],
                workdir="/root",
            ),
            Step(
                ["bash", "-c", "DEBIAN_FRONTEND=noninteractive sudo apt update"],
                workdir="/root",
            ),
            Step(
                [
                    "bash",
                    "-c",
                    "DEBIAN_FRONTEND=noninteractive sudo apt install -y"
                    " devscripts git-buildpackage equivs python3-venv"
                    " default-jdk javahelper dh-python quilt",
                ],
                workdir="/root",
            ),
            *(
                [
                    Step(
                        [
                            "bash",
                            "-c",
                            "DEBIAN_FRONTEND=noninteractive sudo apt install -y ccache",
                        ],
                        workdir="/root",
                    )
                ]
                if cfg.ccache_host_dir
                else []
            ),
        ],
    )


def prepare_tarball_stage(cfg: Config) -> Stage:
    ceph_workdir = cfg.container_workdir  # /root/ceph
    return Stage(
        name="prepare_tarball",
        steps=[
            Step(["mkdir", "-p", cfg.container_log_dir]),
            # Clone upstream and build the source tarball.
            Step(["git", "clone", "https://github.com/ceph/ceph"], workdir="/root"),
            Step(["git", "checkout", f"v{cfg.ceph_version}"], workdir=ceph_workdir),
            Step(["./make-dist"], workdir=ceph_workdir),
            # Shell glob required to find the generated .bz2.
            Step(
                ["bash", "-c", "mv ceph/ceph*.bz2 ceph-tarball.tar.bz2"],
                workdir="/root",
            ),
            Step(["rm", "-rf", "ceph"], workdir="/root"),
            # Clone the Launchpad packaging repo.
            Step(
                [
                    "git",
                    "clone",
                    f"git://git.launchpad.net/~{cfg.launchpad_owner}/ubuntu/+source/ceph",
                ],
                workdir="/root",
            ),
            Step(
                [
                    "git",
                    "remote",
                    "add",
                    "source",
                    "git://git.launchpad.net/ubuntu/+source/ceph",
                ],
                workdir=ceph_workdir,
            ),
            # Both checkouts in one shell invocation because they must be sequential.
            Step(
                ["bash", "-c", "git checkout upstream && git checkout pristine-tar"],
                workdir=ceph_workdir,
            ),
            Step(["git", "fetch", "source"], workdir=ceph_workdir),
            Step(
                [
                    "git",
                    "checkout",
                    "-B",
                    cfg.ubuntu_branch,
                    f"origin/{cfg.ubuntu_branch}",
                ],
                workdir=ceph_workdir,
            ),
            Step(
                [
                    "gbp",
                    "import-orig",
                    "--no-interactive",
                    "--merge-mode=replace",
                    "../ceph-tarball.tar.bz2",
                    "-u",
                    cfg.ceph_version,
                ],
                workdir=ceph_workdir,
            ),
            Step(
                ["bash", "-c", "rm *.buildinfo || true"],
                workdir=ceph_workdir,
            ),
            Step(
                ["git", "checkout", f"upstream/{cfg.ceph_version}"],
                workdir=ceph_workdir,
            ),
            Step(["git", "checkout", "-b", "build"], workdir=ceph_workdir),
            Step(
                ["git", "checkout", cfg.debian_ref, "--", "debian"],
                workdir=ceph_workdir,
            ),
            Step(
                ["bash", "-c", "git rm debian/compat || true"],
                workdir=ceph_workdir,
            ),
            Step(
                ["git", "commit", "-m", "add debian directory"],
                workdir=ceph_workdir,
            ),
        ],
    )


def install_build_requirements_stage(cfg: Config) -> Stage:
    ceph_workdir = cfg.container_workdir
    return Stage(
        name="install_build_requirements",
        steps=[
            Step(["mkdir", "-p", cfg.container_log_dir]),
            Step(
                [
                    "bash",
                    "-c",
                    # Enable deb-src so mk-build-deps can resolve build
                    # dependencies. The location differs by release:
                    #   - Noble (24.04+) uses the deb822 format in
                    #     /etc/apt/sources.list.d/ubuntu.sources, where source
                    #     repos are toggled via the "Types:" field.
                    #   - Jammy (22.04) uses the one-line format in
                    #     /etc/apt/sources.list, with deb-src lines commented
                    #     out by default.
                    "set -e; "
                    "if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then "
                    "  sudo sed -i 's/^Types: deb$/Types: deb deb-src/'"
                    " /etc/apt/sources.list.d/ubuntu.sources; "
                    "elif [ -f /etc/apt/sources.list ]; then "
                    "  sudo sed -Ei 's/^# ?deb-src/deb-src/'"
                    " /etc/apt/sources.list; "
                    "else "
                    "  echo 'no apt sources file found to enable deb-src' >&2;"
                    " exit 1; "
                    "fi",
                ],
                workdir=ceph_workdir,
            ),
            Step(
                [
                    "bash",
                    "-c",
                    "DEBIAN_FRONTEND=noninteractive sudo mk-build-deps -i"
                    " -t 'apt-get -o Debug::pkgProblemResolver=1 -y --no-install-recommends'"
                    " debian/control",
                ],
                workdir=ceph_workdir,
            ),
            Step(
                ["bash", "-c", "rm *.buildinfo *.changes *.deb || true"],
                workdir=ceph_workdir,
            ),
        ],
    )


CONTAINER_CCACHE_DIR = "/root/ccache"

# Shell script that resets the upstream source tree to a clean HEAD baseline.
# Used both at the start of the build stage and during the preflight patch
# walk so both see the same state that dpkg-source would.
#
# Steps in order:
#   1. quilt pop -a  — revert whatever quilt successfully applied (.pc/ may
#                       be present from a previous failed push).
#   2. rm -rf .pc    — drop quilt's bookkeeping directory entirely.
#   3. git checkout HEAD -- with pathspec-exclude — restore every tracked
#                       file outside debian/ to its committed (= tarball) state.
#   4. git clean -fd — remove untracked files outside debian/ (.rej, .orig,
#                       stale build artefacts from a partial compilation).
#
# Every command is || true so the reset is best-effort: if quilt isn't
# installed, or there's nothing to clean, we still proceed.
TREE_RESET_SCRIPT = (
    "quilt pop -a 2>/dev/null || true; "
    "rm -rf .pc; "
    "git checkout HEAD -- . ':(exclude)debian' 2>/dev/null || true; "
    "git clean -fd -e debian/ 2>/dev/null || true; "
    "true"
)


def build_stage(cfg: Config) -> Stage:
    return Stage(
        name="build",
        steps=[
            Step(["mkdir", "-p", cfg.container_log_dir]),
            # Reset to a clean upstream tree before every debuild attempt.
            # See TREE_RESET_SCRIPT above for the full rationale.
            Step(["bash", "-c", TREE_RESET_SCRIPT], workdir=cfg.container_workdir),
            Step(
                ["bash", "-c", _debuild_cmd(cfg)],
                workdir=cfg.container_workdir,
            ),
        ],
    )


def _debuild_cmd(cfg: Config) -> str:
    base = "debuild --no-lintian -us -uc -d -b -j$(nproc)"
    if not cfg.ccache_host_dir:
        return base
    # debuild scrubs its environment and resets PATH before running the build,
    # so a plain "PATH=/usr/lib/ccache:$PATH CCACHE_DIR=... debuild" never
    # reaches the compiler — cmake detects /usr/bin/c++ and ccache is bypassed
    # (zero cache stats). Use debuild's own knobs: --prepend-path puts the
    # ccache shims first so cmake's compiler detection picks /usr/lib/ccache/c++,
    # and --preserve-envvar passes the CCACHE_* settings through to the build.
    # ccache_max_size is already format-validated in config.load(); shlex.quote
    # here is belt-and-suspenders so this interpolation stays shell-safe even
    # if the value ever reaches us unvalidated.
    return (
        f"CCACHE_DIR={CONTAINER_CCACHE_DIR} "
        f"CCACHE_BASEDIR={cfg.container_workdir} "
        f"CCACHE_MAXSIZE={shlex.quote(cfg.ccache_max_size)} "
        f"debuild --prepend-path=/usr/lib/ccache "
        f"--preserve-envvar='CCACHE_*' "
        f"--no-lintian -us -uc -d -b -j$(nproc)"
    )
