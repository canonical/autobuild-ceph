"""Pure-Python build stage definitions.

Each stage is a ``Stage`` whose ``steps`` list is the authoritative sequence
of commands to run inside the container.  Shell globs and ``||`` operators are
expressed as ``["bash", "-c", "..."]`` steps; everything else is a plain argv.
"""

from __future__ import annotations

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
            Step(["sudo", "apt", "update"], workdir="/root"),
            Step(
                [
                    "sudo",
                    "apt",
                    "install",
                    "-y",
                    "devscripts",
                    "git-buildpackage",
                    "equivs",
                    "python3-venv",
                    "default-jdk",
                    "javahelper",
                    "dh-python",
                    "quilt",
                ],
                workdir="/root",
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
                    "sudo",
                    "sed",
                    "-i",
                    "s/^Types: deb$/Types: deb deb-src/",
                    "/etc/apt/sources.list.d/ubuntu.sources",
                ],
                workdir=ceph_workdir,
            ),
            Step(
                [
                    "sudo",
                    "mk-build-deps",
                    "-i",
                    "-t",
                    "apt-get -o Debug::pkgProblemResolver=1 -y --no-install-recommends",
                    "debian/control",
                ],
                workdir=ceph_workdir,
            ),
            Step(
                ["bash", "-c", "rm *.buildinfo *.changes *.deb || true"],
                workdir=ceph_workdir,
            ),
        ],
    )


def build_stage(cfg: Config) -> Stage:
    return Stage(
        name="build",
        steps=[
            Step(["mkdir", "-p", cfg.container_log_dir]),
            # Reset to a clean upstream tree before every debuild attempt.
            #
            # dpkg-source rejects the build with "unexpected upstream changes"
            # if anything outside debian/ differs from the original tarball
            # without being covered by a series patch. A failed quilt push
            # from a previous attempt commonly leaves exactly that state:
            # files partially modified, .pc/ populated, .rej / .orig files
            # scattered around. We clean it up aggressively here:
            #
            #   * quilt pop -a   reverts whatever quilt successfully applied
            #   * rm -rf .pc     drops quilt's bookkeeping
            #   * git checkout HEAD -- with pathspec exclude restores every
            #     tracked file outside debian/ to its committed (= tarball)
            #     state, regardless of whether `git ls-files -m` saw it
            #   * git clean -fd  removes untracked files (.rej, .orig,
            #     stale build artefacts) outside debian/
            #
            # `|| true` on each step keeps the reset best-effort: if quilt
            # isn't installed, or there's nothing to clean, we still proceed.
            Step(
                [
                    "bash",
                    "-c",
                    "quilt pop -a 2>/dev/null || true; "
                    "rm -rf .pc; "
                    "git checkout HEAD -- . ':(exclude)debian' 2>/dev/null || true; "
                    "git clean -fd -e debian/ 2>/dev/null || true; "
                    "true",
                ],
                workdir=cfg.container_workdir,
            ),
            Step(
                ["debuild", "--no-lintian", "-us", "-uc", "-d", "-b"],
                workdir=cfg.container_workdir,
            ),
        ],
    )
