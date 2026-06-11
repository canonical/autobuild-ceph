"""Argparse entry point for the resolver.

Two subcommands:

* ``prep``: launch an LXD container from an Ubuntu image, run the prep stages
  (deps + tarball + build deps), and snapshot the result.
* ``resolve``: run the AI-driven resolution loop against a prepared container.
  Optionally re-runs prep first via ``--prep-image``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import config, orchestrator, prep


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ceph-autobuild-resolver",
        description="AI-driven Ceph build-failure resolution loop.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # prep
    # ------------------------------------------------------------------
    prep_p = sub.add_parser(
        "prep",
        help="Launch an LXD container from an image and snapshot a clean "
        "post-prep state.",
    )
    prep_p.add_argument(
        "--image",
        required=True,
        help="LXD image alias, e.g. 'ubuntu:24.04' or 'ubuntu-daily:resolute'.",
    )
    prep_p.add_argument("--container", required=True)
    prep_p.add_argument(
        "--snapshot",
        default="pristine",
        help="Snapshot name to assign after prep completes (default: pristine).",
    )
    prep_p.add_argument(
        "--force-relaunch",
        action="store_true",
        help="Delete the container if it already exists and start fresh.",
    )

    # ------------------------------------------------------------------
    # resolve
    # ------------------------------------------------------------------
    res_p = sub.add_parser(
        "resolve",
        help="Run the AI resolution loop against a prepared container.",
    )
    res_p.add_argument("--container", required=True)
    res_p.add_argument(
        "--pristine-snapshot",
        default="pristine",
        help="Snapshot to validate against (default: pristine).",
    )
    res_p.add_argument("--matrix-name", required=True)
    res_p.add_argument(
        "--transcript-path", default="resolver-transcript.jsonl"
    )
    res_p.add_argument(
        "--dry-run-output",
        action="store_true",
        help="Print PR/bug payload to stderr instead of contacting Launchpad.",
    )
    res_p.add_argument(
        "--prep-image",
        help="If set, run ``prep`` against this image before resolving.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # pylxd/ws4py log one INFO line per container exec; suppress them.
    logging.getLogger("ws4py").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    try:
        cfg = config.load()
    except config.ConfigError as exc:
        # ``prep`` doesn't need a model API key — fall back to a stub config
        # if the user is just prepping. We still need build.sh knobs which
        # come from env with sensible defaults.
        if args.command == "prep":
            cfg = _prep_only_config()
        else:
            print(f"config error: {exc}", file=sys.stderr)
            return 2

    if args.command == "prep":
        return _do_prep(cfg, args)
    if args.command == "resolve":
        return _do_resolve(cfg, args)
    return 2


def _do_prep(cfg, args) -> int:
    from .lxd import LXDManager

    lxd = LXDManager()
    if args.force_relaunch:
        logging.info("force-relaunch: deleting any existing %s", args.container)
        lxd.delete(args.container, force=True)

    outcome = prep.run(
        cfg=cfg,
        image=args.image,
        container=args.container,
        snapshot=args.snapshot,
        skip_existing=not args.force_relaunch,
        lxd=lxd,
    )
    print(
        f"prepared {outcome.container} from {outcome.image}; "
        f"snapshot={outcome.snapshot}",
        file=sys.stderr,
    )
    return 0


def _do_resolve(cfg, args) -> int:
    if args.prep_image:
        from .lxd import LXDManager

        lxd = LXDManager()
        prep.run(
            cfg=cfg,
            image=args.prep_image,
            container=args.container,
            snapshot=args.pristine_snapshot,
            skip_existing=True,
            lxd=lxd,
        )

    try:
        outcome = orchestrator.run(
            cfg=cfg,
            container=args.container,
            pristine_snapshot=args.pristine_snapshot,
            matrix_name=args.matrix_name,
            transcript_path=args.transcript_path,
            dry_run_output=args.dry_run_output,
        )
    except Exception:
        # A crash anywhere outside the loop (preflight, validation, LXD,
        # output generation) must still produce a CI failure artifact —
        # a raw traceback after hours of paid work records no outcome.
        import traceback

        from .output.ci_output import CIFailurePayload, emit_failure

        logging.exception("resolver crashed outside the resolution loop")
        tail = "\n".join(traceback.format_exc().splitlines()[-30:])
        emit_failure(
            CIFailurePayload(
                matrix_name=args.matrix_name,
                stop_reason="resolver_crash",
                failing_command="orchestrator",
                error_tail=tail,
                transcript_path=args.transcript_path,
            )
        )
        return 1
    print(outcome.description, file=sys.stderr)
    return outcome.exit_code


def _prep_only_config():
    """Synthesise a Config for ``prep`` without requiring a model API key."""
    return config.Config(
        provider="openrouter",
        model_name="unused",
        api_key="unused",
        max_iterations=0,
        max_unchanged_iterations=0,
        run_token_budget=0,
        ubuntu_branch=os.environ.get("UBUNTU_BRANCH", "ubuntu/resolute"),
        debian_ref=os.environ.get("DEBIAN_REF", "origin/ubuntu/latest"),
        launchpad_owner=os.environ.get("LAUNCHPAD_OWNER", "lmlogiudice"),
        ceph_version=os.environ.get("CEPH_VERSION", "20.2.0"),
        ccache_host_dir=os.environ.get("CCACHE_HOST_DIR") or None,
        ccache_max_size=os.environ.get("CCACHE_MAXSIZE") or "20G",
    )


if __name__ == "__main__":
    sys.exit(main())
