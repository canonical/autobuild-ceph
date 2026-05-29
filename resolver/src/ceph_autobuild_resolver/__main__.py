"""Entry point for ``python -m ceph_autobuild_resolver``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
