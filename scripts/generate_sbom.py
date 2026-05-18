#!/usr/bin/env python3
"""Phase 12 S12.3 — SBOM generation (CycloneDX).

Produces a CycloneDX SBOM for the Python deps and (if Node is present) the
frontend deps. Tools are invoked only if installed — this script never
forces heavy deps into the runtime image (same posture as Presidio / CI):

    pip install cyclonedx-bom        # Python SBOM
    # npm sbom is built into npm >= 9 (no extra install)

    python scripts/generate_sbom.py --out sbom

Outputs sbom/python.cdx.json and sbom/frontend.cdx.json when the respective
toolchains are available; logs and skips otherwise.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _run(cmd: list[str], outfile: str) -> bool:
    try:
        with open(outfile, "wb") as fh:
            subprocess.run(cmd, check=True, stdout=fh)
        print(f"  wrote {outfile}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  skipped ({' '.join(cmd[:2])}…): {exc}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sbom")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Python SBOM:")
    if shutil.which("cyclonedx-py"):
        _run(["cyclonedx-py", "environment", "--of", "JSON"],
             os.path.join(args.out, "python.cdx.json"))
    else:
        print("  cyclonedx-py not installed (pip install cyclonedx-bom)")

    print("Frontend SBOM:")
    if shutil.which("npm") and os.path.isdir("frontend"):
        ok = _run(
            ["npm", "--prefix", "frontend", "sbom",
             "--sbom-format", "cyclonedx"],
            os.path.join(args.out, "frontend.cdx.json"),
        )
        if not ok:
            print("  (needs npm >= 9 and an installed frontend node_modules)")
    else:
        print("  npm or frontend/ not available")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
