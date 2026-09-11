"""Reproduce every required problem-3 solve, validation and delivery artifact."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "output" / "problem-3"


def run_python(script, *args):
    """Run a third-question entry point and propagate any failure."""
    command = [sys.executable, "-B", "-X", "utf8", str(HERE / script), *map(str, args)]
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    """Run sequentially; resume explicitly reuses existing named research cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Reuse existing case files; use only when code and inputs are unchanged")
    args = parser.parse_args()
    run_python("solve_problem3.py", "--audit")
    cases = [(f"space{n}", n, 1., "last") for n in [400, 800, 1600, 3200, 6400, 12800]]
    cases += [("production", 12800, .5, "last"),
              ("time12800quarter", 12800, .25, "last"),
              ("mean12800", 12800, .5, "mean30")]
    for name, n, factor, mode in cases:
        if args.resume and all((OUT / f"{name}{suffix}").exists()
                               for suffix in [".json", ".npz", "_event_seed.npz"]):
            print(f"Reusing explicitly requested case: {name}", flush=True)
            continue
        run_python("solve_problem3.py", "--case", name, "--n", n,
                   "--factor", factor, "--mode", mode)
    run_python("verify_problem3.py")
    run_python("deliver_problem3.py")
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    node = os.environ.get("NODE_EXE") or str(bundled / "node/bin/node.exe")
    if not Path(node).exists():
        node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for the verified Artifact Tool workbook exporter")
    if not (HERE / "node_modules").exists():
        modules = Path(os.environ.get("ARTIFACT_NODE_MODULES", str(bundled / "node/node_modules")))
        if not modules.exists():
            raise RuntimeError("Set ARTIFACT_NODE_MODULES to the installed @oai/artifact-tool package root")
        if os.name == "nt":
            subprocess.run(["powershell", "-NoProfile", "-Command",
                            "New-Item", "-ItemType", "Junction", "-Path", str(HERE / "node_modules"),
                            "-Target", str(modules)], check=True)
        else:
            (HERE / "node_modules").symlink_to(modules, target_is_directory=True)
    subprocess.run([node, str(HERE / "build_workbook.mjs")], cwd=ROOT, check=True)
    run_python("verify_problem3.py", "--workbook")
    print("Problem 3 complete: all numerical and workbook checks passed.", flush=True)


if __name__ == "__main__":
    main()
