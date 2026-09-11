"""Capture and check output-only changes without recomputing the drying solution."""

from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import types
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src/problem-3"
OUT = ROOT / "output/problem-3"
REVIEW = OUT / "review"
BASELINE = REVIEW / "presentation_before.json"
SOURCE_NAMES = ("solve_problem3.py", "verify_problem3.py", "deliver_problem3.py",
                "generate_figures.py", "ensemble_problem3.py", "run_all.py", "build_workbook.mjs")
CORE_NAMES = ("properties", "Solver", "Environment", "plateau_fluctuation_stats",
              "simulate", "refine_event", "compare", "event_checks", "integrate_local",
              "synthetic_tests", "boundary_metrics")


def digest(path):
    """Hash a non-image file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def core_signatures(source):
    """Fingerprint numerical code independently of output directory constants."""
    return {node.name: hashlib.sha256(ast.dump(node).encode()).hexdigest()
            for node in ast.parse(source).body if getattr(node, "name", None) in CORE_NAMES}


def capture():
    """Preserve source and numerical-file fingerprints before presentation edits."""
    if BASELINE.exists():
        raise RuntimeError("Presentation baseline already exists; do not replace it")
    files = [p for p in OUT.iterdir() if p.is_file() and p.suffix in {".npz", ".json", ".csv", ".xlsx"}]
    sources = {name: (SRC / name).read_text(encoding="utf-8") for name in SOURCE_NAMES}
    record = {"sources": sources, "numerical_sha256": {p.name: digest(p) for p in files},
              "core": core_signatures(sources["solve_problem3.py"])}
    REVIEW.mkdir(exist_ok=True)
    BASELINE.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Captured {len(files)} numerical artifacts and {len(record['core'])} numerical components.")


def module_from_source(name, source, filename):
    """Load a baseline plotting module in memory without invoking its entry point."""
    module = types.ModuleType(name)
    module.__file__ = str(SRC / filename)
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def figure_record(fig):
    """Capture plotted numeric geometry and all user-facing figure text."""
    from matplotlib.text import Text
    numeric = []
    for axis in fig.axes:
        numeric.append({"limits": [axis.get_xlim(), axis.get_ylim()],
                        "lines": [(np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist())
                                  for line in axis.lines],
                        "patches": [p.get_path().vertices.tolist() for p in axis.patches]})
    text = [t.get_text() for t in fig.findobj(Text) if t.get_text()]
    return numeric, text


def verify():
    """Compare plot geometry and numerical hashes, and exercise Chinese renderers."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    sys.path.insert(0, str(SRC))
    import solve_problem3 as solver
    report = {"core_unchanged": baseline["core"] == core_signatures((SRC / "solve_problem3.py").read_text(encoding="utf-8"))}
    unchanged = {}
    for name, expected in baseline["numerical_sha256"].items():
        candidates = [p for p in OUT.rglob(name) if p.is_file()]
        unchanged[name] = len(candidates) == 1 and digest(candidates[0]) == expected
    report["numeric_artifacts_unchanged"] = unchanged
    values = np.loadtxt(solver.AUDIT_DIR / "observations_readonly_copy.csv", delimiter=",", skiprows=1)
    verification = json.loads((solver.VALIDATION_DIR / "verification.json").read_text(encoding="utf-8"))
    data = np.load(solver.CASES_DIR / "production.npz")
    mean = np.load(solver.CASES_DIR / "mean12800.npz")
    with TemporaryDirectory(prefix=".plot-check-", dir=REVIEW) as cache, patch.dict(os.environ, {"MPLCONFIGDIR": cache}):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from plot_style import configure_chinese
        report["font"] = configure_chinese(plt)
        plot_checks = {}

        def run_figures(module, which):
            """Capture figures without reading or writing output PNG files."""
            captured = []

            def save(fig, *args, **kwargs):
                """Render the figure in memory and record its content."""
                fig.canvas.draw()
                captured.append(figure_record(fig))

            with patch.object(Figure, "savefig", save):
                if which in ("generate_figures.py", "deliver_problem3.py"):
                    if hasattr(module, "_import_matplotlib"):
                        with patch.object(module, "_import_matplotlib", return_value=plt):
                            module.figures(data, mean, verification, values)
                    else:
                        module.figures(data, mean, verification, values)
                elif which == "distribution":
                    with patch.object(module, "_import_matplotlib", return_value=plt):
                        configure_chinese(plt)
                        # Only a renderer fixture; this is not an ensemble calculation.
                        module.plot_distribution(np.array([57., 57.2, 57.4]), 57.1, types.SimpleNamespace())
                else:
                    with patch.object(module, "_import_matplotlib", return_value=plt):
                        configure_chinese(plt)
                        with patch.object(module, "Environment", return_value=lambda t: np.array([50., .05])):
                            if which == "paths":
                                module.plot_plateau_paths(values, types.SimpleNamespace(tau=None, sigma_scale=1.), [0, 1])
                            else:
                                module.stochastic_air_figures(values, [0], None, 1., 14520.)
            plt.close("all")
            return captured

        for name in ["generate_figures.py", "deliver_problem3.py", "ensemble_problem3.py"]:
            old = module_from_source("before_" + name[:-3], baseline["sources"][name], name)
            old.OUT = Path(cache)
            new = importlib.import_module(name[:-3])
            modes = [name, "seed_air"] if name == "generate_figures.py" else [name] if name == "deliver_problem3.py" else ["distribution", "paths"]
            for mode in modes:
                with plt.rc_context(), patch("warnings.warn"):
                    old_plots = run_figures(old, mode)
                with plt.rc_context():
                    configure_chinese(plt)
                    new_plots = run_figures(new, mode)
                # Writers may add JPEG previews, so compare the unique figure records.
                old_numeric = list(dict.fromkeys(json.dumps(p[0], sort_keys=True) for p in old_plots))
                new_numeric = list(dict.fromkeys(json.dumps(p[0], sort_keys=True) for p in new_plots))
                texts = sorted(set(t for _, labels in new_plots for t in labels))
                untranslated = [t for t in texts if re.search(r"[A-Za-z]{3,}", t)
                                and not re.fullmatch(r"\$.*\$", t)]
                plot_checks[mode] = {"geometry_unchanged": old_numeric == new_numeric,
                                     "untranslated_english": untranslated, "texts": texts}
            if hasattr(old, "_PLOT_CACHE"):
                old._PLOT_CACHE.cleanup()
            if hasattr(new, "_PLOT_CACHE"):
                new._PLOT_CACHE.cleanup()
        report["plot_checks"] = plot_checks
    data.close()
    mean.close()
    report["full_solver_rerun"] = "SKIPPED: numerical logic unchanged"
    report["stochastic_renderer_scope"] = "In-memory plotting fixtures only; no ensemble result generated"
    report["status"] = "PASS" if report["core_unchanged"] and all(unchanged.values()) and all(
        item["geometry_unchanged"] and not item["untranslated_english"] for item in plot_checks.values()) else "FAIL"
    (REVIEW / "presentation_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "font": report["font"],
                      "core_unchanged": report["core_unchanged"],
                      "changed_numeric_artifacts": [k for k, v in unchanged.items() if not v],
                      "plots": {k: {key: v for key, v in value.items() if key != "texts"} for k, value in plot_checks.items()}},
                     ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Presentation verification failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    capture() if args.capture else verify()
