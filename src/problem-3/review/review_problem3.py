"""Read-only review probes; all generated evidence stays under output/problem-3/review."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import types
from unittest.mock import patch
import warnings

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src/problem-3"
ORIGINAL_OUT = ROOT / "output/problem-3"
REVIEW = ORIGINAL_OUT / "review"
sys.path.insert(0, str(SRC))
import solve_problem3 as current


def load_module(name, path):
    """Import a review target without invoking its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint(paths):
    """Hash only the explicit non-PNG inputs and numerical artifacts."""
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def caught(action):
    """Record an actual exception or return, without treating a reproduced bug as PASS."""
    try:
        result = action()
        return {"exception": None, "return_type": type(result).__name__}
    except Exception as exc:
        return {"exception": type(exc).__name__, "message": str(exc)}


def main():
    """Run bounded numerical and control-flow probes without changing production files."""
    REVIEW.mkdir(exist_ok=True)
    inputs = [ROOT / "problem.pdf", ROOT / "data/附件1.xlsx",
              ROOT / "data/附件3/result3.xlsx", ROOT / "src/common/fvm.py"]
    inputs += list(SRC.glob("*.py")) + [SRC / "build_workbook.mjs", SRC / "模型与算法说明.md"]
    inputs += [p for folder in [ORIGINAL_OUT, current.AUDIT_DIR, current.CASES_DIR]
               for p in folder.iterdir()
               if p.is_file() and p.suffix.lower() in {".json", ".npz", ".xlsx", ".csv", ".md"}]
    inputs += [REVIEW / "verification.json", REVIEW / "workbook_verification.json"]
    before = fingerprint(inputs)
    result = {"scope": "Review only: short numerical runs and isolated fault probes",
              "source_sha256": fingerprint(list(SRC.glob("*.py")) + [SRC / "build_workbook.mjs"]),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                   text=True).strip()}
    wb = openpyxl.load_workbook(ROOT / "data/附件1.xlsx", read_only=True, data_only=True)
    rows = list(wb["Sheet1"].values)
    wb.close()
    values = np.asarray(rows[1:], dtype=float)
    audit_report = json.loads((current.AUDIT_DIR / "input_audit.json").read_text(encoding="utf-8"))
    result["observations"] = {"shape": list(values.shape), "header": list(rows[0]),
                              "finite": bool(np.isfinite(values).all()),
                              "missing": sum(v is None for row in rows[1:] for v in row),
                              "unique_times": int(len(np.unique(values[:, 0]))),
                              "intervals": np.unique(np.diff(values[:, 0])).tolist(),
                              "first": values[0].tolist(), "last": values[-1].tolist()}
    source = (SRC / "solve_problem3.py").read_text(encoding="utf-8")
    baseline_source = subprocess.check_output(
        ["git", "show", "HEAD:src/problem-3/solve_problem3.py"], cwd=ROOT).decode("utf-8")
    baseline = types.ModuleType("review_baseline")
    baseline.__file__ = str(SRC / "solve_problem3.py")
    exec(compile(baseline_source, baseline.__file__, "exec"), baseline.__dict__)
    nodes = {n.name: n for n in ast.parse(source).body if hasattr(n, "name")}
    old_nodes = {n.name: n for n in ast.parse(baseline_source).body if hasattr(n, "name")}
    result["core_ast_identical_to_HEAD"] = {
        name: ast.dump(nodes[name]) == ast.dump(old_nodes[name])
        for name in ["properties", "Solver", "refine_event"]}
    result["python_syntax"] = {p.name: "PASS" for p in SRC.glob("*.py")
                               if ast.parse(p.read_text(encoding="utf-8"))}
    result["synthetic_and_BDF"] = current.synthetic_tests(values)
    print("Short independent BDF and invariant checks completed.", flush=True)
    log = io.StringIO()
    with TemporaryDirectory(prefix="probes-", dir=REVIEW) as sandbox_name:
        sandbox = Path(sandbox_name)
        new_dir, old_dir = sandbox / "new", sandbox / "old"
        new_dir.mkdir()
        old_dir.mkdir()
        with patch.object(current, "CASES_DIR", new_dir), redirect_stdout(log):
            new_meta = current.simulate(400, .5, "last", "short", stop_s=60.,
                                        values=values, progress=False, verbose=False)
        with patch.object(baseline, "OUT", old_dir), patch.object(
                baseline, "audit", return_value=(values, audit_report)), redirect_stdout(log):
            baseline.simulate(400, .5, "last", "short", stop_s=60.)
        with np.load(new_dir / "short.npz") as new_data, np.load(old_dir / "short.npz") as old_data:
            result["deterministic_60s_regression"] = {
                "N": 400, "factor": .5,
                "full_state_max_abs_difference": float(np.max(abs(new_data["full_states"] - old_data["full_states"]))),
                "delivery_max_abs_difference": float(np.max(abs(new_data["outputs"] - old_data["outputs"]))),
                "mass_relative_defect": new_meta["mass_relative_defect"],
                "heat_relative_defect": new_meta["heat_relative_defect"]}

        result["new_case_loader_on_existing_tree"] = caught(lambda: current.load_case("production"))
        result["path_contract"] = {
            "new_case_dir": str(current.CASES_DIR), "new_delivery_dir": str(current.DELIVERY_DIR),
            "existing_case_in_cases": (current.CASES_DIR / "production.npz").exists(),
            "js_export_dir": "output/problem-3", "new_verifier_export_dir": str(current.DELIVERY_DIR.relative_to(ROOT)),
            "model_doc_actual": str(SRC / "模型与算法说明.md"),
            "model_doc_required_by_new_verifier": str(ROOT / "src/problem-3/模型与算法说明.md")}

        legacy = load_module("review_legacy_runner", SRC / "run_all.py")
        calls = []
        with patch.object(legacy, "run_python", side_effect=lambda *a: calls.append(list(a))), patch.object(
                legacy.subprocess, "run"), patch.object(sys, "argv", ["run_all.py", "--resume"]), redirect_stdout(log):
            legacy.main()
        result["legacy_resume_control_flow"] = {"executed_python_stages_mocked": calls,
                                                 "solver_case_calls": sum("--case" in a for a in calls),
                                                 "legacy_cases_reused": 9}

        delivery_dir = sandbox / "delivery"
        validation_dir = sandbox / "validation"
        delivery_dir.mkdir()
        validation_dir.mkdir()
        shutil.copy2(current.AUDIT_DIR / "workbook_payload.json", delivery_dir / "workbook_payload.json")
        with patch.object(current, "AUDIT_DIR", delivery_dir), patch.object(current, "DELIVERY_DIR", delivery_dir), patch.object(current, "VALIDATION_DIR", validation_dir):
            result["workbook_reader_missing_export"] = caught(current.verify_workbook)

        # Check the real document-read statement without running unrelated verification writers.
        doc_statement = next(n for n in ast.walk(nodes["run_verification"])
                             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                             and n.targets[0].id == "formula_doc")
        doc_code = compile(ast.Module(body=[doc_statement], type_ignores=[]), str(SRC / "solve_problem3.py"), "exec")
        result["model_document_read_after_relocation"] = caught(
            lambda: exec(doc_code, {"ROOT": ROOT, "OUT": ORIGINAL_OUT}))

        reduced = {"status": "PASS", "scope": {"delivery_ready": False},
                   "skipped": {"spatial_and_temporal_convergence": "SKIPPED"}}
        current.dump_json(validation_dir / "verification.json", reduced)
        with patch.object(current, "AUDIT_DIR", delivery_dir), patch.object(current, "DELIVERY_DIR", delivery_dir), patch.object(
                current, "VALIDATION_DIR", validation_dir), patch.object(
                current, "audit", return_value=(values, audit_report)), redirect_stdout(log):
            result["reduced_report_partial_delivery"] = caught(current.run_delivery)
        result["reduced_report_partial_delivery"]["files_after_rejection"] = sorted(p.name for p in delivery_dir.iterdir())

        # Control-flow-only fixture: numerical checks use the actual previously verified record.
        cached = json.loads((current.VALIDATION_DIR / "verification.json").read_text(encoding="utf-8"))
        fake_root, cases, audit_dir = sandbox / "fake_project", sandbox / "cases", sandbox / "audit"
        (fake_root / "src/problem-3").mkdir(parents=True)
        cases.mkdir()
        audit_dir.mkdir()
        shutil.copy2(SRC / "模型与算法说明.md", fake_root / "src/problem-3/模型与算法说明.md")
        for suffix in [".npz", ".json", "_event_seed.npz"]:
            shutil.copy2(current.CASES_DIR / f"production{suffix}", cases / f"production{suffix}")
        current.dump_json(validation_dir / "protected_hashes_before.json", {"src/problem-3/solve_problem3.py": "before"})
        with patch.multiple(current, ROOT=fake_root, CASES_DIR=cases, AUDIT_DIR=audit_dir,
                            VALIDATION_DIR=validation_dir), patch.object(
                current, "audit", return_value=(values, audit_report)), patch.object(
                current, "hashes", return_value={"src/problem-3/solve_problem3.py": "after"}), patch.object(
                current, "boundary_metrics", return_value=cached["production_boundaries"]), patch.object(
                current, "event_checks", return_value=cached["event"]), patch.object(
                current, "synthetic_tests", return_value=result["synthetic_and_BDF"]), patch.object(
                current, "regression", return_value=cached["previous_problem_comparison"]), redirect_stdout(log):
            weak = current.run_verification(convergence=False, sensitivity=False)
        result["source_change_and_reduced_verification_gate"] = {
            "fixture": "Control-flow probe, cached numerical checks; not a new numerical verification",
            "status": weak["status"], "scope": weak["scope"],
            "protected_files": weak["protected_files"], "skipped": weak["skipped"],
            "production_label": json.loads((cases / "production.json").read_text())["all_necessary_numerical_checks"]}

        # Existence-only resume ignores explicitly changed run settings.
        resume_calls = []
        with patch.object(current, "CASES_DIR", cases), patch.object(current, "audit"), patch.object(
                current, "simulate", side_effect=lambda *a, **k: resume_calls.append([a, k])), redirect_stdout(log):
            current.run_all(resume=True, convergence=False, sensitivity=False, verify=False,
                            deliver=False, figures=False, workbook=False, prod_n=400,
                            prod_mode="fluct", prod_seed=19)
        result["changed_settings_resume"] = {"requested_N": 400, "requested_mode": "fluct",
                                              "stored_N": cached["production"]["N"],
                                              "stored_mode": cached["production"]["mode"],
                                              "simulate_call_count": len(resume_calls)}

    print("Path and verification-gate fault probes completed.", flush=True)
    env_a = current.Environment(values, "fluct", seed=0)
    env_b = current.Environment(values, "fluct", seed=0)
    no_seed_a = current.Environment(values, "fluct")
    no_seed_b = current.case_environment(values, {"mode": "fluct", "seed": None})
    sample_times = np.array([14460., 15000., 18000., 86400., 200000.])
    result["stochastic_reproducibility"] = {
        "sample_times_s": sample_times.tolist(),
        "run_all_default_prod_seed": inspect.signature(current.run_all).parameters["prod_seed"].default,
        "fixed_seed_max_difference": float(max(np.max(abs(env_a(t) - env_b(t))) for t in sample_times)),
        "seed_none_max_difference_by_channel": np.max(np.abs(
            np.array([no_seed_a(t) for t in sample_times]) - np.array([no_seed_b(t) for t in sample_times])), axis=0).tolist(),
        "observed_interval_max_difference": float(max(np.max(abs(env_a(t) - row[1:])) for t, row in zip(values[:, 0], values))),
        "stats": env_a.stats, "clipped_phi": env_a.fluct["phi"],
        "grid_nodes": len(env_a.fluct["grid"])}

    def equivalent_query(t):
        """Benchmark algebraically equivalent interpolation without whole-array allocation."""
        return env_a.mean_tail + env_a.fluct["sigma"] * np.array(
            [np.interp(t, env_a.fluct["grid"], z) for z in env_a.fluct["z"]])

    timings = {"current": [], "equivalent": []}
    queries = np.linspace(15000., 205000., 200)
    for _ in range(3):
        for name, function in [("current", env_a), ("equivalent", equivalent_query)]:
            started = time.perf_counter()
            samples = np.array([function(t) for t in queries])
            timings[name].append(time.perf_counter() - started)
    result["stochastic_boundary_benchmark"] = {
        "queries_per_repeat": len(queries), "seconds": timings,
        "median_speed_ratio": float(np.median(timings["current"]) / np.median(timings["equivalent"])),
        "equivalent_query_max_difference": float(max(np.max(abs(env_a(t) - equivalent_query(t))) for t in sample_times)),
        "new_full_length_channel_arrays_per_query": 2,
        "elements_per_channel_array": len(env_a.fluct["grid"])}

    with TemporaryDirectory(prefix="plot-probe-", dir=REVIEW) as plot_cache, patch.dict(
            os.environ, {"MPLCONFIGDIR": plot_cache}):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plotting = load_module("review_figures", SRC / "generate_figures.py")
        labels = {}

        def inspect_figure(fig, name):
            """Inspect in-memory plot data, without writing or reading images."""
            if name == "drying_history":
                labels["mean_case_curve_label"] = fig.axes[1].lines[0].get_label()
            if name == "temperature_and_environment":
                line = fig.axes[0].lines[2]
                labels["displayed_air_tail_label"] = line.get_label()
                labels["displayed_air_tail_degC"] = float(line.get_ydata()[-1])
            plt.close(fig)

        with np.load(current.CASES_DIR / "mean12800.npz") as mean_case, patch.object(
                plotting, "_import_matplotlib", return_value=plt), patch.object(
                plotting, "save_figure", side_effect=inspect_figure):
            plotting.figures(mean_case, None, {}, values)
        labels["actual_case_mode"] = "mean30"
        labels["actual_air_tail_degC"] = float(values[values[:, 0] >= 12600, 1].mean())
        result["figure_boundary_mismatch"] = labels

    ensemble = load_module("review_ensemble", SRC / "ensemble_problem3.py")
    with warnings.catch_warnings(record=True) as warning_log:
        warnings.simplefilter("always")
        one = ensemble._summary(np.array([57.]))
        result["single_member_summary"] = caught(lambda: json.dumps(one, allow_nan=False))
        result["single_member_summary"]["warning_messages"] = [str(w.message) for w in warning_log]

    coarse_meta = json.loads((current.CASES_DIR / "space400.json").read_text())
    fine_meta = json.loads((current.CASES_DIR / "production.json").read_text())
    result["existing_deterministic_grid_error_context"] = {
        "scope": "Existing deterministic cases, not a new stochastic error estimate",
        "space400_minus_production_s": coarse_meta["event"]["estimate_s"] - fine_meta["event"]["estimate_s"]}
    result["unchanged_inputs_and_official_results"] = {
        "status": "PASS" if before == fingerprint(inputs) else "FAIL", "files_checked": len(inputs)}
    result["not_executed"] = ["full 12800-node recomputation", "full stochastic ensemble",
                               "stochastic spatial/time convergence", "visual inspection of output PNGs"]
    (REVIEW / "probe_stdout_current.txt").write_text(log.getvalue(), encoding="utf-8")
    current.dump_json(REVIEW / "review_checks_current.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
