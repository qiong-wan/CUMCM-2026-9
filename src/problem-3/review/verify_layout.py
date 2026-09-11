"""Check categorized I/O contracts without rerunning the numerical solver."""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch
from urllib.parse import unquote

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src/problem-3"
OUT = ROOT / "output/problem-3"
sys.path.insert(0, str(SRC))


def main():
    """Verify both loaders, workbook readers, resume paths, syntax and local links."""
    import solve_problem3 as solver
    import verify_problem3 as legacy_verifier
    import run_all

    report = {"scope": "Output categorization only; numerical solver and full export not rerun"}
    expected = {"AUDIT_DIR": OUT / "data", "CASES_DIR": OUT / "cases",
                "VALIDATION_DIR": OUT / "review", "DELIVERY_DIR": OUT,
                "FIGURES_DIR": OUT / "image"}
    report["directory_constants"] = all(getattr(solver, k) == v for k, v in expected.items())
    loaders = {}
    for path in sorted(solver.CASES_DIR.glob("*.json")):
        a, meta_a = solver.load_case(path.stem)
        b, meta_b = legacy_verifier.load_case(path.stem)
        loaders[path.stem] = (meta_a == meta_b and a.files == b.files
                              and all(np.array_equal(a[k], b[k]) for k in a.files))
        a.close()
        b.close()
    report["case_loaders_agree"] = loaders
    with redirect_stdout(io.StringIO()):
        report["new_workbook_reader"] = solver.verify_workbook()
        report["legacy_workbook_reader"] = legacy_verifier.verify_workbook()
    calls = []
    with patch.object(run_all, "run_python", side_effect=lambda *args: calls.append(args)), patch.object(
            run_all.subprocess, "run"), patch.object(sys, "argv", ["run_all.py", "--resume"]), redirect_stdout(io.StringIO()):
        run_all.main()
    report["resume_reuses_all_nine_cases"] = len(loaders) == 9 and not any("--case" in args for args in calls)
    report["resume_probe_scope"] = "Control flow only; subprocesses mocked"
    report["python_syntax"] = {}
    for path in [*SRC.glob("*.py"), *(SRC / "review").glob("*.py")]:
        ast.parse(path.read_text(encoding="utf-8"))
        report["python_syntax"][str(path.relative_to(SRC))] = "PASS"
    js = (SRC / "build_workbook.mjs").read_text(encoding="utf-8")
    report["workbook_export_paths"] = all(s in js for s in (
        "path.join(out, 'data/workbook_payload.json')", "path.join(out, 'image/previews')",
        "path.join(out, 'review')", "path.join(out, 'result3.xlsx')"))
    checked, broken = 0, []
    for path in [*SRC.glob("*.md"), *(SRC / "review").glob("*.md"), *OUT.glob("*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+://", target) or target.startswith("#"):
                continue
            checked += 1
            file_target = re.sub(r":\d+$", "", unquote(target.split("#")[0].strip("<>")))
            target_path = path.parent / file_target
            if not target_path.exists() and target_path.resolve() != (OUT / "review/layout_checks.json").resolve():
                broken.append({"document": str(path.relative_to(ROOT)), "target": target})
    report["local_links"] = {"checked": checked, "broken": broken}
    root_files = {p.name for p in OUT.iterdir() if p.is_file()}
    report["root_contains_only_deliverables"] = root_files == {
        "README.md", "result3.xlsx", "table5.md", "table5_moisture.csv",
        "moisture_60s_full_precision.csv", "结果与验证.md"}
    report["obsolete_output_directories_absent"] = not any(
        (OUT / name).exists() for name in ["audit", "validation", "delivery", "figures", "previews"])
    report["full_solver_and_export"] = "SKIPPED"
    report["status"] = "PASS" if all([
        report["directory_constants"], bool(loaders), all(loaders.values()),
        report["new_workbook_reader"]["status"] == "PASS", report["legacy_workbook_reader"]["status"] == "PASS",
        report["resume_reuses_all_nine_cases"], report["workbook_export_paths"], not broken,
        report["root_contains_only_deliverables"], report["obsolete_output_directories_absent"]]) else "FAIL"
    (OUT / "review/layout_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Output layout verification failed")


if __name__ == "__main__":
    main()
