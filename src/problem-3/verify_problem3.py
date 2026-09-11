"""Executed convergence, invariant, endpoint and artifact checks for problem 3."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import re

import numpy as np
import openpyxl
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat

from solve_problem3 import (OUT, ROOT, R, H, HM, Environment, Solver, audit,
                            dump_json, hashes, properties, refine_event)


def load_case(name):
    """Read one full-precision case and its recorded run settings."""
    data = np.load(OUT / f"{name}.npz")
    meta = json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
    return data, meta


def field_metric(a, b, times, radii):
    """Return infinity errors and moisture-error location on matched arrays."""
    delta = np.abs(a - b)
    idx = np.unravel_index(np.argmax(delta[:, 1]), delta[:, 1].shape)
    return {"T_max_degC": float(delta[:, 0].max()),
            "C_max_kg_kg": float(delta[:, 1].max()),
            "C_max_time_s": float(times[idx[0]]),
            "C_max_radius_m": float(radii[idx[1]])}


def compare(name_a, name_b):
    """Compare identical times and nested spatial nodes, never unequal endpoints."""
    a, ma = load_case(name_a)
    b, mb = load_case(name_b)
    ratio = mb["N"] // ma["N"]
    if ratio < 1 or mb["N"] % ma["N"]:
        raise ValueError("Comparison requires nested meshes")
    times, ia, ib = np.intersect1d(a["times"], b["times"], return_indices=True)
    outputs = field_metric(a["outputs"][ia], b["outputs"][ib], times, np.linspace(0, R, 21))
    ft, fa, fb = np.intersect1d(a["full_times"], b["full_times"], return_indices=True)
    fields_a, fields_b = a["full_states"][fa], b["full_states"][fb, :, ::ratio]
    sections = {}
    for label, select in [("all_saved_fields", ft > 0), ("startup_to_60s", (ft > 0) & (ft <= 60)),
                          ("six_hour_fields", (ft > 0) & (ft % 21600 == 0)),
                          ("late_48h_plus", ft >= 172800)]:
        sections[label] = field_metric(fields_a[select], fields_b[select], ft[select], a["r"])
    return {"coarse": name_a, "fine": name_b, "coarse_N": ma["N"], "fine_N": mb["N"],
            "factors": [ma["factor"], mb["factor"]], "delivery": outputs,
            **sections, "event_difference_s": mb["event"]["estimate_s"] - ma["event"]["estimate_s"],
            "per_snapshot_C_difference": [[float(t), float(d)] for t, d in
                zip(ft, np.max(abs(fields_a[:, 1] - fields_b[:, 1]), axis=1))]}


def synthetic_tests(values):
    """Test equilibrium, shared flux, nonlinear rejection and an independent integrator."""
    result = {}
    solver = Solver(100, lambda t: np.array([35., .6]))
    uniform = np.stack([np.full(101, 35.), np.full(101, .6)])
    final, balance = solver.step(uniform, 0, 20)
    result["uniform_equilibrium_error"] = float(np.max(abs(final - uniform)))
    C = .4 + .2 * (1 - (solver.r / R)**2)
    T = 30 + 10 * solver.r / R
    a, k, D = properties(T, C)
    flux = solver.net(C, solver.conductance(D), 0., .5)
    result["closed_boundary_flux_sum"] = float(abs(np.sum(flux)))
    variable_flux = solver.net(C, solver.conductance(D), HM, .05)
    result["variable_coefficient_flux_balance"] = float(abs(
        variable_flux.sum() - solver.ar * HM * (.05 - C[-1])))
    try:
        incompatible = np.stack([np.full(101, 28.), np.full(101, 2.55)])
        solver.implicit(incompatible, incompatible, 1., 1., max_iter=1)
        result["nonconvergence_raises"] = False
    except RuntimeError:
        result["nonconvergence_raises"] = True
    for label, wrong in [("zero_C_rejected", np.zeros(101)),
                         ("negative_C_rejected", -np.ones(101)),
                         ("nan_C_rejected", np.full(101, np.nan))]:
        try:
            properties(T, wrong)
            result[label] = False
        except FloatingPointError:
            result[label] = True
    result["D_initial_m2_s"] = float(properties(np.array([28.]), np.array([2.55]))[2][0])
    # Independent BDF integration checks time stages against the same spatial ODE.
    solver = Solver(100, Environment(values))
    initial = np.r_[np.full(101, 28.), np.full(101, 2.55)]

    def rhs(t, y):
        """Evaluate the semidiscrete ODE without implicit-stage linear algebra."""
        T, C = y[:101], y[101:]
        a, k, D = properties(T, C)
        amb = solver.env(t)
        return np.r_[solver.net(T, solver.conductance(k), H, amb[0]) / (solver.v * a),
                     solver.net(C, solver.conductance(D), HM, amb[1]) / solver.v]

    tri = diags([np.ones(100), np.ones(101), np.ones(100)], [-1, 0, 1])
    sparse = bmat([[tri, tri], [tri, tri]], format="csc")
    reference = solve_ivp(rhs, (0., 60.), initial, method="BDF", rtol=2e-10,
                          atol=2e-12, max_step=.2, jac_sparsity=sparse)
    state = initial.reshape(2, 101)
    t = 0.
    while t < 60:
        dt = min(.1, max(.00025, .015 * (t + .01)), 60 - t)
        state, _ = solver.step(state, t, dt)
        t += dt
    result["independent_BDF_success"] = bool(reference.success)
    result["independent_BDF_field_error"] = np.max(
        abs(state - reference.y[:, -1].reshape(2, 101)), axis=1).tolist()
    result["independent_BDF_scope"] = "N=100, 0..60 s, independent time integration, shared spatial operator"
    passed = (result["uniform_equilibrium_error"] < 1e-12
              and result["closed_boundary_flux_sum"] < 1e-20
              and result["variable_coefficient_flux_balance"] < 1e-20
              and all(result[k] for k in ["nonconvergence_raises", "zero_C_rejected",
                                         "negative_C_rejected", "nan_C_rejected",
                                         "independent_BDF_success"])
              and max(result["independent_BDF_field_error"]) < 5e-6)
    result["status"] = "PASS" if passed else "FAIL"
    return result


def integrate_local(solver, state, t0, target, cap):
    """Continue an unrounded saved full field with bounded steps to a common time."""
    while t0 < target - 1e-10:
        step = min(cap, target - t0, 60 * (np.floor(t0 / 60) + 1) - t0)
        state, _ = solver.step(state, t0, step)
        t0 += step
    return state


def event_checks(name, values):
    """Separate root tolerance, local integration and endpoint all-node checks."""
    data, meta = load_case(name)
    solver = Solver(meta["N"], Environment(values, meta["mode"]))
    seed = np.load(OUT / f"{name}_event_seed.npz")
    old, t0, width = seed["state"], float(seed["t"]), float(seed["width"])
    tight, _, _ = refine_event(solver, old, t0, width, 1e-5)
    lower, upper = 0., width
    while upper - lower > 1e-5:
        mid = (lower + upper) / 2
        half, _ = solver.step(old, t0, mid / 2)
        state, _ = solver.step(half, t0 + mid / 2, mid / 2)
        if state[1].max() < .15:
            upper = mid
        else:
            lower = mid
    split_event = t0 + (lower + upper) / 2
    offset = tight["estimate_s"] - t0
    left, _ = solver.step(old, t0, offset - 1.)
    right, _ = solver.step(old, t0, offset + 1.)
    slope = (right[1].max() - left[1].max()) / 2.
    check = {"case": name, "original_event": meta["event"], "tight_event": tight,
             "root_tolerance_change_s": tight["estimate_s"] - meta["event"]["estimate_s"],
             "two_half_steps_event_s": split_event,
             "local_step_change_s": split_event - tight["estimate_s"],
             "slope_kg_kg_per_s": slope, "time_amplification_s_per_5e_5": 5e-5 / abs(slope),
             "fine_nodes_checked": meta["N"] + 1,
             "strict_end_max_C": float(data["final_state"][1].max()),
             "strict_end_margin": float(.15 - data["final_state"][1].max()),
             "max_location_m": float(data["r"][np.argmax(data["final_state"][1])])}
    check["status"] = "PASS" if (
        check["strict_end_max_C"] < .15 and tight["g_lower"] >= 0 > tight["g_upper"]
        and abs(check["root_tolerance_change_s"]) <= .001
        and abs(check["local_step_change_s"]) < .01) else "FAIL"
    return check


def boundary_metrics(name, values):
    """Measure independent gradients at startup, all delivery times and the endpoint."""
    data, meta = load_case(name)
    solver = Solver(meta["N"], Environment(values, meta["mode"]))
    records = []
    for t, state in zip(data["full_times"][1:], data["full_states"][1:]):
        robin, symmetry = solver.boundaries(state, t)
        records.append([float(t), *robin.tolist(), *symmetry.tolist()])
    records = np.array(records)
    delivery = np.max(abs(data["history"][:, 8:12]), axis=0)
    startup = np.max(abs(records[records[:, 0] <= 60, 1:]), axis=0)
    return {"case": name, "delivery_abs_max_RT_RC_symT_symC": delivery.tolist(),
            "startup_abs_max_RT_RC_symT_symC": startup.tolist(),
            "snapshot_records": records.tolist(),
            "status": "PASS" if (delivery[0] < 1e-4 and delivery[1] < 1e-9
                                     and delivery[2] < 1e-6 and delivery[3] < 1e-6) else "FAIL"}


def regression(name):
    """Compare to previous rounded workbook for context, never use it as state."""
    data, meta = load_case(name)
    source = ROOT / "output" / "problem-2" / "result2.xlsx"
    if not source.exists():
        return {"status": "SKIPPED", "reason": "Previous workbook absent"}
    source_bytes = source.read_bytes()
    wb = openpyxl.load_workbook(BytesIO(source_bytes), read_only=True, data_only=True)
    errors = []
    snapshot = []
    for j, sheet in enumerate(["温度", "水分浓度"]):
        rows = list(wb[sheet].values)
        for ti, t in enumerate(data["times"]):
            if 60 <= t <= 10800:
                prior = np.array(rows[int(t)][1:22], dtype=float)
                if rows[int(t)][0] != int(t):
                    raise ValueError("Old workbook time mismatch")
                delta = abs(data["outputs"][ti, j] - prior)
                errors.append([j, float(t), float(delta.max()), int(delta.argmax())])
                snapshot.append([j, float(t), *prior.tolist()])
    wb.close()
    summary = {}
    for j, label in enumerate(["T", "C"]):
        selected = [row for row in errors if row[0] == j]
        worst = max(selected, key=lambda row: row[2])
        summary[label] = {"max_abs_difference": worst[2], "time_s": worst[1],
                          "radius_cm": worst[3] / 10}
    np.savez_compressed(OUT / "problem2_comparison_snapshot.npz", samples=snapshot)
    return {"status": "PASS", "interpretation": "Comparison executed against the recorded workbook snapshot; previous rounded outputs are not exact truth",
            "source_workbook_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "rounding_uncertainty_per_old_value": 5e-5, "sampling": "60 s, 21 nodes, first 3 h", **summary}


def verify_workbook():
    """Reopen all delivered cells and reconcile times, headers, values and formats."""
    payload = json.loads((OUT / "workbook_payload.json").read_text(encoding="utf-8"))
    wb = openpyxl.load_workbook(OUT / "result3.xlsx", read_only=False, data_only=False)
    ws = wb["Sheet1"]
    failures = []
    if wb.sheetnames != ["Sheet1"]:
        failures.append("sheet names")
    if ws.max_row != len(payload["rows"]) + 1 or ws.max_column != 22:
        failures.append("dimensions")
    if [c.value for c in ws[1]] != payload["header"]:
        failures.append("header")
    delta_max, format_errors = 0., 0
    for i, (cells, expected) in enumerate(zip(ws.iter_rows(min_row=2,
            max_row=len(payload["rows"]) + 1, max_col=22), payload["rows"]), start=2):
        actual = [c.value for c in cells]
        if any(not isinstance(v, (int, float)) or not np.isfinite(v) for v in actual):
            failures.append(f"nonnumeric row {i}")
            continue
        delta_max = max(delta_max, float(np.max(abs(np.array(actual) - expected))))
        format_errors += sum(cell.number_format != "0.0000" for cell in cells[1:])
    wb.close()
    times = np.array(payload["rows"])[:, 0]
    regular = times[:-1] if times[-1] % 60 != 0 else times
    if not np.array_equal(regular, np.arange(60, regular[-1] + 1, 60)):
        failures.append("regular times")
    if delta_max > 5e-10 or format_errors:
        failures.append("numeric values or format")
    table = np.loadtxt(OUT / "table5_moisture.csv", delimiter=",", skiprows=1)
    for row in table:
        idx = int(np.argmin(abs(times - row[1])))
        if abs(times[idx] - row[1]) > 1e-9 or np.max(abs(
                np.array(payload["rows"])[idx, [1, 6, 11, 16, 21]] - row[2:])) > 1e-14:
            failures.append("table5 consistency")
    report = {"status": "PASS" if not failures else "FAIL", "failures": failures,
              "rows_including_header": len(payload["rows"]) + 1, "columns": 22,
              "numeric_moisture_cells": len(payload["rows"]) * 21,
              "maximum_roundtrip_difference": delta_max, "format_errors": format_errors,
              "regular_rows": len(regular), "extra_actual_end_time_s": float(times[-1]),
              "table5_checked_points": int(len(table) * 5)}
    dump_json(OUT / "workbook_verification.json", report)
    narrative = OUT / "结果与验证.md"
    if narrative.exists():
        heading = "## 12. 工作簿最终复核"
        body = narrative.read_text(encoding="utf-8").split(heading)[0].rstrip()
        summary = (f"\n\n{heading}\n\n"
                   f"本次全量重读核查状态为 **{report['status']}**。工作簿共有 "
                   f"{report['rows_including_header']} 行（含表头）、22 列；"
                   f"全部 {report['numeric_moisture_cells']} 个含水率格与主解载荷的最大差为 "
                   f"{delta_max:.1e}，数值格式错误 {format_errors} 个；"
                   f"表 5 的 {report['table5_checked_points']} 个对应值逐点一致。"
                   f"规则时间从 60 s 开始共 {len(regular)} 行，末尾实际结束时刻为 "
                   f"{times[-1]:.12f} s。\n")
        narrative.write_text(body + summary, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("Workbook verification FAIL")
    return report


def main():
    """Build an auditable validation record, failing on missing or failed necessities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", action="store_true")
    args = parser.parse_args()
    if args.workbook:
        verify_workbook()
        return
    values, _ = audit()
    spatial_names = [f"space{n}" for n in [400, 800, 1600, 3200, 6400, 12800]]
    spatial = [compare(a, b) for a, b in zip(spatial_names[:-1], spatial_names[1:])]
    temporal = [compare("space12800", "production"), compare("production", "time12800quarter")]
    production, meta = load_case("production")
    sensitivity, smeta = load_case("mean12800")
    boundary = [boundary_metrics(name, values) for name in spatial_names]
    primary_boundary = boundary_metrics("production", values)
    event = event_checks("production", values)
    unit = synthetic_tests(values)
    unit["zeroed_boundary_counterexample_relative_defect"] = abs(meta["mass_change_scaled"]) / max(
        abs(meta["mass_change_scaled"]), R**2)
    unit["zeroed_boundary_counterexample_status"] = "PASS" if unit[
        "zeroed_boundary_counterexample_relative_defect"] > 1e-7 else "FAIL"
    near_states = []
    for name in ["space6400", "space12800", "production", "time12800quarter"]:
        data, settings = load_case(name)
        solver = Solver(settings["N"], Environment(values))
        idx = np.where(data["full_times"] == 194400.)[0][0]
        state = integrate_local(solver, data["full_states"][idx], 194400., 205800., 30 * settings["factor"])
        near_states.append(state)
    near = {"common_time_s": 205800.,
            "space6400_to_12800_full_C_max": float(np.max(abs(near_states[0][1] - near_states[1][1, ::2]))),
            "time_factor1_to_half_full_C_max": float(np.max(abs(near_states[1][1] - near_states[2][1]))),
            "time_factorhalf_to_quarter_full_C_max": float(np.max(abs(near_states[2][1] - near_states[3][1])))}
    np.savez_compressed(OUT / "near_end_common_fields.npz", time_s=205800.,
                        space6400=near_states[0], space12800=near_states[1],
                        production=near_states[2], time12800quarter=near_states[3])
    times, pi, si = np.intersect1d(production["times"], sensitivity["times"], return_indices=True)
    mask = times <= 14400
    unchanged_obs = float(np.max(abs(production["outputs"][pi[mask]] - sensitivity["outputs"][si[mask]])))
    delta_space = abs(spatial[-1]["event_difference_s"])
    p_space = float(np.log2(abs(spatial[-2]["event_difference_s"]) / delta_space))
    delta_time = abs(temporal[-1]["event_difference_s"])
    estimate_space = delta_space / (2**p_space - 1)
    required = {
        "input_audit": True,
        "synthetic_and_independent_short_time_checks": unit["status"] == "PASS",
        "independent_boundary_corruption_detected": unit["zeroed_boundary_counterexample_status"] == "PASS",
        "space_delivery_C_below_5e_5": spatial[-1]["delivery"]["C_max_kg_kg"] < 5e-5,
        "space_full_saved_C_below_5e_5": spatial[-1]["all_saved_fields"]["C_max_kg_kg"] < 5e-5,
        "time_full_saved_C_below_5e_5": temporal[-1]["all_saved_fields"]["C_max_kg_kg"] < 5e-5,
        "space_time_change_below_36s": delta_space < 36,
        "time_time_change_below_1s": delta_time < 1,
        "independent_moisture_balance": meta["mass_relative_defect"] < 1e-7,
        "variable_capacity_heat_balance": meta["heat_relative_defect"] < 1e-7,
        "nonlinear_change_and_equation_defect": meta["max_scaled_residual"] <= 2e-11 and meta["max_scaled_change"] <= 2e-11,
        "positive_finite_states": meta["moisture_range"][0] > 0 and meta["diffusivity_range"][0] > 0,
        "full_domain_radial_order": meta["radial_inversion_max"] <= 1e-9,
        "full_domain_maximum_at_center": meta["max_C_above_center"] <= 1e-9,
        "independent_robin_and_symmetry": primary_boundary["status"] == "PASS",
        "event_and_strict_end": event["status"] == "PASS",
        "observations_identical_in_sensitivity": unchanged_obs == 0,
        "sensitivity_run_balances_and_strict_end": (
            smeta["mass_relative_defect"] < 1e-7 and smeta["heat_relative_defect"] < 1e-7
            and smeta["max_scaled_residual"] <= 2e-11
            and smeta["moisture_range"][0] > 0 and smeta["event"]["g_end"] < 0
            and smeta["radial_inversion_max"] <= 1e-9),
    }
    report = {"status": "PASS" if all(required.values()) else "FAIL",
              "checks": {k: "PASS" if v else "FAIL" for k, v in required.items()},
              "production_case": "production", "production": meta,
              "space_convergence": spatial, "time_convergence": temporal,
              "near_end_common_time_fields": near, "boundary_convergence": boundary,
              "production_boundaries": primary_boundary, "event": event,
              "synthetic_tests": unit, "previous_problem_comparison": regression("production"),
              "empirical_error": {"space_observed_order": p_space,
                  "space_fine_residual_time_estimate_s": estimate_space,
                  "space_last_difference_s": delta_space,
                  "time_last_difference_s": delta_time,
                  "conservative_numeric_allowance_s": float(np.ceil(2 * (delta_space + delta_time) + .101)),
                  "interpretation": "Twice the last space/time differences plus event and strict-end margins, rounded up; empirical, not rigorous or process safety margin"},
              "boundary_sensitivity": {"mode": "inclusive 12600..14400 s arithmetic mean, 31 observations",
                  "tail": smeta["environment_tail"], "event_s": smeta["event"]["estimate_s"],
                  "difference_s": smeta["event"]["estimate_s"] - meta["event"]["estimate_s"],
                  "difference_h": (smeta["event"]["estimate_s"] - meta["event"]["estimate_s"]) / 3600,
                  "observed_interval_difference": unchanged_obs,
                  "mass_balance": smeta["mass_relative_defect"], "heat_balance": smeta["heat_relative_defect"]},
              "skipped": {"independent_full_duration_spatial_method": "SKIPPED",
                          "two_dimensional_end_faces": "SKIPPED",
                          "latent_heat_and_full_enthalpy_closure": "SKIPPED",
                          "validation_against_measured_material_fields": "SKIPPED"}}
    formula_doc = (OUT / "模型与算法说明.md").read_text(encoding="utf-8")
    numbers = [int(v) for v in re.findall(r"\\tag\{(\d+)\}", formula_doc)]
    report["formula_numbering"] = "PASS" if numbers == list(range(1, 66)) else "FAIL"
    before = json.loads((OUT / "protected_hashes_before.json").read_text(encoding="utf-8"))
    after = hashes()
    changed = [p for p, h in before.items() if after.get(p) != h]
    critical_prefixes = ("data/", "src/common/", "src/problem-1/")
    critical_changed = [p for p in changed if p.replace("\\", "/").startswith(critical_prefixes)
                        or p.replace("\\", "/") in ["problem.pdf", "AGENTS.md"]]
    report["protected_files"] = {"status": "PASS" if not changed else "FAIL",
                                  "checked": len(before), "changed": changed,
                                  "critical_model_inputs_status": "FAIL" if critical_changed else "PASS",
                                  "critical_model_inputs_changed": critical_changed,
                                  "external_change_note": "The task did not write previous-question files. Concurrent updates are retained and recorded; original baseline hashes are not replaced.",
                                  "excluded": "Office lock files, bytecode and output PNGs (repository instruction)"}
    dump_json(OUT / "protected_hashes_after.json", after)
    if critical_changed or report["formula_numbering"] == "FAIL":
        report["status"] = "FAIL"
    report["status_scope"] = "Executed problem-3 numerical requirements and immutable model inputs; external previous-question hash mismatches remain FAIL in protected_files"
    if report["status"] == "PASS":
        meta["all_necessary_numerical_checks"] = "PASS"
        dump_json(OUT / "production.json", meta)
    if (OUT / "verification.json").exists():
        previous = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
        if previous["status"] == "FAIL" and not (OUT / "verification_initial_scope_failure.json").exists():
            dump_json(OUT / "verification_initial_scope_failure.json", previous)
    dump_json(OUT / "verification.json", report)
    print(json.dumps({"status": report["status"], "checks": report["checks"],
                      "error": report["empirical_error"], "sensitivity": report["boundary_sensitivity"]},
                     ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Required validation FAIL; do not deliver")


if __name__ == "__main__":
    main()
