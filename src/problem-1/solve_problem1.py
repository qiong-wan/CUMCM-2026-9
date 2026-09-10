"""Problem 1: radial temperature and moisture concentration of a cylindrical
medicinal material during the preheating/equilibrium stage (0--1800 s).

Model (1D axisymmetric cylinder, radius R, constant length L)
------------------------------------------------------------
Temperature (Fourier, constant properties from Appendix 2):

    rho*cp*dT/dt = (1/r) d/dr ( k r dT/dr ),   0 < r < R,
    dT/dr(0,t) = 0,
    -k dT/dr(R,t) = h [T(R,t) - T_inf(t)],
    T(r,0) = 28 degC.

Moisture (Fick, concentration-dependent diffusivity):

    dC/dt = (1/r) d/dr ( r D(C) dC/dr ),   0 < r < R,
    dC/dr(0,t) = 0,
    -D(C) dC/dr(R,t) = h_m [C(R,t) - C_inf(t)],
    C(r,0) = 2.55 kg/kg,
    D(C) = 7e-9 * exp(-0.89 / C)  [m^2/s].

The ambient temperature T_inf(t) and moisture concentration C_inf(t) are read
from data/附件1.xlsx (every 60 s) and linearly interpolated in time.

Numerical method
----------------
Node-centered finite-volume discretisation on the uniform grid
r_i = i*dr (i = 0..N, dr = R/N).  Backward-Euler in time with sub-step
dt = 1/nsub s and integer-second sampling.  The resulting tridiagonal systems
are solved with scipy.linalg.solve_banded; the nonlinear moisture system is
handled by Picard iteration on the face diffusivities (harmonic mean of D(C)).

A refined "computational" mesh (N_REF cells) is used and sampled at the
required 0.1 cm "delivery" nodes, which removes the near-surface discretisation
bias of the coarse mesh.  The workbook stores t = 1..1800 s as in the template.

Run:
    python src/problem-1/solve_problem1.py            # results + tables
    python src/problem-1/solve_problem1.py --verify   # also convergence study

See src/problem-1/问题1_算法与模型说明.md for the model and algorithm notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import openpyxl
from scipy.linalg import solve_banded

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "附件1.xlsx"
TEMPLATE_FILE = ROOT / "data" / "附件3" / "result1.xlsx"
OUT_DIR = ROOT / "output" / "problem-1"
RESULT_FILE = OUT_DIR / "result1.xlsx"
AUDIT_FILE = OUT_DIR / "verification.txt"

# --------------------------------------------------------------------------
# Physical parameters (Appendix 2)
# --------------------------------------------------------------------------
R = 0.02            # cylinder radius [m]
L = 0.25            # cylinder length [m] (unused in the 1D radial model)
RHO = 820.0         # density [kg/m^3]
CP = 2600.0         # specific heat [J/(kg K)]
K = 0.36            # thermal conductivity [W/(m K)]
H = 25.0            # convective heat transfer coefficient [W/(m^2 K)]
HM = 8.0e-7         # convective mass transfer coefficient [m/s]
ALPHA = K / (RHO * CP)

T_INIT = 28.0       # initial temperature [degC]
C_INIT = 2.55       # initial dry-basis moisture [kg/kg]

T_END = 1800.0      # simulation horizon [s]
N_OUT = 20          # delivery grid -> 0.1 cm radial spacing
N_REF = 1600        # production (computational) mesh -> dr = 0.00125 cm
NSUB = 40           # implicit sub-steps per output second (dt = 0.025 s)

PICARD_TOL = 1.0e-12
PICARD_MAX_IT = 60

TABLE_TIMES = np.array([100, 300, 600, 900, 1200, 1500, 1800])
TABLE_DIST_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])


def D_of_C(C: np.ndarray) -> np.ndarray:
    """Moisture diffusivity D(C) = 7e-9 exp(-0.89/C) [m^2/s]."""
    C_safe = np.maximum(C, 1.0e-8)
    return 7.0e-9 * np.exp(-0.89 / C_safe)


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Harmonic mean 2ab/(a+b), element-wise, safe when a+b == 0."""
    denom = a + b
    out = np.zeros_like(denom)
    nz = denom > 0.0
    out[nz] = 2.0 * a[nz] * b[nz] / denom[nz]
    return out


def load_ambient() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and audit the ambient data for 0 <= t <= 1800 s.

    Raises
    ------
    ValueError
        If a column is missing, a value is absent/non-numeric, the times are
        not strictly increasing, or the record does not cover [0, 1800] s.
    """
    wb = openpyxl.load_workbook(DATA_FILE, data_only=True)
    ws = wb.active
    if ws.max_column < 3:
        raise ValueError("附件1 must contain time, temperature and moisture columns")

    times: list[float] = []
    temps: list[float] = []
    moist: list[float] = []
    for idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        if row[0] is None and row[1] is None and row[2] is None:
            continue
        if row[0] is None or row[1] is None or row[2] is None:
            raise ValueError(f"附件1 has a missing value at row {idx}")
        try:
            times.append(float(row[0]))
            temps.append(float(row[1]))
            moist.append(float(row[2]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"附件1 has a non-numeric value at row {idx}") from exc

    if not times:
        raise ValueError("附件1 contains no data rows")
    t = np.array(times)
    t_inf = np.array(temps)
    c_inf = np.array(moist)
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("附件1 times must be strictly increasing (no duplicates)")
    if t[0] > 1e-9 or t[-1] < T_END - 1e-9:
        raise ValueError("附件1 does not cover the interval [0, 1800] s")

    keep = t <= T_END + 1e-9
    return t[keep], t_inf[keep], c_inf[keep]


def build_geometry(N: int):
    """Build finite-volume geometry on r_i = i*dr (areas scaled by 1/(pi*L))."""
    dr = R / N
    r = np.arange(N + 1) * dr
    vol = np.empty(N + 1)
    vol[0] = (dr / 2.0) ** 2
    vol[1:N] = (r[1:N] + dr / 2.0) ** 2 - (r[1:N] - dr / 2.0) ** 2
    vol[N] = R**2 - (R - dr / 2.0) ** 2
    # east face area of node i (between i and i+1), i = 0..N-1
    ge = 2.0 * (r[:N] + dr / 2.0)
    # west face area of node i (between i-1 and i), i = 1..N (index 0 unused)
    gw = np.empty(N + 1)
    gw[1:] = 2.0 * (r[1:] - dr / 2.0)
    a_r = 2.0 * R  # outer surface area scaled by 1/(pi*L)
    return dr, r, vol, ge, gw, a_r


def build_tridiag(vol, ge, gw, a_r, N, dr, dt, s, h, ge_cond, gw_cond):
    """Assemble lower/diagonal/upper of the implicit finite-volume system."""
    lower = np.zeros(N + 1)
    diag = s * vol / dt
    upper = np.zeros(N + 1)

    diag[:N] += ge * ge_cond / dr
    diag[1:] += gw[1:] * gw_cond[1:] / dr
    diag[N] += a_r * h

    lower[1:] = -gw[1:] * gw_cond[1:] / dr
    upper[:N] = -ge * ge_cond / dr
    return lower, diag, upper


def to_banded(lower, diag, upper):
    """Pack a tridiagonal system into scipy banded storage (1 upper, 1 lower)."""
    n = diag.size
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return ab


def solve_tridiag(lower, diag, upper, rhs):
    """Solve a tridiagonal system via scipy's banded solver."""
    return solve_banded((1, 1), to_banded(lower, diag, upper), rhs)


def build_rhs(vol, u_old, s, dt, a_r, h, u_inf):
    """Right-hand side of the implicit finite-volume system."""
    rhs = s * vol / dt * u_old
    rhs[-1] += a_r * h * u_inf
    return rhs


def simulate(N: int = N_REF, nsub: int = NSUB, sample_step: int | None = None):
    """Run the heat/moisture simulation.

    Parameters
    ----------
    N : number of radial cells (grid nodes = N+1).
    nsub : implicit sub-steps per 1 s output interval.
    sample_step : radial stride used when returning the fields; default keeps
        the full mesh.  Use N // N_OUT to sample onto the 0.1 cm grid.

    Returns
    -------
    r : node radii [m] (sampled).
    T_hist, C_hist : (nsec+1, n_sample) arrays at integer seconds 0..1800.
    aux : diagnostic dictionary with scalar history and Picard statistics.
    """
    dr, r, vol, ge, gw, a_r = build_geometry(N)
    t_obs, t_inf_obs, c_inf_obs = load_ambient()

    dt = 1.0 / nsub
    nsec = int(round(T_END))

    T = np.full(N + 1, T_INIT)
    C = np.full(N + 1, C_INIT)

    if sample_step is None:
        sample_step = 1
    idx = np.arange(0, N + 1, sample_step)
    T_hist = np.empty((nsec + 1, idx.size))
    C_hist = np.empty((nsec + 1, idx.size))
    T_hist[0] = T[idx]
    C_hist[0] = C[idx]

    # --- temperature: constant tridiagonal matrix ---
    s_t = RHO * CP
    k_e = np.full(N, K)
    k_w = np.full(N + 1, K)
    low_t, diag_t, up_t = build_tridiag(vol, ge, gw, a_r, N, dr, dt, s_t, H, k_e, k_w)
    ab_t = to_banded(low_t, diag_t, up_t)

    # --- diagnostic scalar histories ---
    e_tot = np.empty(nsec + 1)
    m_tot = np.empty(nsec + 1)
    t_surf = np.empty(nsec + 1)
    c_surf = np.empty(nsec + 1)
    robin_t = np.zeros(nsec + 1)
    robin_c = np.zeros(nsec + 1)
    e_tot[0] = np.sum(RHO * CP * vol * T)
    m_tot[0] = np.sum(vol * C)
    t_surf[0] = T[-1]
    c_surf[0] = C[-1]

    picard_max_it = 0
    picard_max_res = 0.0

    for sec in range(1, nsec + 1):
        t_inf_end = float(np.interp(sec, t_obs, t_inf_obs))
        c_inf_end = float(np.interp(sec, t_obs, c_inf_obs))

        for sub in range(nsub):
            t_new = ((sec - 1) * nsub + sub + 1) * dt
            t_inf = np.interp(t_new, t_obs, t_inf_obs)
            c_inf = np.interp(t_new, t_obs, c_inf_obs)

            # temperature update (linear)
            rhs = build_rhs(vol, T, s_t, dt, a_r, H, t_inf)
            T = solve_banded((1, 1), ab_t, rhs)

            # moisture update (Picard on D(C))
            rhs_c = build_rhs(vol, C, 1.0, dt, a_r, HM, c_inf)
            C_new = C.copy()
            converged = False
            res = np.inf
            for it in range(1, PICARD_MAX_IT + 1):
                D = D_of_C(C_new)
                ge_cond = harmonic_mean(D[:N], D[1:])
                gw_cond = np.empty(N + 1)
                gw_cond[1:] = ge_cond
                low_c, diag_c, up_c = build_tridiag(
                    vol, ge, gw, a_r, N, dr, dt, 1.0, HM, ge_cond, gw_cond
                )
                C_sol = solve_tridiag(low_c, diag_c, up_c, rhs_c)
                res = float(np.max(np.abs(C_sol - C_new)))
                C_new = C_sol
                if res < PICARD_TOL:
                    converged = True
                    break
            if not converged:
                raise RuntimeError(
                    f"Picard iteration did not converge at t = {t_new:.3f} s: "
                    f"max residual = {res:.3e} after {PICARD_MAX_IT} iterations"
                )
            picard_max_it = max(picard_max_it, it)
            picard_max_res = max(picard_max_res, res)
            C = C_new

        T_hist[sec] = T[idx]
        C_hist[sec] = C[idx]
        e_tot[sec] = np.sum(RHO * CP * vol * T)
        m_tot[sec] = np.sum(vol * C)
        t_surf[sec] = T[-1]
        c_surf[sec] = C[-1]

        # point-wise Robin residuals using a second-order one-sided derivative
        robin_t[sec] = (
            -K * (3.0 * T[-1] - 4.0 * T[-2] + T[-3]) / (2.0 * dr)
            - H * (T[-1] - t_inf_end)
        )
        robin_c[sec] = (
            -D_of_C(np.array([C[-1]]))[0]
            * (3.0 * C[-1] - 4.0 * C[-2] + C[-3])
            / (2.0 * dr)
            - HM * (C[-1] - c_inf_end)
        )

    aux = {
        "e_tot": e_tot,
        "m_tot": m_tot,
        "t_surf": t_surf,
        "c_surf": c_surf,
        "robin_t": robin_t,
        "robin_c": robin_c,
        "picard_max_it": picard_max_it,
        "picard_max_res": picard_max_res,
        "N": N,
        "nsub": nsub,
    }
    return r[idx], T_hist, C_hist, aux


def check_balances(aux) -> tuple[float, float]:
    """Return relative energy and moisture balance errors of the discrete scheme."""
    N = aux["N"]
    _, _, _, _, _, a_r = build_geometry(N)
    t_obs, t_inf_obs, c_inf_obs = load_ambient()
    t = np.arange(aux["e_tot"].size, dtype=float)
    t_inf = np.interp(t, t_obs, t_inf_obs)
    c_inf = np.interp(t, t_obs, c_inf_obs)

    heat_in = np.sum(
        0.5
        * (
            a_r * H * (t_inf[1:] - aux["t_surf"][1:])
            + a_r * H * (t_inf[:-1] - aux["t_surf"][:-1])
        )
        * np.diff(t)
    )
    mass_in = np.sum(
        0.5
        * (
            a_r * HM * (c_inf[1:] - aux["c_surf"][1:])
            + a_r * HM * (c_inf[:-1] - aux["c_surf"][:-1])
        )
        * np.diff(t)
    )

    energy_change = aux["e_tot"][-1] - aux["e_tot"][0]
    moist_change = aux["m_tot"][-1] - aux["m_tot"][0]
    e_err = abs(energy_change - heat_in) / max(abs(energy_change), 1e-30)
    m_err = abs(moist_change - mass_in) / max(abs(moist_change), 1e-30)
    return e_err, m_err


def check_physical(T_hist, C_hist, aux) -> list[str]:
    """Check finiteness, ranges and monotonic trends; return failure messages."""
    failures: list[str] = []
    if not (np.all(np.isfinite(T_hist)) and np.all(np.isfinite(C_hist))):
        failures.append("non-finite value in temperature or moisture field")
        return failures

    if np.min(T_hist) < T_INIT - 1e-9 or np.max(T_hist) > 42.0:
        failures.append("temperature outside [28, 42] degC")
    if np.min(C_hist) < 0.0 or np.max(C_hist) > C_INIT + 1e-9:
        failures.append("moisture outside [0, 2.55] kg/kg")

    # surface-first trends: T rises with r, C falls with r (weakly, per row)
    if np.any(np.diff(T_hist, axis=1) < -1e-9):
        failures.append("temperature is not non-decreasing with radius")
    if np.any(np.diff(C_hist, axis=1) > 1e-9):
        failures.append("moisture is not non-increasing with radius")

    # time trends at the surface
    if np.any(np.diff(aux["t_surf"]) < -1e-9):
        failures.append("surface temperature decreases in time")
    if np.any(np.diff(aux["c_surf"]) > 1e-9):
        failures.append("surface moisture increases in time")
    return failures


def extract_tables(r, T_hist, C_hist):
    """Extract Table 1 (temperature) and Table 2 (moisture) for the paper."""
    t_idx = TABLE_TIMES.astype(int)
    dr = r[1] - r[0]
    r_idx = np.round(TABLE_DIST_CM / 100.0 / dr).astype(int)
    tab_t = T_hist[t_idx][:, r_idx]
    tab_c = C_hist[t_idx][:, r_idx]
    return tab_t, tab_c


def save_result(r, T_hist, C_hist) -> None:
    """Write the t = 1..1800 s fields to result1.xlsx (template layout)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dist_cm = np.round(r * 100.0, 10)

    wb = openpyxl.Workbook()
    for sheet_name, data in (("温度", T_hist), ("水分浓度", C_hist)):
        ws = wb.create_sheet(sheet_name)
        ws.cell(row=1, column=1, value="时间\\到药材中心的距离")
        for j, d in enumerate(dist_cm, start=2):
            ws.cell(row=1, column=j, value=float(d))
        for i in range(1, data.shape[0]):  # t = 1 .. 1800
            ws.cell(row=i + 1, column=1, value=float(i))
            for j in range(data.shape[1]):
                cell = ws.cell(row=i + 1, column=j + 2, value=round(float(data[i, j]), 4))
                cell.number_format = "0.0000"
    del wb["Sheet"]
    wb.save(RESULT_FILE)


def write_tables(tab_t, tab_c) -> None:
    """Save Table 1 and Table 2 as CSV files (for the paper)."""
    for fname, tab in (("table1_temperature.csv", tab_t), ("table2_moisture.csv", tab_c)):
        lines = ["时间/s," + ",".join(f"{d:g}" for d in TABLE_DIST_CM)]
        for i, t in enumerate(TABLE_TIMES):
            lines.append(f"{t}," + ",".join(f"{v:.4f}" for v in tab[i]))
        (OUT_DIR / fname).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def write_audit(lines: list[str]) -> None:
    """Persist the verification report as plain text."""
    (OUT_DIR / "verification.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify(T_hist, C_hist, aux) -> None:
    """Run the acceptance checks and raise if any hard threshold is violated."""
    report = ["Problem 1 verification", "=" * 40]
    e_err, m_err = check_balances(aux)
    report.append(f"energy balance relative error   : {e_err:.3e}")
    report.append(f"moisture balance relative error : {m_err:.3e}")
    report.append(f"Picard max iterations / residual: {aux['picard_max_it']} / {aux['picard_max_res']:.3e}")
    report.append(f"max |Robin temperature residual|: {np.max(np.abs(aux['robin_t'])):.3e} W/m^2")
    report.append(f"max |Robin moisture residual|   : {np.max(np.abs(aux['robin_c'])):.3e} kg/(kg m s)")

    failures = check_physical(T_hist, C_hist, aux)
    if e_err > 1e-4:
        failures.append(f"energy balance error {e_err:.3e} exceeds 1e-4")
    if m_err > 1e-4:
        failures.append(f"moisture balance error {m_err:.3e} exceeds 1e-4")

    report.append("physical checks: " + ("PASS" if not failures else "FAIL"))
    for msg in failures:
        report.append(f"  - {msg}")
    write_audit(report)
    print("\n".join(report))
    if failures:
        raise SystemExit("Verification failed: " + "; ".join(failures))


def run_convergence() -> None:
    """Print spatial and temporal convergence of the delivered fields."""
    print("\nSpatial convergence (sampled on the 0.1 cm grid):")
    prev = None
    for N in (100, 200, 400, 800, 1600):
        _, TT, CC, _ = simulate(N=N, nsub=NSUB, sample_step=N // N_OUT)
        if prev is not None:
            d_t = np.max(np.abs(TT - prev[0]))
            d_c = np.max(np.abs(CC - prev[1]))
            print(f"  N={N:4d} vs previous: dT = {d_t:.3e} degC, dC = {d_c:.3e} kg/kg")
        prev = (TT, CC)

    print("\nTemporal convergence (N=200, sampled on the 0.1 cm grid):")
    _, T_ref, C_ref, _ = simulate(N=200, nsub=NSUB, sample_step=200 // N_OUT)
    prev = (T_ref, C_ref)
    for nsub in (80, 160):
        _, TT, CC, _ = simulate(N=200, nsub=nsub, sample_step=200 // N_OUT)
        d_t = np.max(np.abs(TT - prev[0]))
        d_c = np.max(np.abs(CC - prev[1]))
        print(f"  nsub={nsub:3d} vs previous: dT = {d_t:.3e} degC, dC = {d_c:.3e} kg/kg")
        prev = (TT, CC)


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 1 (preheating stage).")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="also run the grid/time convergence study (slower)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # production run: refined mesh sampled onto the required 0.1 cm grid
    r, T_hist, C_hist, aux = simulate(
        N=N_REF, nsub=NSUB, sample_step=N_REF // N_OUT
    )
    save_result(r, T_hist, C_hist)

    tab_t, tab_c = extract_tables(r, T_hist, C_hist)
    write_tables(tab_t, tab_c)

    print("Table 1  Temperature [degC]")
    print("t/s   " + "".join(f"{d:>10g}" for d in TABLE_DIST_CM))
    for i, t in enumerate(TABLE_TIMES):
        print(f"{t:<5d}" + "".join(f"{v:>10.4f}" for v in tab_t[i]))

    print("\nTable 2  Moisture concentration [kg/kg]")
    print("t/s   " + "".join(f"{d:>10g}" for d in TABLE_DIST_CM))
    for i, t in enumerate(TABLE_TIMES):
        print(f"{t:<5d}" + "".join(f"{v:>10.4f}" for v in tab_c[i]))

    verify(T_hist, C_hist, aux)

    if args.verify:
        run_convergence()


if __name__ == "__main__":
    main()
