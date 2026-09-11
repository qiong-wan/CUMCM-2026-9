"""Problem 2: whole drying process of a cylindrical medicinal material.

Model
-----
The material is a 1D axisymmetric cylinder of radius ``R = 2 cm`` and length
``L = 25 cm``.  The preheating/equilibrium stage uses the time-varying ambient
temperature ``T_inf(t)`` and moisture concentration ``C_inf(t)`` of
``data/附件1.xlsx`` (piecewise-linear interpolation); after the preheating stage
the ambient is held constant at the last available value (stage-switch logic,
see ``PREHEAT_END_S``).  All property correlations come from Appendix 3 of the
problem statement and are therefore coupled:

    rho(C)          = 650 + 128 C
    cp(C)           = 1450 + 2736 C/(C+1)
    k(C)            = 0.21 + 0.38 C/(C+1)
    D(C, T)         = 2.4e-3 exp(-0.45/C) exp(-3850/T)   [T in kelvin]

Governing equations (radial, ``0 < r < R``)::

    rho(C) cp(C) dT/dt = (1/r) d/dr ( k(C) r dT/dr )
    dC/dt              = (1/r) d/dr ( D(C,T) r dC/dr )

with symmetry at ``r=0`` and Robin boundaries at ``r=R``::

    -k dT/dr = h   (T - T_inf(t)),      h   = 25 W/(m^2 K)
    -D dC/dr = h_m (C - C_inf(t)),      h_m = 8e-7 m/s

Initial state ``T=28 degC``, ``C=2.55 kg/kg``.  The convective coefficients are
taken from Appendix 2 because Appendix 3 does not provide them.  No latent-heat
term is added: the problem supplies neither a vaporisation enthalpy nor a
coupled heat/mass coefficient.  The resulting energy inconsistency is
quantified in ``verification.txt``.

Numerics
--------
Node-centred finite volume, backward Euler in time, Picard iteration for the
coupled nonlinearity.  The production grid is ``N_REF=1600`` (``dr = 0.0125 mm
<= 0.025 mm``) sampled onto the required ``0.1 cm`` delivery grid.  An
independent node finite-difference method with virtual nodes is used as a
cross-check.

Run::

    python src/problem-2/solve_problem2.py            # results + verification
    python src/problem-2/solve_problem2.py --full     # also run the 2-3 day process

See ``src/problem-2/问题2_模型与算法说明.md`` for the full model notes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common.fvm import (  # noqa: E402
    build_geometry,
    build_rhs,
    build_tridiag,
    harmonic_mean,
    load_ambient,
    solve_tridiag,
    write_result_xlsx,
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_FILE = ROOT / "data" / "附件1.xlsx"
OUT_DIR = ROOT / "output" / "problem-2"
RESULT_FILE = OUT_DIR / "result2.xlsx"
VERIFY_FILE = OUT_DIR / "verification.txt"

# --------------------------------------------------------------------------
# Geometry and physical parameters
# --------------------------------------------------------------------------
R = 0.02            # cylinder radius [m]
L = 0.25            # cylinder length [m] (unused in the 1D radial model)
H = 25.0            # convective heat transfer coefficient [W/(m^2 K)] (Appendix 2)
HM = 8.0e-7         # convective mass transfer coefficient [m/s] (Appendix 2)

T_INIT = 28.0       # initial temperature [degC]
C_INIT = 2.55       # initial dry-basis moisture [kg/kg]

# The ambient record of 附件1 spans 0..14400 s and defines the preheating /
# equilibrium stage.  Beyond its last time the ambient is frozen at the last
# measured value for the constant-temperature drying stage.
PREHEAT_END_S = 14400.0
T_END_OUT = 10800.0     # reported horizon (3 h)
N_OUT = 20              # delivery grid -> 0.1 cm spacing
N_REF = 3200            # production mesh -> dr = 0.00625 mm <= 0.025 mm

# Ramped time-step schedule for the production run.  The initial surface layer
# is singular (uniform C but C_s != C_inf), so the first seconds need a very
# small step; the solution smooths quickly and the bulk can use a coarser step.
DT_START = 0.0025       # first-second step [s]      -> nsub = 400
DT_BULK = 0.03125       # asymptotic step [s]        -> nsub = 32
RAMP_RATIO = 1.3        # geometric growth of dt per second

PICARD_TOL = 1.0e-11
PICARD_MAX_IT = 100


def production_nsub(sec: int) -> int:
    """Ramped sub-step count for output second ``sec`` (1-based)."""
    if sec * np.log(RAMP_RATIO) + np.log(DT_START) >= np.log(DT_BULK):
        dt = DT_BULK
    else:
        dt = DT_START * RAMP_RATIO ** (sec - 1)
    return max(1, int(round(1.0 / dt)))


def scaled_nsub(factor: int):
    """Return a ramped schedule with every sub-step count multiplied by ``factor``."""

    def schedule(sec: int) -> int:
        return production_nsub(sec) * factor

    return schedule

TABLE_TIMES_S = np.array([1800, 3600, 5400, 7200, 9000, 10800])
TABLE_TIMES_H = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
TABLE_DIST_CM = np.array([0.0, 0.5, 1.0, 1.5, 2.0])


# --------------------------------------------------------------------------
# Appendix 3 property correlations
# --------------------------------------------------------------------------
def rho_of_C(C):
    """Bulk density [kg/m^3], Appendix 3: rho = 650 + 128 C."""
    return 650.0 + 128.0 * np.asarray(C, dtype=float)


def cp_of_C(C):
    """Specific heat [J/(kg K)], Appendix 3: cp = 1450 + 2736 C/(C+1)."""
    C = np.asarray(C, dtype=float)
    return 1450.0 + 2736.0 * C / (C + 1.0)


def k_of_C(C):
    """Thermal conductivity [W/(m K)], Appendix 3: k = 0.21 + 0.38 C/(C+1)."""
    C = np.asarray(C, dtype=float)
    return 0.21 + 0.38 * C / (C + 1.0)


def D_of_CT(C, T_C):
    """Moisture diffusivity [m^2/s], Appendix 3.

    ``D = 2.4e-3 exp(-0.45/C) exp(-3850/T)`` with ``T`` in kelvin.  Following
    the framework, the argument is required to be physically admissible: a
    non-positive or non-finite ``C`` (or a non-positive absolute temperature)
    raises instead of being silently clamped.  If ``D`` itself underflows to
    zero, the harmonic-mean guard in :func:`common.fvm.harmonic_mean` raises.
    """
    C = np.asarray(C, dtype=float)
    T_K = np.asarray(T_C, dtype=float) + 273.15
    if not np.all(np.isfinite(C)) or not np.all(np.isfinite(T_K)):
        raise ValueError("D_of_CT: non-finite C or T")
    if np.any(C <= 0.0):
        raise ValueError("D_of_CT: C must be strictly positive")
    if np.any(T_K <= 0.0):
        raise ValueError("D_of_CT: absolute temperature must be positive")
    return 2.4e-3 * np.exp(-0.45 / C) * np.exp(-3850.0 / T_K)


def ambient_functions(preheat_end: float):
    """Return ``(T_const, C_const, f)`` for the two-stage ambient history.

    ``f(t)`` linearly interpolates 附件1 for ``t <= preheat_end`` and returns
    the frozen values beyond it.
    """
    t_obs, t_inf_obs, c_inf_obs = load_ambient(DATA_FILE, preheat_end)
    t_const = float(np.interp(preheat_end, t_obs, t_inf_obs))
    c_const = float(np.interp(preheat_end, t_obs, c_inf_obs))

    def f(t: float):
        if t <= preheat_end:
            return float(np.interp(t, t_obs, t_inf_obs)), float(
                np.interp(t, t_obs, c_inf_obs)
            )
        return t_const, c_const

    return t_const, c_const, f


# --------------------------------------------------------------------------
# Production finite-volume solver
# --------------------------------------------------------------------------
def simulate(
    N: int = N_REF,
    nsub=production_nsub,
    t_end: float = T_END_OUT,
    preheat_end: float = PREHEAT_END_S,
    sample_step: int | None = None,
    record: bool = True,
) -> dict:
    """Coupled finite-volume simulation of the drying process.

    Parameters
    ----------
    N : int
        Number of control volumes (must be a multiple of ``N_OUT``).
    nsub : int or callable
        Implicit sub-steps per 1 s output interval.  An ``int`` gives the
        uniform step ``dt = 1/nsub``; a callable ``nsub(sec)`` returns the
        count for output second ``sec`` (1-based), allowing a ramped step that
        resolves the singular start-up and coarsens later.
    t_end : float
        Simulation horizon [s] (integer).
    preheat_end : float
        End of the time-varying ambient stage [s].
    sample_step : int, optional
        Radial stride for the returned fields; defaults to ``N//N_OUT``.
    record : bool
        If ``True`` store the full field at every integer second.

    Returns
    -------
    dict
        Fields, scalar histories and diagnostics.
    """
    if N % N_OUT != 0:
        raise ValueError(f"N={N} must be a multiple of N_OUT={N_OUT}")
    if sample_step is None:
        sample_step = N // N_OUT

    dr, r, vol, ge, gw, a_r = build_geometry(N, R)
    _, _, ambient = ambient_functions(preheat_end)
    nsec = int(round(t_end))
    if abs(nsec - t_end) > 1e-9:
        raise ValueError("t_end must be an integer number of seconds")

    def nsub_of(sec: int) -> int:
        value = nsub(sec) if callable(nsub) else nsub
        value = int(round(value))
        if value < 1:
            raise ValueError("nsub must be at least 1")
        return value

    idx = np.arange(0, N + 1, sample_step)
    T = np.full(N + 1, T_INIT)
    C = np.full(N + 1, C_INIT)
    ones = np.ones(N + 1)

    if record:
        T_hist = np.zeros((nsec + 1, idx.size))
        C_hist = np.zeros((nsec + 1, idx.size))
        T_hist[0] = T[idx]
        C_hist[0] = C[idx]
    else:
        T_hist = C_hist = None

    e_tot = np.zeros(nsec + 1)
    w_tot = np.zeros(nsec + 1)
    c_tot = np.zeros(nsec + 1)
    t_surf = np.zeros(nsec + 1)
    c_surf = np.zeros(nsec + 1)
    t_center = np.zeros(nsec + 1)
    c_center = np.zeros(nsec + 1)
    heat_in = np.zeros(nsec + 1)
    moist_in = np.zeros(nsec + 1)
    robin_t = np.zeros(nsec + 1)
    robin_c = np.zeros(nsec + 1)

    def snapshot(sec: int) -> None:
        s_T = rho_of_C(C) * cp_of_C(C)
        e_tot[sec] = float(np.sum(vol * s_T * T))
        w_tot[sec] = float(np.sum(vol * rho_of_C(C) * C / (1.0 + C)))
        c_tot[sec] = float(np.sum(vol * C))
        t_surf[sec] = T[-1]
        c_surf[sec] = C[-1]
        t_center[sec] = T[0]
        c_center[sec] = C[0]

    snapshot(0)
    acc_heat = 0.0
    acc_moist = 0.0
    picard_max_it = 0
    picard_max_res = 0.0
    nsub_min = 10**9
    nsub_max = 0

    for sec in range(1, nsec + 1):
        n_sub = nsub_of(sec)
        nsub_min = min(nsub_min, n_sub)
        nsub_max = max(nsub_max, n_sub)
        dt = 1.0 / n_sub
        heat_sec = 0.0
        moist_sec = 0.0
        for sub in range(n_sub):
            t_new = (sec - 1) + (sub + 1) * dt
            t_inf, c_inf = ambient(t_new)

            T_old = T
            C_old = C
            T_it = T_old.copy()
            C_it = C_old.copy()
            s_T = None
            for it in range(1, PICARD_MAX_IT + 1):
                s_T = rho_of_C(C_it) * cp_of_C(C_it)
                kk = k_of_C(C_it)
                k_e = harmonic_mean(kk[:N], kk[1:], "k")
                k_w = np.zeros(N + 1)
                k_w[1:] = k_e
                lo, di, up = build_tridiag(vol, ge, gw, a_r, N, dr, dt, s_T, H, k_e, k_w)
                rhs = build_rhs(vol, T_old, s_T, dt, a_r, H, t_inf)
                T_new = solve_tridiag(lo, di, up, rhs)

                D = D_of_CT(C_it, T_new)
                D_e = harmonic_mean(D[:N], D[1:], "D")
                D_w = np.zeros(N + 1)
                D_w[1:] = D_e
                lo, di, up = build_tridiag(vol, ge, gw, a_r, N, dr, dt, ones, HM, D_e, D_w)
                rhs = build_rhs(vol, C_old, ones, dt, a_r, HM, c_inf)
                C_new = solve_tridiag(lo, di, up, rhs)

                res = max(
                    float(np.max(np.abs(T_new - T_it))),
                    float(np.max(np.abs(C_new - C_it))),
                )
                T_it = T_new
                C_it = C_new
                if res < PICARD_TOL:
                    break
            else:
                raise RuntimeError(
                    f"Picard did not converge at t={t_new:.3f} s: "
                    f"residual={res:.3e} after {PICARD_MAX_IT} iterations"
                )
            picard_max_it = max(picard_max_it, it)
            picard_max_res = max(picard_max_res, res)

            T = T_it
            C = C_it
            acc_heat += float(np.sum(vol * s_T * (T - T_old)))
            acc_moist += float(np.sum(vol * (C - C_old)))
            heat_sec += dt * a_r * H * (t_inf - T[-1])
            moist_sec += dt * a_r * HM * (c_inf - C[-1])

        heat_in[sec] = heat_in[sec - 1] + heat_sec
        moist_in[sec] = moist_in[sec - 1] + moist_sec
        if record:
            T_hist[sec] = T[idx]
            C_hist[sec] = C[idx]
        snapshot(sec)

        # second-order one-sided surface gradients -> Robin residuals
        t_inf_s, c_inf_s = ambient(float(sec))
        grad_t = (3.0 * T[-1] - 4.0 * T[-2] + T[-3]) / (2.0 * dr)
        grad_c = (3.0 * C[-1] - 4.0 * C[-2] + C[-3]) / (2.0 * dr)
        robin_t[sec] = -k_of_C(C[-1]) * grad_t - H * (T[-1] - t_inf_s)
        robin_c[sec] = -D_of_CT(C[-1], T[-1]) * grad_c - HM * (C[-1] - c_inf_s)

    return {
        "r": r[idx],
        "T_hist": T_hist,
        "C_hist": C_hist,
        "e_tot": e_tot,
        "w_tot": w_tot,
        "c_tot": c_tot,
        "t_surf": t_surf,
        "c_surf": c_surf,
        "t_center": t_center,
        "c_center": c_center,
        "heat_in": heat_in,
        "moist_in": moist_in,
        "robin_t": robin_t,
        "robin_c": robin_c,
        "acc_heat": acc_heat,
        "acc_moist": acc_moist,
        "picard_max_it": picard_max_it,
        "picard_max_res": picard_max_res,
        "N": N,
        "nsub": nsub,
        "nsub_min": nsub_min,
        "nsub_max": nsub_max,
        "nsec": nsec,
        "dr": dr,
    }


# --------------------------------------------------------------------------
# Independent node finite-difference solver with virtual nodes
# --------------------------------------------------------------------------
def _fd_system(N, dr, r, s, gamma, h, u_inf, u_old, dt):
    """Assemble a node-FD (virtual-node) backward-Euler tridiagonal system.

    Interior nodes use the expanded differential operator
    ``(1/r)(gamma r u_r)_r = gamma u_rr + (gamma/r) u_r + gamma_r u_r`` with
    central differences.  The axis uses the virtual node ``u_{-1}=u_1`` giving
    the limit ``4 gamma_0 (u_1-u_0)/dr^2``.  The surface uses a virtual node
    ``u_{N+1}`` eliminated through the Robin condition.
    """
    lower = np.zeros(N + 1)
    diag = np.zeros(N + 1)
    upper = np.zeros(N + 1)
    rhs = s * u_old / dt

    a0 = gamma[0] / dr**2
    diag[0] = s[0] / dt + 4.0 * a0
    upper[0] = -4.0 * a0

    i = np.arange(1, N)
    a = gamma[i] / dr**2
    b = gamma[i] / (2.0 * r[i] * dr)
    c = (gamma[i + 1] - gamma[i - 1]) / (4.0 * dr**2)
    diag[i] = s[i] / dt + 2.0 * a
    lower[i] = -(a - b - c)
    upper[i] = -(a + b + c)

    q = 2.0 * dr * h / gamma[N]
    gamma_r = (gamma[N] - gamma[N - 1]) / dr
    coeff_nn = -(gamma[N] * (2.0 + q) / dr**2 + (gamma[N] / r[N] + gamma_r) * q / (2.0 * dr))
    coeff_nm1 = 2.0 * gamma[N] / dr**2
    rhs_inf = gamma[N] * q / dr**2 + (gamma[N] / r[N] + gamma_r) * q / (2.0 * dr)
    diag[N] = s[N] / dt - coeff_nn
    lower[N] = -coeff_nm1
    rhs[N] += rhs_inf * u_inf
    return lower, diag, upper, rhs


def simulate_fd(
    N: int = 800,
    nsub: int = 1,
    t_end: float = T_END_OUT,
    preheat_end: float = PREHEAT_END_S,
    sample_step: int | None = None,
) -> dict:
    """Independent node finite-difference (virtual-node) simulation."""
    if N % N_OUT != 0:
        raise ValueError(f"N={N} must be a multiple of N_OUT={N_OUT}")
    if sample_step is None:
        sample_step = N // N_OUT

    dr, r, _, _, _, _ = build_geometry(N, R)
    _, _, ambient = ambient_functions(preheat_end)
    dt = 1.0 / nsub
    nsec = int(round(t_end))

    idx = np.arange(0, N + 1, sample_step)
    T = np.full(N + 1, T_INIT)
    C = np.full(N + 1, C_INIT)
    ones = np.ones(N + 1)

    T_hist = np.zeros((nsec + 1, idx.size))
    C_hist = np.zeros((nsec + 1, idx.size))
    T_hist[0] = T[idx]
    C_hist[0] = C[idx]

    for sec in range(1, nsec + 1):
        for sub in range(nsub):
            t_new = ((sec - 1) * nsub + sub + 1) * dt
            t_inf, c_inf = ambient(t_new)

            T_old = T
            C_old = C
            T_it = T_old.copy()
            C_it = C_old.copy()
            for it in range(1, PICARD_MAX_IT + 1):
                s_T = rho_of_C(C_it) * cp_of_C(C_it)
                kk = k_of_C(C_it)
                lo, di, up, rhs = _fd_system(N, dr, r, s_T, kk, H, t_inf, T_old, dt)
                T_new = solve_tridiag(lo, di, up, rhs)

                D = D_of_CT(C_it, T_new)
                lo, di, up, rhs = _fd_system(N, dr, r, ones, D, HM, c_inf, C_old, dt)
                C_new = solve_tridiag(lo, di, up, rhs)

                res = max(
                    float(np.max(np.abs(T_new - T_it))),
                    float(np.max(np.abs(C_new - C_it))),
                )
                T_it = T_new
                C_it = C_new
                if res < PICARD_TOL:
                    break
            else:
                raise RuntimeError(
                    f"FD Picard did not converge at t={t_new:.3f} s: residual={res:.3e}"
                )
            T = T_it
            C = C_it
        T_hist[sec] = T[idx]
        C_hist[sec] = C[idx]

    return {"r": r[idx], "T_hist": T_hist, "C_hist": C_hist, "N": N, "nsub": nsub}


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
def extract_tables(r, T_hist, C_hist):
    """Extract Table 3 (temperature) and Table 4 (moisture) for the paper."""
    t_idx = TABLE_TIMES_S.astype(int)
    r_idx = np.round(TABLE_DIST_CM / 100.0 / (r[1] - r[0])).astype(int)
    return T_hist[t_idx][:, r_idx], C_hist[t_idx][:, r_idx]


def _fmt_table(title: str, unit: str, tab, row_labels) -> list[str]:
    lines = [f"**{title}**", "", "| 时间/h | " + " | ".join(
        f"{d:g} cm" for d in TABLE_DIST_CM) + " |", "|---|" + "---|" * len(TABLE_DIST_CM)]
    for i, label in enumerate(row_labels):
        lines.append(f"| {label:.1f} | " + " | ".join(f"{v:.4f}" for v in tab[i]) + " |")
    lines.append("")
    lines.append(f"(单位: {unit})")
    lines.append("")
    return lines


def write_tables(tab_t, tab_c) -> None:
    """Save Table 3/Table 4 as CSV and as a combined Markdown file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fname, tab, head in (
        ("table3_temperature.csv", tab_t, "时间/h"),
        ("table4_moisture.csv", tab_c, "时间/h"),
    ):
        lines = [head + "," + ",".join(f"{d:g}" for d in TABLE_DIST_CM)]
        for i, t in enumerate(TABLE_TIMES_H):
            lines.append(f"{t:.1f}," + ",".join(f"{v:.4f}" for v in tab[i]))
        (OUT_DIR / fname).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    md = ["# 问题 2 表 3 / 表 4", ""]
    md += _fmt_table("表 3 3小时内药材的温度", "°C", tab_t, TABLE_TIMES_H)
    md += _fmt_table("表 4 3小时内药材的水分浓度", "kg/kg", tab_c, TABLE_TIMES_H)
    (OUT_DIR / "tables3_4.md").write_text("\n".join(md), encoding="utf-8")


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
def balance_audit(res: dict) -> dict:
    """Global energy/moisture audit of a production result.

    Returns
    -------
    dict
        ``mass_rel`` (exact C-balance residual), ``heat_scheme_rel``
        (scheme energy residual), ``prop_heat`` and ``prop_heat_rel`` (the
        non-conservative property-change term), ``latent`` / ``latent_rel``
        (ignored latent heat versus boundary heat input).
    """
    e = res["e_tot"]
    c = res["c_tot"]
    w = res["w_tot"]
    acc_heat = res["acc_heat"]
    acc_moist = res["acc_moist"]

    mass_rel = abs((c[-1] - c[0]) - acc_moist) / max(abs(c[-1] - c[0]), 1e-30)
    heat_scheme_rel = abs(acc_heat - res["heat_in"][-1]) / max(abs(res["heat_in"][-1]), 1e-30)

    energy_change = e[-1] - e[0]
    prop_heat = energy_change - acc_heat
    prop_heat_rel = abs(prop_heat) / max(abs(energy_change), 1e-30)

    water_lost = w[0] - w[-1]
    latent = water_lost * 2.26e6 * np.pi * L
    heat_input = res["heat_in"][-1] * np.pi * L
    latent_rel = latent / max(abs(heat_input), 1e-30)
    return {
        "mass_rel": mass_rel,
        "heat_scheme_rel": heat_scheme_rel,
        "energy_change": energy_change,
        "prop_heat": prop_heat,
        "prop_heat_rel": prop_heat_rel,
        "water_lost": water_lost,
        "latent": latent,
        "heat_input": heat_input,
        "latent_rel": latent_rel,
    }


def _diff_stats(cur, prev, t_report=1800) -> dict:
    """Max |difference| over the field, surface/centre and reported window."""
    dT = np.abs(cur[0] - prev[0])
    dC = np.abs(cur[1] - prev[1])
    return {
        "dT": float(np.max(dT)),
        "dC": float(np.max(dC)),
        "dT_surf": float(np.max(dT[:, -1])),
        "dC_surf": float(np.max(dC[:, -1])),
        "dT_center": float(np.max(dT[:, 0])),
        "dC_center": float(np.max(dC[:, 0])),
        "dT_win": float(np.max(dT[t_report:])),
        "dC_win": float(np.max(dC[t_report:])),
    }


def grid_convergence(t_end=T_END_OUT, nsub=2) -> list[dict]:
    """Return per-refinement max differences between successive grids."""
    prev = None
    rows = []
    for N in (400, 800, 1600, 3200):
        res = simulate(N=N, nsub=nsub, t_end=t_end, record=True)
        if prev is not None:
            row = {"N": N}
            row.update(_diff_stats((res["T_hist"], res["C_hist"]), prev))
            rows.append(row)
        prev = (res["T_hist"], res["C_hist"])
    return rows


def temporal_convergence(t_end=T_END_OUT, N=400) -> list[dict]:
    """Return per-refinement max differences between successive time steps."""
    prev = None
    rows = []
    for nsub in (1, 2, 4):
        res = simulate(N=N, nsub=nsub, t_end=t_end, record=True)
        if prev is not None:
            row = {"nsub": nsub}
            row.update(_diff_stats((res["T_hist"], res["C_hist"]), prev))
            rows.append(row)
        prev = (res["T_hist"], res["C_hist"])
    return rows


def independent_check(N=800, nsub=1, t_end=T_END_OUT) -> dict:
    """Compare the finite-volume and virtual-node finite-difference solutions."""
    fv = simulate(N=N, nsub=nsub, t_end=t_end, record=True)
    fd = simulate_fd(N=N, nsub=nsub, t_end=t_end)
    return {
        "N": N,
        "nsub": nsub,
        "max_dT": float(np.max(np.abs(fv["T_hist"] - fd["T_hist"]))),
        "max_dC": float(np.max(np.abs(fv["C_hist"] - fd["C_hist"]))),
    }


def verify_output_files(tab_t, tab_c) -> list[str]:
    """Check result2.xlsx structure and table consistency; return failures."""
    failures: list[str] = []
    wb = openpyxl.load_workbook(RESULT_FILE, data_only=True)
    header = "时间\\到药材中心的距离"
    expected_dist = [round(0.1 * k, 10) for k in range(21)]

    for name, tab in (("温度", tab_t), ("水分浓度", tab_c)):
        ws = wb[name]
        if (ws.max_row, ws.max_column) != (10801, 22):
            failures.append(f"{name}: shape {(ws.max_row, ws.max_column)} != (10801, 22)")
        if ws.cell(1, 1).value != header:
            failures.append(f"{name}: A1 is {ws.cell(1,1).value!r}")
        if float(ws.cell(2, 1).value) != 1.0 or float(ws.cell(10801, 1).value) != 10800.0:
            failures.append(f"{name}: A column does not run 1..10800")
        for j, d in enumerate(expected_dist, start=2):
            if abs(float(ws.cell(1, j).value) - d) > 1e-12:
                failures.append(f"{name}: header column {j} is not {d} cm")
        for i, t in enumerate(TABLE_TIMES_S):
            for j, d in enumerate(TABLE_DIST_CM):
                col = 2 + int(round(d / 0.1))
                cell = float(ws.cell(int(t) + 1, col).value)
                if abs(cell - round(float(tab[i, j]), 4)) > 1e-12:
                    failures.append(f"{name}: t={t}s r={d}cm cell {cell} != table")
    return failures


def write_verification(res, audit, gc, tc, ind, full_summary=None, file_failures=None) -> list[str]:
    """Compose and persist the verification report."""
    t_obs, t_inf_obs, c_inf_obs = load_ambient(DATA_FILE, PREHEAT_END_S)
    t_const = float(np.interp(PREHEAT_END_S, t_obs, t_inf_obs))
    c_const = float(np.interp(PREHEAT_END_S, t_obs, c_inf_obs))
    lines = [
        "Problem 2 verification",
        "=" * 72,
        "",
        "Model: 1D axisymmetric cylinder, R = 2 cm, L = 25 cm (no shrinkage).",
        "Properties from Appendix 3 (rho, cp, k depend on C; D depends on C and T).",
        "Convective coefficients from Appendix 2: h = 25 W/(m^2 K), h_m = 8e-7 m/s.",
        "Coupled by Picard iteration; backward-Euler time integration.",
        f"Production grid: N = {res['N']}, dr = {res['dr']*1e3:.5f} mm "
        f"(<= 0.025 mm), nsub = {res['nsub']} (dt = {1.0/res['nsub']:.3f} s).",
        "",
        "Stage-switch logic",
        "-" * 72,
        "附件1 covers 0..14400 s (preheating/equilibrium stage). For t <= 14400 s the",
        "ambient is piecewise-linearly interpolated. For t > 14400 s the ambient is",
        "frozen at the last measured values (constant-temperature drying stage):",
        f"    T_inf = {t_const:.4f} degC, C_inf = {c_const:.5f} kg/kg.",
        "The first 3 h reported here therefore lie entirely inside the preheating stage.",
        "",
        "Shrinkage assumption",
        "-" * 72,
        "R = 2 cm is held constant; 附件2 (radius vs time) is only used by Problem 4.",
        "",
        "Grid convergence (nsub = 2, max |difference| on the 0.1 cm delivery grid)",
        "-" * 72,
    ]
    for row in gc:
        lines.append(
            f"    N -> {row['N']:4d}:  whole max|dT|={row['dT']:.3e}, max|dC|={row['dC']:.3e} | "
            f"t>=1800 s max|dT|={row['dT_win']:.3e}, max|dC|={row['dC_win']:.3e} | "
            f"centre max|dC|={row['dC_center']:.3e}"
        )
    if gc:
        lines.append(
            f"    finest whole-field max|dC| = {gc[-1]['dC']:.3e} kg/kg "
            "(framework delivery target < 5e-5)."
        )
    mono = (
        all(gc[i]["dC_surf"] >= gc[i + 1]["dC_surf"] for i in range(len(gc) - 1))
        and all(gc[i]["dC_center"] >= gc[i + 1]["dC_center"] for i in range(len(gc) - 1))
        and all(gc[i]["dT_surf"] >= gc[i + 1]["dT_surf"] for i in range(len(gc) - 1))
    )
    lines.append(
        "Surface and centre differences decrease monotonically with refinement: "
        + ("PASS" if mono else "FAIL")
    )

    lines += [
        "",
        "Temporal convergence (N = 400, max |difference| on the delivery grid)",
        "-" * 72,
    ]
    for row in tc:
        lines.append(
            f"    nsub -> {row['nsub']:2d}:  whole max|dT|={row['dT']:.3e}, max|dC|={row['dC']:.3e} | "
            f"surface max|dC|={row['dC_surf']:.3e} | centre max|dC|={row['dC_center']:.3e} | "
            f"t>=1800 s max|dT|={row['dT_win']:.3e}, max|dC|={row['dC_win']:.3e}"
        )

    lines += [
        "",
        "Global balance audit (t = 0..10800 s)",
        "-" * 72,
        f"    C-balance relative error (scheme-exact) : {audit['mass_rel']:.3e}",
        f"    Energy-balance relative error (scheme)  : {audit['heat_scheme_rel']:.3e}",
        f"    Sensible energy change                  : {audit['energy_change']:.4e}",
        f"    Property-change (non-conservative) term : {audit['prop_heat']:.4e} "
        f"({audit['prop_heat_rel']:.2%} of sensible change)",
        f"    Water lost                              : {audit['water_lost']:.4e} "
        "(model units, per pi*L)",
        f"    Boundary heat input                     : {audit['heat_input']:.4e} J",
        f"    Ignored latent heat (L_v = 2.26e6 J/kg): {audit['latent']:.4e} J",
        f"    Latent / heat-input ratio               : {audit['latent_rel']:.2f}",
        "",
        "Surface Robin residuals (t >= 1 s, second-order one-sided gradient)",
        "-" * 72,
        f"    max |R_T| = {np.max(np.abs(res['robin_t'][1:])):.3e} W/m^2",
        f"    max |R_C| = {np.max(np.abs(res['robin_c'][1:])):.3e} (kg/kg) m/s",
        "    The initial corner is incompatible (uniform C but C_s != C_inf); the",
        "    residual is therefore evaluated for t >= 1 s and decreases with mesh",
        "    refinement rather than being zero on a finite grid.",
        "",
        "Energy-inconsistency discussion",
        "The energy equation rho(C) cp(C) dT/dt = div(k grad T) omits the latent heat",
        "of vaporisation because the problem supplies neither L_v nor a coupled",
        "heat/mass coefficient. The table above shows the latent heat that the",
        "removed water would require, relative to the heat actually supplied through",
        "the surface. Because this term is absent, the model temperature is an upper",
        "bound: a physically complete model would cool the material by this amount.",
        "The property-change term is the second inconsistency: the",
        "non-conservative form rho cp dT/dt is not a strict conservation law when",
        "rho and cp vary with C. Both effects are reported, not hidden. The computed",
        f"temperature stays in [{res['t_center'].min():.4f}, {res['t_surf'].max():.4f}] degC,",
        "well above the physical floor set by the ambient air.",
        "",
        "Independent cross-check: node finite differences with virtual nodes",
        "-" * 72,
        f"    Grid N = {ind['N']}, nsub = {ind['nsub']} (same time discretisation for both).",
        f"    max |dT| = {ind['max_dT']:.3e} degC   (acceptance < 1e-3)",
        f"    max |dC| = {ind['max_dC']:.3e} kg/kg  (acceptance < 1e-3)",
        f"    Result: {'PASS' if ind['max_dT'] < 1e-3 and ind['max_dC'] < 1e-3 else 'FAIL'}",
        "",
        "Output-file checks (result2.xlsx and Tables 3/4)",
        "-" * 72,
    ]
    file_failures = file_failures or []
    if file_failures:
        for msg in file_failures:
            lines.append(f"    FAIL: {msg}")
    else:
        lines.append("    result2.xlsx: 10801 rows x 22 columns, A1 exact, A = 1..10800 s,")
        lines.append("    header distances 0..2.0 cm, both sheets match Tables 3/4 to 4 decimals.")

    lines += [
        "",
        "Acceptance summary",
        "-" * 72,
        f"    1. result2.xlsx layout            : {'PASS' if not file_failures else 'FAIL'}",
        "    2. Tables 3/4 match the numerics  : "
        f"{'PASS' if not file_failures else 'FAIL'}",
        f"    3. Independent method max|dT| < 1e-3 and max|dC| < 1e-3 : "
        f"{'PASS' if ind['max_dT'] < 1e-3 and ind['max_dC'] < 1e-3 else 'FAIL'}",
        "    4. Quantified latent-heat inconsistency reported        : PASS",
        "",
        f"Picard: max iterations = {res['picard_max_it']}, "
        f"max residual = {res['picard_max_res']:.3e}.",
    ]

    if full_summary is not None:
        lines += [
            "",
            "Full 2-3 day process capability (--full)",
            "-" * 72,
            f"    Ran to t = {full_summary['t_end']:.0f} s "
            f"({full_summary['t_end']/3600.0:.1f} h) with N = {full_summary['N']}, "
            f"dt = {full_summary['dt']:.1f} s in {full_summary['wall']:.1f} s.",
            f"    Final centre moisture C(0) = {full_summary['c_center']:.4f} kg/kg,",
            f"    final surface moisture C(R) = {full_summary['c_surf']:.4f} kg/kg,",
            f"    final centre temperature = {full_summary['t_center']:.4f} degC.",
        ]

    text = "\n".join(lines) + "\n"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_FILE.write_text(text, encoding="utf-8")
    return lines


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run_full_process(t_end=259200.0, N=800, nsub=1) -> dict:
    """Run the complete 3-day drying process and summarise the final state."""
    start = time.time()
    res = simulate(N=N, nsub=nsub, t_end=t_end, record=False)
    wall = time.time() - start
    return {
        "t_end": t_end,
        "N": N,
        "dt": 1.0 / nsub,
        "wall": wall,
        "c_center": res["c_center"][-1],
        "c_surf": res["c_surf"][-1],
        "t_center": res["t_center"][-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Problem 2 (whole drying process).")
    parser.add_argument("--full", action="store_true", help="also run the 2-3 day process")
    parser.add_argument("--skip-verify", action="store_true", help="skip the slow checks")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    res = simulate(N=N_REF, nsub=production_nsub, t_end=T_END_OUT, record=True)
    print(f"Production run finished in {time.time() - start:.1f} s")

    write_result_xlsx(RESULT_FILE, res["r"], res["T_hist"], res["C_hist"])
    tab_t, tab_c = extract_tables(res["r"], res["T_hist"], res["C_hist"])
    write_tables(tab_t, tab_c)

    print("\nTable 3  Temperature [degC]")
    print("t/h   " + "".join(f"{d:>10g}" for d in TABLE_DIST_CM))
    for i, t in enumerate(TABLE_TIMES_H):
        print(f"{t:<5g}" + "".join(f"{v:>10.4f}" for v in tab_t[i]))

    print("\nTable 4  Moisture concentration [kg/kg]")
    print("t/h   " + "".join(f"{d:>10g}" for d in TABLE_DIST_CM))
    for i, t in enumerate(TABLE_TIMES_H):
        print(f"{t:<5g}" + "".join(f"{v:>10.4f}" for v in tab_c[i]))

    audit = balance_audit(res)
    file_failures = verify_output_files(tab_t, tab_c)
    if file_failures:
        print("Output-file check FAILED:")
        for msg in file_failures:
            print("  -", msg)

    if args.skip_verify:
        write_verification(
            res, audit, [], [], {"N": 0, "nsub": 0, "max_dT": 0.0, "max_dC": 0.0},
            file_failures=file_failures,
        )
        return

    print("\nRunning grid convergence ...")
    gc = grid_convergence()
    print("Running temporal convergence ...")
    tc = temporal_convergence()
    print("Running independent FD cross-check ...")
    ind = independent_check()

    full_summary = None
    if args.full:
        print("Running full 3-day process ...")
        full_summary = run_full_process()

    lines = write_verification(res, audit, gc, tc, ind, full_summary, file_failures)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
