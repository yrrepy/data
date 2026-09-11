"""One-phonon inversion of a shipped MF7/MT4 table into a LEAPR phonon spectrum.

For tables whose evaluation inputs are not distributed (the JEFF-4.0 JSI
ZrH15/ZrH2 tables) the phonon spectrum rho(w) is recovered from the shipped
S(alpha,beta) itself and LEAPR is then re-run from it.  Verified 2026-09-10
(claude/leapr_proto/inversion_results_2026-09-10.txt): the four JSI tables are
reproduced to sum|dS|/sumS 7e-5 .. 1e-3 at every shipped temperature and their
temperature differences to <= 1e-3, against 1.3-2.2 % for the interpolated
first attempt.

1. Grid.  The JSI beta grids start with 200 uniform steps (0.955 meV for
   ZrH2, 0.969 meV for ZrH15) up to the end of the DOS range and continue
   geometrically to 4.96 eV.  Those uniform nodes are taken as the LEAPR rho
   grid (delta = beta step * 0.0253 eV), so the one-phonon term at a node
   depends on rho at that node alone and no interpolation enters.
2. First pass.  At the reference temperature (293.6 K when shipped) the
   smallest-alpha column is one-phonon dominated,
       S_sym(alpha_min, beta) ~ exp(-alpha lambda) alpha rho(beta) / (2 beta sinh(beta/2)),
   (alpha, beta in units of kT at that temperature; lat=1 grids are rescaled),
   which gives rho at every node directly.  The multiphonon contamination is
   alpha/beta ~ 1e-2 and independent of temperature.
3. Refinement.  LEAPR is run at the reference temperature only (about two
   seconds for a 200 x 399 grid at nphon=100) and rho is corrected with a
   diagonal Newton step, d rho_j = -(S_model - S_shipped)_j / g_j, g_j being the
   analytic one-phonon sensitivity, at every node whose shipped S exceeds 1e-6
   of the table maximum.  A node driven negative, or below 1e-6 of the peak
   while the model is too high, is set to exactly zero: this is how the band
   gap and the region above the optical band, where S is purely multiphonon,
   end up at zero as a DFT spectrum has them.  (Deciding freedom from the
   one-phonon share of the model S instead was tried and rejected: for Zr it
   drops the weak optical participation band, 1.5 % of the weight, and the
   normalisation shifts by 2 %.)  A uniform rescaling of rho is invisible to
   LEAPR (rho is normalised internally), so a uniform component of the
   residual cannot be fitted; it is reported as the scale offset (4e-5 for
   H(ZrH2), 1.6e-4 for H(ZrH15)).
4. Iterate at least min_iter times, then stop when the reference-temperature
   sum|dS|/sumS improves by less than tol between iterations, or at max_iter;
   the best iterate is kept (the H tables converge in two steps, the Zr tables,
   whose multiphonon share at alpha_min is 30 %, in about ten).

The phonon order is part of the model.  The JSI tables were produced with the
NJOY default nphon=100: a scan at 100/200/300 reproduces the high-alpha,
high-temperature corner only at 100.  The registry entry carries it.

Debye-Waller: the model's W'(T) differs from the shipped one by 1e-4 (H) to
1.2e-3 (Zr, whose S at alpha ~ 0.3 is a five-phonon quantity and therefore
hypersensitive to lambda); postedit.apply replaces W'(T) by the shipped value
at every shipped temperature and by the model value rescaled with the
interpolated shipped/model ratio elsewhere.
"""
import time
from pathlib import Path

import numpy as np

from . import decks
from .leapr import run_leapr
from .mf7 import read_mt2, read_mt4

KB = 8.617333262e-5          # eV/K, used for beta_hat in the first pass and the sensitivities
LAT_REF = decks.LAT_REF      # eV, lat=1 reference energy

DEFAULTS = dict(T_ref=293.6, nphon=100, max_iter=15, min_iter=6, tol=0.02,
                min_uniform=50, damp=(0.2, 5.0))


def uniform_head(beta, rtol=1e-3):
    """Number of leading uniform intervals of the beta grid and their step."""
    d = np.diff(np.asarray(beta, dtype=float))
    step = d[0]
    ok = np.abs(d - step) <= rtol * step
    n = len(d) if ok.all() else int(np.argmin(ok))
    return n, step


def first_pass(S_col, alpha0, bhat, ni):
    """rho on the first ni nodes from the alpha_min column (rho(0) = 0, normalised)."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        rho = np.nan_to_num(S_col[:ni] / alpha0 * 2.0 * bhat[:ni] * np.sinh(bhat[:ni] / 2.0))
    rho[0] = 0.0
    rho[rho < 0] = 0.0
    return rho


def sensitivity(ahat, lam, bhat, kT, ni):
    """Analytic one-phonon dS(alpha_min, beta_j)/d rho_j for rho normalised in eV^-1."""
    g = np.zeros(ni)
    g[1:] = ahat * np.exp(-ahat * lam) * kT / (2.0 * bhat[1:ni] * np.sinh(bhat[1:ni] / 2.0))
    return g


def solve(entry, work_dir, njoy='njoy', log_path=None, rho_path=None):
    """Recover rho for a registry entry of kind 'inverted-dos'.

    Returns dict(delta, rho, ni, nphon, T_ref, iT, history, offset, comments).
    `log_path` receives the iteration log, `rho_path` the final (w, rho) table.
    """
    p = dict(DEFAULTS)
    p.update(entry.get('invert', {}))
    shipped = Path(entry['shipped'])
    work_dir = Path(work_dir)
    d = read_mt4(shipped)
    alpha, beta, temps, S = d['alpha'], d['beta'], np.asarray(d['temps'], float), d['S']
    awr = float(d['awr'])

    nuni, step = uniform_head(beta)
    if nuni < p['min_uniform']:
        raise ValueError(f'{shipped.name}: only {nuni} uniform beta intervals, '
                         f'cannot define a node-aligned rho grid')
    ni = nuni + 1
    delta = step * LAT_REF
    w = np.arange(ni) * delta

    iT = int(np.argmin(np.abs(temps - p['T_ref'])))
    T = float(temps[iT])
    kT = KB * T
    ahat = float(alpha[0]) * LAT_REF / kT
    bhat = beta * LAT_REF / kT
    Ss = S[iT]
    Smax = Ss.max()
    col_ship = Ss[:ni, 0]

    rho = first_pass(Ss[:, 0], float(alpha[0]), bhat, ni)
    rho /= np.trapezoid(rho, w)

    free = np.zeros(ni, dtype=bool)
    free[1:] = col_ship[1:] > 1e-6 * Smax
    hist = []
    best = None            # (sumabs, rho) of the best iterate
    log = [f'inversion of {shipped.name}: reference T {T:g} K (index {iT}), '
           f'rho grid delta {delta * 1e3:.4f} meV, {ni} nodes (0..{w[-1] * 1e3:.1f} meV), '
           f'alpha_min {alpha[0]:.5g} (alpha_hat {ahat:.5f}), nphon {p["nphon"]}',
           f'{"it":>3} {"njoy s":>7} {"sum|dS|/S":>11} {"maxrel":>9} {"medrel":>9} '
           f'{"free":>5} {"zero":>5} {"col med":>9} {"offset":>10}']
    for it in range(p['max_iter'] + 1):
        deck, _meta = decks.from_uniform_dos(shipped, delta, rho, [T], p['nphon'],
                                            kind='inverted-dos', source=shipped,
                                            extra=[f'inversion iteration {it}'])
        t0 = time.time()
        tape, _out = run_leapr(deck, work_dir / f'it{it:02d}', njoy=njoy)
        dt = time.time() - t0
        m4 = read_mt4(tape)
        Sm = m4['S'][0]
        Wm = float(read_mt2(tape)['inc']['W'][0])
        lam = Wm * awr * kT
        g = sensitivity(ahat, lam, bhat, kT, ni)

        col = Sm[:ni, 0]
        mask = Ss > 1e-6 * Smax
        rel = np.abs(Sm[mask] - Ss[mask]) / Ss[mask]
        sumabs = float(np.abs(Sm - Ss).sum() / Ss.sum())
        ratio = col[free] / col_ship[free]
        offset = float(np.median(ratio) - 1.0) if free.any() else np.nan
        colmed = float(np.median(np.abs(ratio - 1.0))) if free.any() else np.nan
        hist.append(dict(it=it, seconds=dt, sumabs=sumabs, maxrel=float(rel.max()),
                         medrel=float(np.median(rel)), nfree=int(free.sum()),
                         nzero=int((rho[1:] == 0).sum()), colmed=colmed, offset=offset))
        log.append(f'{it:3d} {dt:7.1f} {sumabs:11.3e} {rel.max():9.3e} '
                   f'{np.median(rel):9.3e} {free.sum():5d} {(rho[1:] == 0).sum():5d} '
                   f'{colmed:9.3e} {offset:+10.2e}')

        if best is None or sumabs < best[0]:
            best = (sumabs, rho.copy(), it)
        if it == p['max_iter']:
            break
        if it >= p['min_iter'] and hist[-2]['sumabs'] > 0 and \
                (hist[-2]['sumabs'] - sumabs) / hist[-2]['sumabs'] < p['tol']:
            log.append(f'stop: improvement below {p["tol"]:.0%} at iteration {it}')
            break

        step_ = np.zeros(ni)
        step_[free] = (col[free] - col_ship[free]) / g[free]
        new = rho - step_
        lo, hi = p['damp']
        pos = free & (rho > 0)
        new[pos] = np.clip(new[pos], lo * rho[pos], hi * rho[pos])
        new[~free] = 0.0
        new[new < 0] = 0.0
        new[(step_ > 0) & (new < 1e-6 * rho.max())] = 0.0
        new[0] = 0.0
        rho = new / np.trapezoid(new, w)

    sumabs_best, rho, it_best = best
    log.append(f'final: best iterate {it_best}, sum|dS|/S {sumabs_best:.3e} at {T:g} K, '
               f'scale offset {hist[it_best]["offset"]:+.2e} (uniform, not fittable), '
               f'{hist[it_best]["nfree"]} free nodes, {hist[it_best]["nzero"]} zero nodes')
    text = '\n'.join(log) + '\n'
    if log_path is not None:
        Path(log_path).write_text(text)
    if rho_path is not None:
        np.savetxt(rho_path, np.c_[w, rho], fmt='%.6e',
                   header=f'w (eV)  rho (1/eV), inverted from {shipped.name} at {T:g} K')
    comments = [f'rho: one-phonon inversion of the shipped table at {T:g} K, '
                f'{it_best} newton steps, sum|dS|/S {sumabs_best:.1e}']
    return dict(delta=delta, rho=rho, ni=ni, nphon=p['nphon'], T_ref=T, iT=iT,
                history=hist, best=it_best, sumabs=sumabs_best,
                offset=hist[it_best]['offset'], comments=comments, log=text)
