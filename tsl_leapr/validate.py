"""Validation gate: regenerated (post-edited) tape versus the shipped evaluation.

The shipped temperatures are the built-in validation set: the regenerated tape
contains all of them, so every shipped S(alpha,beta) block and every shipped
W'(T) must be reproduced.  Metrics follow claude/leapr_proto/compare_mf7.py.
"""
import numpy as np

from .mf7 import read_mt2, read_mt4
from .postedit import (DW_FACTOR, _reference_shipped, _shipped_b1_text,
                       _shipped_sigma_b_text)

GATE_SUM_ABS = 1.0e-4      # sum|dS| / sum S at every shipped temperature
GATE_WPRIME = 1.0e-3       # |dW'| / W' at every shipped temperature
GATE_LOO = 1.0e-4          # LTHR=3 leave-one-out Debye-Waller rescale
GATE_DT = 3.0e-3           # adjacent-temperature difference, inverted-dos kind only

# Tables rebuilt from a phonon spectrum recovered from the shipped S(alpha,beta)
# (kind 'inverted-dos') are approximate: the recovered spectrum reproduces the
# JSI tables to 7e-5 .. 1.1e-3 (measured 2026-09-10), limited by a uniform scale
# offset the normalised spectrum cannot fit.  Their post-edit rescales the
# delivered tape to the shipped values at every shipped temperature, so for
# this kind the gates below apply to the LEAPR tape BEFORE the post-edit (the
# model fidelity, meta['model_result']), and the delivered tape must reproduce
# the shipped blocks to GATE_COPY.  The adjacent-temperature difference (the
# quantity a finite-difference temperature coefficient uses) is gated as well.
GATE_SUM_ABS_KIND = {'inverted-dos': 2.0e-3}
GATE_COPY = 1.0e-5


def _rows_mt4(new, ship, drop=None):
    """Per-shipped-temperature MT4 metrics; `drop` is a beta index to exclude."""
    rows = []
    keep = np.ones(len(ship['beta']), dtype=bool)
    if drop is not None:
        keep[drop] = False
    for j, T in enumerate(ship['temps']):
        i = int(np.argmin(np.abs(np.asarray(new['temps']) - T)))
        Sa, Sb = new['S'][i], ship['S'][j]
        m = Sb > 1e-6 * Sb.max()
        rel = np.abs(Sa[m] - Sb[m]) / Sb[m]
        row = dict(T=float(T), T_new=float(new['temps'][i]),
                   sumabs=float(np.abs(Sa - Sb).sum() / Sb.sum()),
                   maxabs=float(np.abs(Sa - Sb).max() / Sb.max()),
                   maxrel=float(rel.max()), medrel=float(np.median(rel)),
                   n1pct=int((rel > 0.01).sum()))
        if drop is not None:
            Sa2, Sb2 = Sa[keep], Sb[keep]
            m2 = Sb2 > 1e-6 * Sb2.max()
            rel2 = np.abs(Sa2[m2] - Sb2[m2]) / Sb2[m2]
            row['sumabs_cut'] = float(np.abs(Sa2 - Sb2).sum() / Sb2.sum())
            row['maxrel_cut'] = float(rel2.max())
            row['n1pct_cut'] = int((rel2 > 0.01).sum())
        rows.append(row)
    return rows


def _rows_dt(new, ship):
    """Adjacent shipped-temperature differences: sum|d(dS)| / sum|dS| and the
    median relative error of dS where |dS| is above 1e-3 of its maximum."""
    rows = []
    Tn = np.asarray(new['temps'], dtype=float)
    Ts = np.asarray(ship['temps'], dtype=float)
    for j in range(len(Ts) - 1):
        i0 = int(np.argmin(np.abs(Tn - Ts[j])))
        i1 = int(np.argmin(np.abs(Tn - Ts[j + 1])))
        dSa = new['S'][i1] - new['S'][i0]
        dSb = ship['S'][j + 1] - ship['S'][j]
        m = np.abs(dSb) > 1e-3 * np.abs(dSb).max()
        rows.append(dict(T0=float(Ts[j]), T1=float(Ts[j + 1]),
                         sumabs=float(np.abs(dSa - dSb).sum() / np.abs(dSb).sum()),
                         medrel=float(np.median(np.abs(dSa[m] - dSb[m]) / np.abs(dSb[m])))))
    return rows


def _rows_mt2(new, ship):
    rows = []
    Tn = np.asarray(new['inc']['temps'], dtype=float)
    Wn = np.asarray(new['inc']['W'], dtype=float)
    for T, W in zip(np.asarray(ship['inc']['temps'], dtype=float),
                    np.asarray(ship['inc']['W'], dtype=float)):
        i = int(np.argmin(np.abs(Tn - T)))
        rows.append(dict(T=float(T), w_new=float(Wn[i]), w_ship=float(W),
                         rel=float(abs(Wn[i] - W) / W)))
    return rows


def _rows_loo(new, ship, temps):
    """LTHR=3 leave-one-out: predict each shipped Bragg column from the nearest
    shipped column BELOW it (the production rule, postedit._reference_shipped)
    using LEAPR's dW'.

    Rescaling upward is accurate to ~5e-7; rescaling downward over a large gap
    amplifies the rounding of the shipped seven-digit cumulative columns (293.6
    -> 77 K measured 4.9e-3), which is why production never does it.  The
    lowest shipped temperature has no lower reference; it is predicted from the
    nearest column instead and reported but not gated ('in_gate' False), since
    in production it is copied verbatim.
    """
    coh = ship['coh']
    E, S, Tsh = coh['E'], np.asarray(coh['S'], dtype=float), np.asarray(coh['temps'])
    Tt = np.asarray(temps, dtype=float)
    W = np.asarray(new['inc']['W'], dtype=float)
    rows = []
    for j, T in enumerate(Tsh):
        others = np.delete(np.arange(len(Tsh)), j)
        k = int(others[_reference_shipped(T, Tsh[others])])
        in_gate = bool((Tsh[others] < T).any())
        ij = int(np.argmin(np.abs(Tt - Tsh[j])))
        ik = int(np.argmin(np.abs(Tt - Tsh[k])))
        s = np.diff(np.concatenate([[0.0], S[k]]))
        pred = np.cumsum(s * np.exp(-DW_FACTOR * (W[ij] - W[ik]) * E))
        m = S[j] > 1e-3 * S[j][-1]
        rel = np.abs(pred[m] - S[j][m]) / S[j][m]
        rows.append(dict(T=float(T), T_ref=float(Tsh[k]),
                         maxrel=float(rel.max()), medrel=float(np.median(rel)),
                         ratio_emax=float(pred[-1] / S[j][-1]),
                         in_gate=in_gate))
    return rows


def compare(entry, tape, temps, meta):
    """All metrics plus the pass/fail decision for one regenerated table."""
    shipped = entry['shipped']
    new4, ship4 = read_mt4(tape), read_mt4(shipped)
    new2, ship2 = read_mt2(tape), read_mt2(shipped)

    same_grid = (len(new4['alpha']) == len(ship4['alpha'])
                 and len(new4['beta']) == len(ship4['beta'])
                 and np.allclose(new4['alpha'], ship4['alpha'], rtol=1e-5)
                 and np.allclose(new4['beta'], ship4['beta'], rtol=1e-5))
    have_temps = all(np.isclose(np.asarray(new4['temps']), T, rtol=1e-6,
                                atol=1e-6).any() for T in ship4['temps'])

    drop = None
    if meta.get('beta_cut'):
        drop = int(np.argmin(np.abs(np.asarray(ship4['beta']) - meta['beta_cut'])))

    res = dict(lib=entry['lib'], name=entry['name'], kind=entry['kind'],
               nT=len(temps), temps=list(map(float, temps)), same_grid=same_grid,
               have_temps=have_temps, beta_cut=meta.get('beta_cut'),
               drop_beta=None if drop is None else float(ship4['beta'][drop]),
               b1_new=_shipped_b1_text(tape), b1_ship=_shipped_b1_text(shipped),
               sb_new=_shipped_sigma_b_text(tape),
               sb_ship=_shipped_sigma_b_text(shipped))
    res['mt4'] = _rows_mt4(new4, ship4, drop) if same_grid else []
    res['dt'] = _rows_dt(new4, ship4) if same_grid else []
    res['mt2'] = _rows_mt2(new2, ship2)
    res['loo'] = _rows_loo(new2, ship2, temps) if entry['lthr3'] else []
    res['gate_sumabs'] = GATE_SUM_ABS_KIND.get(entry['kind'], GATE_SUM_ABS)
    res['inversion'] = meta.get('inversion')
    res['model'] = meta.get('model_result')      # raw-tape metrics, inverted-dos only

    key = 'sumabs_cut' if drop is not None else 'sumabs'
    res['worst_sumabs'] = max((r[key] for r in res['mt4']), default=np.inf)
    res['worst_sumabs_all'] = max((r['sumabs'] for r in res['mt4']), default=np.inf)
    res['worst_wrel'] = max((r['rel'] for r in res['mt2']), default=np.inf)
    res['worst_loo'] = max((r['maxrel'] for r in res['loo'] if r['in_gate']),
                           default=0.0)
    res['worst_loo_all'] = max((r['maxrel'] for r in res['loo']), default=0.0)
    res['worst_dt'] = max((r['sumabs'] for r in res['dt']), default=np.inf)

    fails = []
    if not same_grid:
        fails.append('alpha/beta grid differs from shipped')
    if not have_temps:
        fails.append('not every shipped temperature is present')
    if entry['kind'] == 'inverted-dos' and res['model'] is not None:
        m = res['model']
        if m['worst_sumabs'] >= res['gate_sumabs']:
            fails.append(f'model MT4 sum|dS|/sumS {m["worst_sumabs"]:.3e} >= {res["gate_sumabs"]:g}')
        if m['worst_dt'] >= GATE_DT:
            fails.append(f'model adjacent-T difference {m["worst_dt"]:.3e} >= {GATE_DT:g}')
        if res['worst_sumabs'] >= GATE_COPY:
            fails.append(f'delivered MT4 sum|dS|/sumS {res["worst_sumabs"]:.3e} >= {GATE_COPY:g} '
                         f'(rescale to shipped failed)')
    else:
        if res['worst_sumabs'] >= res['gate_sumabs']:
            fails.append(f'MT4 sum|dS|/sumS {res["worst_sumabs"]:.3e} >= {res["gate_sumabs"]:g}')
    if res['worst_wrel'] >= GATE_WPRIME:
        fails.append(f"MT2 W' rel {res['worst_wrel']:.3e} >= {GATE_WPRIME:g}")
    if res['b1_new'].strip() != res['b1_ship'].strip():
        fails.append(f'B(1) {res["b1_new"]!r} != shipped {res["b1_ship"]!r}')
    if res['sb_new'].strip() != res['sb_ship'].strip():
        fails.append(f'sigma_b {res["sb_new"]!r} != shipped {res["sb_ship"]!r}')
    if entry['lthr3'] and res['worst_loo'] >= GATE_LOO:
        fails.append(f'LTHR=3 leave-one-out {res["worst_loo"]:.3e} >= {GATE_LOO:g}')
    res['fails'] = fails
    res['passed'] = not fails
    return res


def report(res, meta, path):
    """Write the per-table text report."""
    L = []
    L.append(f'table      : {res["lib"]} {res["name"]}  ({res["kind"]})')
    L.append(f'model      : {meta["source"]}'
             + (f'  nphon {meta["nphon"]}' if meta.get('nphon') else ''))
    L.append(f'temperatures ({res["nT"]}): '
             + ' '.join(f'{t:g}' for t in res['temps']))
    L.append(f'same alpha/beta grid: {res["same_grid"]}   '
             f'all shipped temperatures present: {res["have_temps"]}')
    L.append(f'B(1)    new {res["b1_new"]!r}  shipped {res["b1_ship"]!r}')
    L.append(f'sigma_b new {res["sb_new"]!r}  shipped {res["sb_ship"]!r}')
    if res['beta_cut']:
        L.append(f'DOS cutoff beta {res["beta_cut"]:.4f} (lat=1 units); '
                 f'excluded beta row {res["drop_beta"]:.4f}')
    L.append('')
    L.append('MF7/MT4 S(alpha,beta) at every shipped temperature')
    cut = res['beta_cut'] is not None
    hdr = (f'{"T_ship":>8} {"T_new":>8} {"sum|dS|/sumS":>13} {"max|dS|/Smax":>13} '
           f'{"maxrel(S>1e-6)":>15} {"medrel":>9} {"n>1%":>6}')
    if cut:
        hdr += f' {"sumabs(excl)":>13} {"maxrel(excl)":>13} {"n>1%(ex)":>9}'
    L.append(hdr)
    for r in res['mt4']:
        line = (f'{r["T"]:8g} {r["T_new"]:8g} {r["sumabs"]:13.3e} '
                f'{r["maxabs"]:13.3e} {r["maxrel"]:15.3e} {r["medrel"]:9.3e} '
                f'{r["n1pct"]:6d}')
        if cut:
            line += (f' {r["sumabs_cut"]:13.3e} {r["maxrel_cut"]:13.3e} '
                     f'{r["n1pct_cut"]:9d}')
        L.append(line)
    if res['dt']:
        L.append('')
        L.append('MF7/MT4 adjacent shipped-temperature differences S(T1)-S(T0)')
        L.append(f'{"T0":>8} {"T1":>8} {"sum|d(dS)|/sum|dS|":>19} {"medrel(|dS|>1e-3)":>18}')
        for r in res['dt']:
            L.append(f'{r["T0"]:8g} {r["T1"]:8g} {r["sumabs"]:19.3e} {r["medrel"]:18.3e}')
    L.append('')
    L.append("MF7/MT2 Debye-Waller W'(T) at every shipped temperature")
    L.append(f'{"T_ship":>8} {"W_new":>12} {"W_shipped":>12} {"rel":>11}')
    for r in res['mt2']:
        L.append(f'{r["T"]:8g} {r["w_new"]:12.6g} {r["w_ship"]:12.6g} {r["rel"]:11.3e}')
    m = res.get('model')
    if m:
        L.append('')
        L.append('LEAPR tape before the post-edit (model fidelity; these rows are gated):')
        L.append(f'{"T_ship":>8} {"sum|dS|/sumS":>13} {"maxrel(S>1e-6)":>15} {"medrel":>9} '
                 f'{"n>1%":>6} {"W\' rel":>9}')
        w_by_T = {r['T']: r['rel'] for r in m['mt2']}
        for r in m['mt4']:
            L.append(f'{r["T"]:8g} {r["sumabs"]:13.3e} {r["maxrel"]:15.3e} {r["medrel"]:9.3e} '
                     f'{r["n1pct"]:6d} {w_by_T.get(r["T"], float("nan")):9.1e}')
        L.append(f'{"T0":>8} {"T1":>8} {"sum|d(dS)|/sum|dS|":>19} {"medrel(|dS|>1e-3)":>18}')
        for r in m['dt']:
            L.append(f'{r["T0"]:8g} {r["T1"]:8g} {r["sumabs"]:19.3e} {r["medrel"]:18.3e}')
        L.append(f'model worst sum|dS|/sumS {m["worst_sumabs"]:.3e}   worst adjacent-T '
                 f'difference {m["worst_dt"]:.3e}   worst W\' rel {m["worst_wrel"]:.3e}')
    inv = res.get('inversion')
    if inv:
        L.append('')
        L.append(f'phonon spectrum: one-phonon inversion at {inv["T_ref"]:g} K, '
                 f'best of {len(inv["history"]) - 1} newton steps (iterate {inv["best"]}), '
                 f'{inv["ni"]} nodes of {inv["delta"] * 1e3:.4f} meV, nphon {inv["nphon"]}, '
                 f'sum|dS|/S {inv["sumabs"]:.3e} at the reference T, scale offset '
                 f'{inv["offset"]:+.2e} (see the _inversion.txt log)')
    if res['loo']:
        L.append('')
        L.append("LTHR=3 leave-one-out Bragg rescale exp(-4 dW' E) with LEAPR W', "
                 "reference = nearest lower shipped T")
        L.append(f'{"T":>8} {"T_ref":>8} {"maxrel":>11} {"medrel":>11} '
                 f'{"S(Emax) ratio":>14}  gated')
        for r in res['loo']:
            L.append(f'{r["T"]:8g} {r["T_ref"]:8g} {r["maxrel"]:11.3e} '
                     f'{r["medrel"]:11.3e} {r["ratio_emax"]:14.6f}  '
                     + ('yes' if r['in_gate'] else 'no (no lower shipped T; '
                        'copied verbatim in production)'))
    L.append('')
    if res['kind'] == 'inverted-dos':
        L.append(f'gates (model tape): sum|dS|/sumS < {res["gate_sumabs"]:g}, adjacent-T '
                 f'difference < {GATE_DT:g}; (delivered tape): sum|dS|/sumS < {GATE_COPY:g}, '
                 f'W\' rel < {GATE_WPRIME:g}')
    else:
        L.append(f'gates: sum|dS|/sumS < {res["gate_sumabs"]:g}, W\' rel < {GATE_WPRIME:g}'
                 + (f', leave-one-out < {GATE_LOO:g}' if res['loo'] else ''))
    L.append(f'worst sum|dS|/sumS {res["worst_sumabs"]:.3e}   '
             f'worst W\' rel {res["worst_wrel"]:.3e}'
             + (f'   worst gated leave-one-out {res["worst_loo"]:.3e}'
                f' (ungated {res["worst_loo_all"]:.3e})' if res['loo'] else ''))
    for f in res['fails']:
        L.append(f'FAIL: {f}')
    L.append('PASS' if res['passed'] else 'FAIL')
    path.write_text('\n'.join(L) + '\n')


SUMMARY_HEADER = (f'{"lib":>8} {"table":>14} {"kind":>12} {"nT":>3} '
                  f'{"worst sum|dS|/S":>15} {"worst W\' rel":>13} '
                  f'{"worst LOO":>11}  result   (inverted-dos: model tape before the rescale)')


def summary_line(res):
    """One summary row; for inverted-dos tables the gated model-tape metrics are shown
    (the delivered tape reproduces the shipped blocks to GATE_COPY by construction)."""
    loo = f'{res["worst_loo"]:11.3e}' if res['loo'] else f'{"-":>11}'
    m = res.get('model')
    sumabs = m['worst_sumabs'] if m else res['worst_sumabs']
    wrel = m['worst_wrel'] if m else res['worst_wrel']
    return (f'{res["lib"]:>8} {res["name"]:>14} {res["kind"]:>12} {res["nT"]:3d} '
            f'{sumabs:15.3e} {wrel:13.3e} {loo}  '
            + ('PASS' if res['passed'] else 'FAIL'))
