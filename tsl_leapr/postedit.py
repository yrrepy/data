"""Post-edit of a LEAPR output tape so it matches the shipped evaluation.

Three edits (build spec section 4):

1. MF7/MT4 B(1) := shipped B(1).  LEAPR writes B(1)=spr from the deck, but the
   shipped files were renormalised (VIII.0 20.43634 vs deck 20.478, JEFF-4.0
   NEA JUL24 edit).  THERMR uses B(1) as the inelastic bound cross section, so
   without this edit the HDF5 inelastic xs is uniformly 2e-3 high.
2. MF7/MT2 incoherent TAB1 sigma_b := shipped sigma_b (LTHR=2 tables).  The
   FLASSH tables ship the incoherent bound cross section, not spr*((A+1)/A)^2.
3. LTHR=3 (mixed elastic) tables: the whole MT2 section is replaced.  The
   coherent part reuses the shipped Bragg edge grid; at a shipped temperature
   the shipped cumulative column is copied verbatim, at a new temperature it is
   the Debye-Waller rescale S_cum(E,T) = cumsum(s_i(T_ref) exp(-4 dW' E_i))
   with dW' from LEAPR's own W'(T).  T_ref is the nearest shipped temperature
   BELOW T (the nearest one only if none is lower): rescaling upward damps the
   rounding of the shipped seven-digit cumulative columns (verified 5e-7),
   rescaling downward amplifies it, and a common lower reference keeps a
   finite-difference pair such as 900/905 K (both from 800 K) consistent.
   The incoherent part is LEAPR's W'(T) table with the shipped sigma_b.
4. 'inverted-dos' tables: MT4 S(alpha,beta), the MT4 effective temperature
   and the LTHR=2 MT2 W'(T) are all rescaled to the shipped evaluation, in the
   same way the Bragg columns are for LTHR=3.  At every shipped temperature the
   shipped values are written verbatim; at a new temperature the LEAPR value is
   multiplied by the shipped/LEAPR ratio interpolated linearly in T between the
   neighbouring shipped temperatures (nearest ratio outside the shipped range).
   The recovered phonon spectrum reproduces the JSI tables to 1e-4 .. 1e-3
   only, and THERMR amplifies that: for Zr the whole quasi-elastic region at
   thermal energies lies below the table's smallest alpha and is extrapolated
   from the first two beta rows there, so a 7e-4 error in those rows became a
   2 % error in the low-energy inelastic cross section (measured 2026-09-10).
   With the rescale a shipped temperature reproduces the shipped table exactly
   and a finite-difference pair such as 293.6/298.6 K carries LEAPR's own
   temperature dependence on top of the shipped level: the ratio changes by
   ~1e-3 over 100 K where the model is good, i.e. by ~5e-5 over 5 K.  Cells
   where either table is at the ENDF floor (1e-75) keep the LEAPR value.

The card-20 ZSYMAM/provenance comments are set in the deck, not here.
"""
import numpy as np

from .mf7 import (Rec, endf_float, field, line_mat, mt4_slots, read_mt2,
                  read_mt4, replace_section, section_span, set_field,
                  tape_lines, write_tape, SectionWriter)

DW_FACTOR = 4.0     # S_cum(E,T2) = sum_i s_i(T1) exp(-DW_FACTOR (W'2-W'1) E_i)
S_FLOOR = 1.0e-70   # cells at the ENDF floor (1e-75) are not rescaled
RATIO_CLIP = (1.0e-3, 1.0e3)


def _mt_lines(lines, mf, mt):
    a, b = section_span(lines, mf, mt)
    return a, b, lines[a:b]


def _shipped_b1_text(shipped):
    """The 11-column B(1) field of the shipped MF7/MT4 LIST record."""
    sec = [l for l in tape_lines(shipped)
           if len(l) >= 75 and l[70:72].strip() == '7' and l[72:75].strip() == '4']
    return field(sec[2], 0)


def _shipped_sigma_b_text(shipped):
    """The 11-column sigma_b field of the shipped MF7/MT2 incoherent TAB1 head."""
    sec = [l for l in tape_lines(shipped)
           if len(l) >= 75 and l[70:72].strip() == '7' and l[72:75].strip() == '2']
    r = Rec(sec)
    head = r.cont()
    if head[2] in (1, 3):
        c, _interp, _E, _s = r.tab1()
        for _ in range(c[2]):
            r.list()
    return field(sec[r.i], 0)


def _reference_shipped(T, shipped_temps):
    """Index of the reference column for T: the nearest shipped temperature
    below T, or the nearest one at all if no shipped temperature lies below."""
    shipped_temps = np.asarray(shipped_temps, dtype=float)
    below = np.where(shipped_temps < T)[0]
    if below.size:
        return int(below[np.argmax(shipped_temps[below])])
    return int(np.argmin(np.abs(shipped_temps - T)))


def lthr3_section(mat, tape_mt2, shipped_mt2, temps, sigma_b_text):
    """Build the LTHR=3 MF7/MT2 section lines for the regenerated temperature list."""
    coh = shipped_mt2['coh']
    E, Ssh, Tsh = coh['E'], coh['S'], coh['temps']
    W = np.asarray(tape_mt2['inc']['W'], dtype=float)
    Tt = np.asarray(temps, dtype=float)
    if len(W) != len(Tt):
        raise ValueError(f"LEAPR W'(T) has {len(W)} points, expected {len(Tt)}")

    li_vals = coh['li']
    li_default = int(li_vals[0]) if li_vals else 2

    cols, refs = [], []
    for i, T in enumerate(Tt):
        hit = np.where(np.isclose(Tsh, T, rtol=1e-6, atol=1e-6))[0]
        if hit.size:
            cols.append(np.array(Ssh[hit[0]], dtype=float))
            refs.append(int(hit[0]))
            continue
        j = _reference_shipped(T, Tsh)
        k = int(np.where(np.isclose(Tt, Tsh[j], rtol=1e-6, atol=1e-6))[0][0])
        s = np.diff(np.concatenate([[0.0], np.asarray(Ssh[j], dtype=float)]))
        cols.append(np.cumsum(s * np.exp(-DW_FACTOR * (W[i] - W[k]) * E)))
        refs.append(j)

    w = SectionWriter(mat, 7, 2)
    w.cont(shipped_mt2['za'], shipped_mt2['awr'], 3, 0, 0, 0)
    w.tab1(Tt[0], 0.0, len(Tt) - 1, 0, coh['interp'], E, cols[0])
    for i in range(1, len(Tt)):
        hit = np.where(np.isclose(Tsh, Tt[i], rtol=1e-6, atol=1e-6))[0]
        li = int(li_vals[hit[0] - 1]) if hit.size and hit[0] > 0 else li_default
        w.list(Tt[i], 0.0, li, 0, 0, cols[i])
    inc_head = len(w.lines)
    w.tab1(shipped_mt2['inc']['sb'], 0.0, 0, 0, [len(Tt), 2], Tt, W)
    w.lines[inc_head] = set_field(w.lines[inc_head], 0, sigma_b_text)
    return w.lines, refs


def rescaled_wprime(tape_mt2, shipped_mt2, temps):
    """W'(T) for the regenerated list: shipped values at shipped temperatures,
    LEAPR values times the interpolated shipped/LEAPR ratio elsewhere."""
    Tt = np.asarray(temps, dtype=float)
    Wm = np.asarray(tape_mt2['inc']['W'], dtype=float)
    if len(Wm) != len(Tt):
        raise ValueError(f"LEAPR W'(T) has {len(Wm)} points, expected {len(Tt)}")
    Tsh = np.asarray(shipped_mt2['inc']['temps'], dtype=float)
    Wsh = np.asarray(shipped_mt2['inc']['W'], dtype=float)
    idx = []
    for T in Tsh:
        hit = np.where(np.isclose(Tt, T, rtol=1e-6, atol=1e-6))[0]
        if not hit.size:
            raise ValueError(f'shipped temperature {T} K missing from the regenerated list')
        idx.append(int(hit[0]))
    ratio = Wsh / Wm[idx]
    order = np.argsort(Tsh)
    W = Wm * np.interp(Tt, Tsh[order], ratio[order])
    for j, i in enumerate(idx):
        W[i] = Wsh[j]
    return W


def _interp_ratio(T, Tsh, ratios):
    """Ratio at T: linear in T between the two neighbouring shipped temperatures,
    the nearest shipped ratio outside their range.  `ratios` is indexed like Tsh."""
    order = np.argsort(Tsh)
    Ts = np.asarray(Tsh, dtype=float)[order]
    if T <= Ts[0]:
        return ratios[order[0]]
    if T >= Ts[-1]:
        return ratios[order[-1]]
    k = int(np.searchsorted(Ts, T)) - 1
    f = (T - Ts[k]) / (Ts[k + 1] - Ts[k])
    return (1.0 - f) * ratios[order[k]] + f * ratios[order[k + 1]]


def rescale_mt4(lines, tape_path, shipped, temps):
    """Rewrite the MT4 S(alpha,beta) and T_eff values of `lines` in place (see edit 4)."""
    a4, b4 = section_span(lines, 7, 4)
    sec = lines[a4:b4]
    slots = mt4_slots(sec)
    model, ship = read_mt4(tape_path), read_mt4(shipped)
    Tt = np.asarray(temps, dtype=float)
    Tm = np.asarray(slots['temps'], dtype=float)
    if not (len(Tm) == len(Tt) and np.allclose(Tm, Tt, rtol=1e-6, atol=1e-6)):
        raise ValueError(f'MT4 temperatures {Tm} != {Tt}')
    if (model['S'].shape[1:] != ship['S'].shape[1:]
            or not np.allclose(model['alpha'], ship['alpha'], rtol=1e-5)
            or not np.allclose(model['beta'], ship['beta'], rtol=1e-5)):
        raise ValueError('MT4 alpha/beta grid differs from the shipped file')
    Tsh = np.asarray(ship['temps'], dtype=float)
    idx = [int(np.where(np.isclose(Tt, T, rtol=1e-6, atol=1e-6))[0][0]) for T in Tsh]
    ratios = []
    for k, i in enumerate(idx):
        Sm, Ss = model['S'][i], ship['S'][k]
        ok = (Sm > S_FLOOR) & (Ss > S_FLOOR)
        r = np.ones_like(Sm)
        r[ok] = np.clip(Ss[ok] / Sm[ok], *RATIO_CLIP)
        ratios.append(r)
    ratios = np.array(ratios)
    Tsh_list = list(Tsh)
    for i, T in enumerate(Tt):
        hit = np.where(np.isclose(Tsh, T, rtol=1e-6, atol=1e-6))[0]
        if hit.size:
            new = ship['S'][hit[0]]
        else:
            new = model['S'][i] * _interp_ratio(T, Tsh_list, ratios)
        for ib in range(slots['nbeta']):
            row = new[ib]
            for ia, (ln, fld) in enumerate(slots['s'][i][ib]):
                sec[ln] = set_field(sec[ln], fld, endf_float(row[ia]))
    # effective temperature for the SCT approximation
    if slots['teff'] and ship['teff'] and model['teff']:
        Ts_e, Es = ship['teff'][0]
        Tm_e, Em = model['teff'][0]
        Es, Em = np.asarray(Es, float), np.asarray(Em, float)
        rat = [Es[k] / Em[int(np.argmin(np.abs(np.asarray(Tm_e) - T)))] for k, T in enumerate(Ts_e)]
        for i, T in enumerate(Tt):
            hit = np.where(np.isclose(np.asarray(Ts_e), T, rtol=1e-6, atol=1e-6))[0]
            val = Es[hit[0]] if hit.size else Em[i] * _interp_ratio(T, list(Ts_e), rat)
            ln, fld = slots['teff'][i]
            sec[ln] = set_field(sec[ln], fld, endf_float(val))
    return lines[:a4] + sec + lines[b4:]


def lthr2_section(mat, za, awr, temps, W, sigma_b_text, interp_law=2):
    """Build an LTHR=2 (incoherent elastic) MF7/MT2 section."""
    w = SectionWriter(mat, 7, 2)
    w.cont(za, awr, 2, 0, 0, 0)
    w.tab1(0.0, 0.0, 0, 0, [len(temps), int(interp_law)], np.asarray(temps, float), W)
    w.lines[1] = set_field(w.lines[1], 0, sigma_b_text)
    return w.lines


def apply(tape_path, entry, temps, out_path):
    """Post-edit `tape_path` for a registry entry, write `out_path`, round-trip check."""
    shipped = entry['shipped']
    lines = tape_lines(tape_path)

    b1_text = _shipped_b1_text(shipped)
    a4, _b4, sec4 = _mt_lines(lines, 7, 4)
    lines[a4 + 2] = set_field(lines[a4 + 2], 0, b1_text)

    sb_text = _shipped_sigma_b_text(shipped)
    if entry['lthr3']:
        tape_mt2 = read_mt2(tape_path)
        shipped_mt2 = read_mt2(shipped)
        if shipped_mt2['lthr'] != 3:
            raise ValueError(f'{shipped}: LTHR={shipped_mt2["lthr"]}, expected 3')
        a2, _b2, sec2 = _mt_lines(lines, 7, 2)
        mat = line_mat(sec2[0])
        new, _refs = lthr3_section(mat, tape_mt2, shipped_mt2, temps, sb_text)
        lines = replace_section(lines, 7, 2, new)
    elif entry['kind'] == 'inverted-dos':
        tape_mt2 = read_mt2(tape_path)
        shipped_mt2 = read_mt2(shipped)
        if tape_mt2['lthr'] != 2 or shipped_mt2['lthr'] != 2:
            raise ValueError(f'{shipped}: LTHR {tape_mt2["lthr"]}/{shipped_mt2["lthr"]}, '
                             f'expected 2 for an inverted-dos table')
        a2, _b2, sec2 = _mt_lines(lines, 7, 2)
        mat = line_mat(sec2[0])
        W = rescaled_wprime(tape_mt2, shipped_mt2, temps)
        law = shipped_mt2['inc']['interp'][1] if len(shipped_mt2['inc']['interp']) > 1 else 2
        new = lthr2_section(mat, tape_mt2['za'], tape_mt2['awr'], temps, W, sb_text, law)
        lines = replace_section(lines, 7, 2, new)
        lines = rescale_mt4(lines, tape_path, shipped, temps)
    else:
        a2, _b2, _sec2 = _mt_lines(lines, 7, 2)
        lines[a2 + 1] = set_field(lines[a2 + 1], 0, sb_text)

    write_tape(out_path, lines)
    _roundtrip(out_path, entry, temps, b1_text, sb_text)
    return out_path


def _roundtrip(path, entry, temps, b1_text, sb_text):
    d4 = read_mt4(path)
    if _shipped_b1_text(path) != b1_text:
        raise AssertionError(f'{path}: B(1) field not written ({_shipped_b1_text(path)!r})')
    if _shipped_sigma_b_text(path) != sb_text:
        raise AssertionError(f'{path}: sigma_b field not written')
    got = np.asarray(d4['temps'], dtype=float)
    want = np.asarray(temps, dtype=float)
    if got.size != want.size or not np.allclose(got, want, rtol=1e-6, atol=1e-6):
        raise AssertionError(f'{path}: MT4 temperatures {got} != {want}')

    e = read_mt2(path)
    if entry['lthr3']:
        if e['lthr'] != 3:
            raise AssertionError(f'{path}: LTHR={e["lthr"]} after rebuild')
        s = read_mt2(entry['shipped'])
        Wt = np.asarray(e['inc']['temps'], dtype=float)
        if not np.allclose(Wt, want, rtol=1e-6, atol=1e-6):
            raise AssertionError(f'{path}: MT2 incoherent temperatures {Wt} != {want}')
        for j, T in enumerate(s['coh']['temps']):
            i = int(np.argmin(np.abs(np.asarray(e['coh']['temps']) - T)))
            a = np.asarray(e['coh']['S'][i], dtype=float)
            b = np.asarray(s['coh']['S'][j], dtype=float)
            m = b > 0
            rel = np.abs(a[m] - b[m]) / b[m]
            if rel.size and rel.max() > 1e-6:
                raise AssertionError(
                    f'{path}: coherent column at {T} K differs from shipped by '
                    f'{rel.max():.3e} (rel)')
    else:
        Wt = np.asarray(e['inc']['temps'], dtype=float)
        if not np.allclose(Wt, want, rtol=1e-6, atol=1e-6):
            raise AssertionError(f'{path}: MT2 temperatures {Wt} != {want}')
        if entry['kind'] == 'inverted-dos':
            s = read_mt2(entry['shipped'])
            Wn = np.asarray(e['inc']['W'], dtype=float)
            for T, W in zip(np.asarray(s['inc']['temps'], float),
                            np.asarray(s['inc']['W'], float)):
                i = int(np.argmin(np.abs(Wt - T)))
                if abs(Wn[i] - W) > 1e-6 * W:
                    raise AssertionError(
                        f"{path}: W'({T} K) {Wn[i]} not rescaled to shipped {W}")
            s4 = read_mt4(entry['shipped'])
            for k, T in enumerate(np.asarray(s4['temps'], float)):
                i = int(np.argmin(np.abs(got - T)))
                a, b = d4['S'][i], s4['S'][k]
                m = b > S_FLOOR
                rel = np.abs(a[m] - b[m]) / b[m]
                if rel.size and rel.max() > 1e-6:
                    raise AssertionError(
                        f'{path}: MT4 block at {T} K differs from shipped by '
                        f'{rel.max():.3e} (rel) after the rescale')
