"""Compare regenerated OpenMC thermal HDF5 tables against the shipped library.

For every regenerated `c_*.h5` in leapr-exact-temp/<lib>/thermal this lists the
temperature set actually stored (OpenMC names the kTs groups by the temperature
rounded to the nearest kelvin, so 293.6 K becomes 294K) and, at every
temperature the shipped table also has, compares the elastic and inelastic
cross sections on a log energy grid.  After the B(1) post-edit the inelastic
cross section should agree to ~1e-5 and the elastic to ~1e-6.

Run:  python -m tsl_leapr.h5compare --library endfb81
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import openmc.data

from .registry import BUILT_LIBRARY, LIBRARIES


def load(path):
    with h5py.File(path, 'r') as f:
        return openmc.data.ThermalScattering.from_hdf5(f[list(f.keys())[0]])


def kt_names(path):
    with h5py.File(path, 'r') as f:
        return sorted(f[list(f.keys())[0]]['kTs'].keys(),
                      key=lambda s: float(s[:-1]))


def compare_table(new_path, old_path, npoints=500):
    new, old = load(new_path), load(old_path)
    emax = min(new.energy_max, old.energy_max)
    E = np.logspace(-5, np.log10(emax * 0.999), npoints)
    rows = []
    shared = [t for t in old.temperatures if t in new.temperatures]
    for T in shared:
        for rx in ('elastic', 'inelastic'):
            a, b = getattr(new, rx), getattr(old, rx)
            if a is None or b is None:
                continue
            ya, yb = a.xs[T](E), b.xs[T](E)
            m = yb > 0
            rel = np.abs(ya[m] - yb[m]) / yb[m]
            rows.append(dict(T=T, rx=rx, maxrel=float(rel.max()),
                             medrel=float(np.median(rel)),
                             new_0253=float(a.xs[T](0.0253)),
                             old_0253=float(b.xs[T](0.0253))))
    return dict(name=new.name, emax=emax,
                temps_new=list(new.temperatures), temps_old=list(old.temperatures),
                only_new=[t for t in new.temperatures if t not in old.temperatures],
                rows=rows)


def run(lib, out_dir=None):
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir) if out_dir else root / 'leapr-exact-temp' / lib
    shipped_dir = BUILT_LIBRARY[lib] / 'thermal'
    L = [f'regenerated OpenMC HDF5 vs shipped {shipped_dir}', '']
    worst = []
    for p in sorted((out_dir / 'thermal').glob('c_*.h5')):
        if p.is_symlink():
            continue
        old = shipped_dir / p.name
        L.append(f'== {p.name}')
        L.append(f'   kTs groups (regenerated): {" ".join(kt_names(p))}')
        if not old.exists():
            L.append(f'   no shipped counterpart in {shipped_dir}')
            L.append('')
            continue
        L.append(f'   kTs groups (shipped)    : {" ".join(kt_names(old))}')
        c = compare_table(p, old)
        L.append(f'   E_max {c["emax"]:g} eV; temperatures only in the regenerated '
                 f'table: {" ".join(c["only_new"])}')
        L.append(f'   {"T":>8} {"reaction":>10} {"max rel":>11} {"median rel":>11} '
                 f'{"xs(0.0253) new":>15} {"shipped":>13}')
        for r in c['rows']:
            L.append(f'   {r["T"]:>8} {r["rx"]:>10} {r["maxrel"]:11.3e} '
                     f'{r["medrel"]:11.3e} {r["new_0253"]:15.6f} '
                     f'{r["old_0253"]:13.6f}')
            worst.append((p.name, r['rx'], r['maxrel']))
        L.append('')
    if worst:
        L.append('worst max rel over all shared temperatures:')
        for rx in ('elastic', 'inelastic'):
            w = [x for x in worst if x[1] == rx]
            if w:
                n, _, v = max(w, key=lambda x: x[2])
                L.append(f'   {rx:>10}: {v:.3e}  ({n})')
    path = out_dir / 'validation' / 'hdf5_vs_shipped.txt'
    path.write_text('\n'.join(L) + '\n')
    print('\n'.join(L[-4:]))
    print(f'wrote {path}')
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--library', choices=LIBRARIES, required=True, help='Library to compare')
    parser.add_argument('--out',     type=Path, default=None,          help='Regenerated library directory (default: leapr-exact-temp/<library>)')
    args = parser.parse_args()
    run(args.library, args.out)
