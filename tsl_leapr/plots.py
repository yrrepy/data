#!/usr/bin/env python
"""Plots of the exact-temperature TSL tables against OpenMC's temperature treatment.

For every regenerated thermal table in ``<root>/<lib>/thermal`` the script
compares three cross sections at the temperature the model was regenerated for:

exact           the regenerated LEAPR table evaluated at that temperature,
nearest         the shipped library table at its temperature closest to it
                (OpenMC ``temperature_method='nearest'``),
interpolation   the linear blend of the two bracketing shipped tables
                (OpenMC ``temperature_method='interpolation'``).

Two figures per table are written to ``<root>/<lib>/plots``: ``<name>_xs.png``
(cross sections and the ratio to exact at 298.6 K and 905 K) and
``<name>_mtc.png`` (the temperature-difference signal that drives the moderator
temperature coefficient, 293.6 -> 298.6 K and 900 -> 905 K).  A compact numeric
summary is written to ``<root>/<lib>/plots/summary.txt`` and printed.

Only the integral cross sections are needed, so each table is read once and the
secondary angle-energy distributions are dropped immediately (they are most of
the ~0.6 GB of an exact-temperature file).
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator
from openmc.data import K_BOLTZMANN, ThermalScattering

OPENMC_DATA = Path(__file__).resolve().parent.parent

# library id -> directory name of the shipped HDF5 library
SHIPPED_DIRNAME = {'endfb80': 'endfb8.0',
                   'endfb81': 'endfb8.1',
                   'jeff40':  'jeff40',
                   'jendl5':  'jendl5'}
LIBRARIES = list(SHIPPED_DIRNAME)

TEMPERATURES = (298.6, 905.0)                        # figure 1 columns
MTC_PAIRS = ((293.6, 298.6), (900.0, 905.0))         # figure 2 columns
COMPONENTS = ('total', 'elastic', 'inelastic')       # figure rows
E_THERMAL = 0.0253
E_MIN = 1.0e-5
N_GRID = 400

COLOR = {'exact': '#2a78d6', 'nearest': '#eb6834', 'interpolation': '#1baf7a'}
STYLE = {'exact': dict(ls='-', lw=1.6), 'nearest': dict(ls='--', lw=1.4),
         'interpolation': dict(ls=':', lw=1.8)}
FG, FG2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'


def tK(T):
    """Temperature as it is meant to read: the kT round-trip leaves 800.004 K."""
    r = round(float(T), 1)
    return f'{round(r):g}' if abs(r - round(r)) < 0.05 else f'{r:g}'


# ---------------------------------------------------------------- data access

class Table:
    """Integral thermal cross sections of one HDF5 table, keyed by temperature."""

    def __init__(self, path):
        self.path = Path(path)
        data = ThermalScattering.from_hdf5(str(path))
        self.name = data.name
        temps = np.array([kT / K_BOLTZMANN for kT in data.kTs])
        order = np.argsort(temps)
        self.temperatures = temps[order]
        keys = [data.temperatures[i] for i in order]
        elastic = getattr(data, 'elastic', None)
        self.elastic = {}
        self.inelastic = {}
        for T, key in zip(self.temperatures, keys):
            self.inelastic[float(T)] = data.inelastic.xs[key]
            if elastic is not None:
                self.elastic[float(T)] = elastic.xs[key]
        first = float(self.temperatures[0])
        self.elastic_type = (type(self.elastic[first]).__name__ if self.elastic
                             else 'none')
        if self.elastic_type == 'Sum':
            parts = [type(f).__name__ for f in self.elastic[first].functions]
            self.elastic_type = 'Sum(' + ' + '.join(parts) + ')'
        self.energy_max = float(self.inelastic[first].x[-1])
        del data          # drop the secondary distributions

    def _key(self, T):
        """The table temperature that stands for `T`; fails if none is close."""
        Tk = float(self.temperatures[np.argmin(np.abs(self.temperatures - T))])
        if abs(Tk - T) > 1.0:
            raise KeyError(f'{self.path.name}: no temperature near {T} K '
                           f'(closest is {Tk} K)')
        return Tk

    def nearest(self, T):
        """Table temperature closest to `T` (OpenMC method='nearest')."""
        return float(self.temperatures[np.argmin(np.abs(self.temperatures - T))])

    def bracket(self, T):
        """(T_lo, T_hi, w_lo, w_hi) of OpenMC method='interpolation' at `T`."""
        i = int(np.searchsorted(self.temperatures, T))
        i = min(max(i, 1), len(self.temperatures) - 1)
        T_lo = float(self.temperatures[i - 1])
        T_hi = float(self.temperatures[i])
        w_lo = (T_hi - T) / (T_hi - T_lo)
        return T_lo, T_hi, w_lo, 1.0 - w_lo

    def xs(self, component, T, E):
        """`component` cross section in barns at table temperature `T`."""
        Tk = self._key(T)
        E = np.asarray(E, dtype=float)
        inelastic = np.asarray(self.inelastic[Tk](E), dtype=float)
        elastic = (np.asarray(self.elastic[Tk](E), dtype=float) if self.elastic
                   else np.zeros_like(E))
        return {'total': elastic + inelastic,
                'elastic': elastic,
                'inelastic': inelastic}[component]


def neighbours(shipped, T):
    """Nearest / bracketing shipped temperatures and interpolation weights."""
    T_lo, T_hi, w_lo, w_hi = shipped.bracket(T)
    tie = abs(abs(T - T_lo) - abs(T_hi - T))
    return dict(T=T, near=shipped.nearest(T), T_lo=T_lo, T_hi=T_hi,
                w_lo=w_lo, w_hi=w_hi, tie=tie)


def treatments(exact, shipped, nb, component, T, E):
    """The three cross section curves at temperature `T` on grid `E`."""
    return {'exact': exact.xs(component, T, E),
            'nearest': shipped.xs(component, nb['near'], E),
            'interpolation': (nb['w_lo'] * shipped.xs(component, nb['T_lo'], E)
                              + nb['w_hi'] * shipped.xs(component, nb['T_hi'], E))}


def ratio(curve, exact):
    """curve / exact, NaN where the exact cross section vanishes."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(exact > 0.0, curve / exact, np.nan)


# -------------------------------------------------------------------- styling

def style(ax, bottom=True):
    ax.grid(True, which='both', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which='both', colors=FG2, labelsize=7.5, labelcolor=FG2)
    if not bottom:
        ax.tick_params(labelbottom=False)


def cell_title(ax, text):
    ax.set_title(text, color=FG, fontsize=9, pad=3)


def label(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel, color=FG2, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=FG2, fontsize=8.5)


def open_space_for_legend(ax, log):
    """Raise the top of `ax` until its legend clears the curves underneath."""
    ax.figure.canvas.draw()
    box = ax.get_legend().get_window_extent().transformed(ax.transAxes.inverted())
    room = box.y0 - 0.03
    if room <= 0.05:
        return
    lo, hi = ax.get_ylim()
    top = max((np.nanmax(line.get_ydata()) for line in ax.get_lines()),
              default=hi)
    if log:
        lo = max(lo, 1e-300)
        top = max(top, lo * 1.0001)
        span = (np.log10(top) - np.log10(lo)) / room
        ax.set_ylim(lo, 10.0 ** (np.log10(lo) + span))
    else:
        top = max(top, lo + 1e-300)
        ax.set_ylim(lo, lo + (top - lo) / room)


def figure_legend(ax, handles, labels):
    leg = ax.legend(handles, labels, loc='upper right', fontsize=7,
                    framealpha=0.93, edgecolor=GRID, facecolor='white',
                    borderpad=0.5, labelspacing=0.35, handlelength=2.6)
    leg.get_frame().set_linewidth(0.6)
    for text in leg.get_texts():
        text.set_color(FG)


# -------------------------------------------------------------------- figures

def figure_xs(lib, exact, shipped, info, path, dpi):
    """Cross sections and ratio-to-exact at both temperatures."""
    E = np.logspace(np.log10(E_MIN), np.log10(exact.energy_max), N_GRID)
    fig = plt.figure(figsize=(11, 10), layout='constrained', facecolor='white')
    fig.get_layout_engine().set(hspace=0.03, wspace=0.03)
    outer = fig.add_gridspec(3, 2)
    handles, labels, first = [], [], None

    for row, component in enumerate(COMPONENTS):
        for col, T in enumerate(TEMPERATURES):
            nb = info[T]
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1],
                                                hspace=0.0)
            ax = fig.add_subplot(inner[0])
            rax = fig.add_subplot(inner[1], sharex=ax)
            curves = treatments(exact, shipped, nb, component, T, E)

            for key in ('exact', 'nearest', 'interpolation'):
                line, = ax.loglog(E, curves[key], color=COLOR[key], **STYLE[key])
                if row == 0:
                    handles.append(line)
                    labels.append(curve_label(key, nb))

            rax.axhline(1.0, color=COLOR['exact'], ls=(0, (1, 2.5)), lw=1.1)
            spread = [0.005]
            for key in ('nearest', 'interpolation'):
                r = ratio(curves[key], curves['exact'])
                rax.semilogx(E, r, color=COLOR[key], **STYLE[key])
                if np.isfinite(r).any():
                    spread.append(np.nanmax(np.abs(r - 1.0)))
            m = max(spread)
            rax.set_ylim(1.0 - 1.10 * m, 1.0 + 1.10 * m)
            digits = max(2, int(np.ceil(-np.log10(m))) + 2)
            rax.yaxis.set_major_locator(FixedLocator([1.0 - m, 1.0, 1.0 + m]))
            rax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _p, d=digits: f'{v:.{d}f}'))
            rax.set_xlim(E[0], E[-1])

            cell_title(ax, f'{component} — {tK(T)} K')
            style(ax, bottom=False)
            style(rax)
            label(ax, ylabel='sigma (b)')
            label(rax, xlabel='E (eV)' if row == 2 else None,
                  ylabel='ratio to exact')
            if first is None:
                first = ax

    figure_legend(first, handles, labels)
    open_space_for_legend(first, log=True)
    fig.suptitle(f'{lib} {exact.name}: thermal S(a,b) cross sections '
                 f'at 298.6 K and 905 K', color=FG, fontsize=12)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


def curve_label(key, nb):
    if key == 'exact':
        return f'LEAPR exact {tK(nb["T"])} K'
    if key == 'nearest':
        return f'nearest ({tK(nb["near"])} K table)'
    return (f'interpolation ({nb["w_lo"]:.3f} x {tK(nb["T_lo"])} K '
            f'+ {nb["w_hi"]:.3f} x {tK(nb["T_hi"])} K)')


def mtc_label(key, lo, hi):
    if key == 'exact':
        return f'LEAPR exact {tK(lo["T"])} -> {tK(hi["T"])} K'
    if key == 'nearest':
        if lo['near'] == hi['near']:
            return f'nearest (both {tK(lo["near"])} K, no signal)'
        return f'nearest ({tK(lo["near"])} K -> {tK(hi["near"])} K tables)'
    brackets = [f'{tK(nb["T_lo"])}/{tK(nb["T_hi"])} K' for nb in (lo, hi)]
    if brackets[0] == brackets[1]:
        return f'interpolation ({brackets[0]} chord)'
    return f'interpolation ({brackets[0]} -> {brackets[1]} chord)'


def mtc_curves(exact, shipped, info, component, T_lo, T_hi, E):
    """Percent change of each treatment between T_lo and T_hi."""
    base = exact.xs(component, T_lo, E)
    lo = treatments(exact, shipped, info[T_lo], component, T_lo, E)
    hi = treatments(exact, shipped, info[T_hi], component, T_hi, E)
    out = {}
    for key in ('exact', 'nearest', 'interpolation'):
        with np.errstate(divide='ignore', invalid='ignore'):
            out[key] = np.where(base > 0.0, 100.0 * (hi[key] - lo[key]) / base,
                                np.nan)
    return out


def figure_mtc(lib, exact, shipped, info, path, dpi):
    """Temperature sensitivity of the cross sections: the MTC signal."""
    E = np.logspace(np.log10(E_MIN), np.log10(exact.energy_max), N_GRID)
    fig = plt.figure(figsize=(11, 10), layout='constrained', facecolor='white')
    fig.get_layout_engine().set(hspace=0.05, wspace=0.05)
    axes = fig.subplots(3, 2)
    handles, labels = [], []

    for row, component in enumerate(COMPONENTS):
        for col, (T_lo, T_hi) in enumerate(MTC_PAIRS):
            ax = axes[row][col]
            curves = mtc_curves(exact, shipped, info, component, T_lo, T_hi, E)
            ax.axhline(0.0, color=FG2, lw=0.8)
            for key in ('exact', 'nearest', 'interpolation'):
                line, = ax.semilogx(E, curves[key], color=COLOR[key],
                                    **STYLE[key])
                if row == 0:
                    handles.append(line)
                    labels.append(mtc_label(key, info[T_lo], info[T_hi]))
            ax.set_xlim(E[0], E[-1])
            cell_title(ax, f'{component} — {tK(T_lo)} -> {tK(T_hi)} K')
            style(ax)
            label(ax, xlabel='E (eV)' if row == 2 else None,
                  ylabel='change in sigma (%)')

    figure_legend(axes[0][0], handles, labels)
    open_space_for_legend(axes[0][0], log=False)
    fig.suptitle(f'{lib} {exact.name}: temperature sensitivity of the thermal '
                 f'cross sections, exact vs OpenMC temperature treatment',
                 color=FG, fontsize=12)
    fig.savefig(path, dpi=dpi, facecolor='white')
    plt.close(fig)


# -------------------------------------------------------------------- summary

HEADER = ('  {:<10s} {:>7s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} '
          '{:>11s}')
ROW = ('  {:<10s} {:>7.1f} {:>11.3f} {:>11.3f} {:>11.3f} {:>11.3f} '
       '{:>11.4f} {:>11.4f} {:>11.4f}')


def summarize(lib, exact, shipped, info):
    """Numeric block for one table, as a list of lines."""
    E = np.logspace(np.log10(E_MIN), np.log10(exact.energy_max), N_GRID)
    Et = np.array([E_THERMAL])
    out = [f'== {lib} {exact.name} ==',
           f'  regenerated : {exact.path}',
           f'  shipped     : {shipped.path}',
           '  exact table T   : ' + ' '.join(tK(t) for t in exact.temperatures),
           '  shipped table T : ' + ' '.join(tK(t) for t in shipped.temperatures),
           f'  elastic form    : {exact.elastic_type}']
    for T in TEMPERATURES:
        nb = info[T]
        out.append(f'  {tK(T)} K: nearest {tK(nb["near"])} K | interpolation '
                   f'{nb["w_lo"]:.4f} x {tK(nb["T_lo"])} K + '
                   f'{nb["w_hi"]:.4f} x {tK(nb["T_hi"])} K')

    for T in sorted(info):
        nb = info[T]
        if nb['tie'] < 1.0:
            out.append(f'  NOTE: at {tK(T)} K the two candidate nearest tables '
                       f'({tK(nb["T_lo"])} K, {tK(nb["T_hi"])} K) are equidistant '
                       f'to within {nb["tie"]:.2g} K; OpenMC keeps the first '
                       f'minimum, i.e. {tK(nb["near"])} K')
    out += ['', '  |ratio-1| in %, grid = 400 log-uniform points '
            f'{E_MIN:g}-{exact.energy_max:g} eV; MTC = 100*(sigma(T_hi)'
            '-sigma(T_lo))/sigma_exact(T_lo) at 0.0253 eV',
            '  MTC pair: the 298.6 K row uses 293.6 -> 298.6 K, '
            'the 905 K row uses 900 -> 905 K',
            HEADER.format('component', 'T [K]', '<|r-1|>near', '<|r-1|>intp',
                          '|r-1|near', '|r-1|intp', 'MTC exact', 'MTC near',
                          'MTC intp'),
            HEADER.format('', '', '[%]', '[%]', '@0.0253[%]', '@0.0253[%]',
                          '[%]', '[%]', '[%]')]

    for component in COMPONENTS:
        for T, (T_lo, T_hi) in zip(TEMPERATURES, MTC_PAIRS):
            grid = treatments(exact, shipped, info[T], component, T, E)
            point = treatments(exact, shipped, info[T], component, T, Et)
            avg, at = [], []
            for key in ('nearest', 'interpolation'):
                avg.append(100.0 * np.nanmean(
                    np.abs(ratio(grid[key], grid['exact']) - 1.0)))
                at.append(100.0 * np.abs(
                    ratio(point[key], point['exact'])[0] - 1.0))
            mtc = mtc_curves(exact, shipped, info, component, T_lo, T_hi, Et)
            out.append(ROW.format(component, T, avg[0], avg[1], at[0], at[1],
                                  mtc['exact'][0], mtc['nearest'][0],
                                  mtc['interpolation'][0]))
    out.append('')
    return out


# ------------------------------------------------------------------------ CLI

def regenerated_tables(thermal_dir):
    """Names of the regenerated tables: real files, not symlinks to shipped."""
    return sorted(p.stem for p in thermal_dir.glob('c_*.h5')
                  if p.is_file() and not p.is_symlink())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--library',  choices=LIBRARIES, required=True,       help='library id of the regenerated tables')
    parser.add_argument('--root',     type=Path, default=Path('leapr-exact-temp'), help='exact-temperature library root, relative to openmc-data')
    parser.add_argument('--shipped',  type=Path, default=Path('/home/perry/Desktop/Radiant_Interview/Models/NukeData'), help='directory holding the shipped HDF5 libraries')
    parser.add_argument('--tables',   nargs='*', default=None,                help='table names to plot (default: every regenerated table)')
    parser.add_argument('--dpi',      type=int, default=150,                  help='PNG resolution')
    args = parser.parse_args()

    lib = args.library
    root = args.root if args.root.is_absolute() else OPENMC_DATA / args.root
    thermal = root / lib / 'thermal'
    shipped_dir = args.shipped / SHIPPED_DIRNAME.get(lib, lib) / 'thermal'
    plots = root / lib / 'plots'
    plots.mkdir(parents=True, exist_ok=True)

    names = args.tables or regenerated_tables(thermal)
    names = [n if n.startswith('c_') else 'c_' + n for n in names]
    names = [n[:-3] if n.endswith('.h5') else n for n in names]
    if not names:
        raise SystemExit(f'no regenerated tables in {thermal}')

    lines, written = [], []
    for name in names:
        t0 = time.time()
        exact = Table(thermal / f'{name}.h5')
        shipped = Table(shipped_dir / f'{name}.h5')
        info = {T: neighbours(shipped, T) for T in TEMPERATURES}
        for T_lo, T_hi in MTC_PAIRS:
            info.setdefault(T_lo, neighbours(shipped, T_lo))
            info.setdefault(T_hi, neighbours(shipped, T_hi))

        xs_png = plots / f'{name}_xs.png'
        mtc_png = plots / f'{name}_mtc.png'
        figure_xs(lib, exact, shipped, info, xs_png, args.dpi)
        figure_mtc(lib, exact, shipped, info, mtc_png, args.dpi)
        written += [xs_png, mtc_png]
        lines += summarize(lib, exact, shipped, info)
        print(f'[{name}] {time.time() - t0:.1f} s -> {xs_png.name}, '
              f'{mtc_png.name}', flush=True)
        del exact, shipped

    text = '\n'.join(lines)
    (plots / 'summary.txt').write_text(text + '\n')
    print(text)
    print('written:')
    for p in written + [plots / 'summary.txt']:
        print(f'  {p}')


if __name__ == '__main__':
    main()
