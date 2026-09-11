"""Registry of TSL tables that can be regenerated from their evaluation model.

One entry per (library, OpenMC table name).  Every path is resolved against the
openmc-data checkout that contains this package and is checked when an entry is
looked up (not at import time: generate_*.py import this package before the
shipped files have been downloaded).

Fields
------
lib        library id used by generate_tsl_leapr.py (--library)
name       OpenMC thermal table name (drives the HDF5 file name)
kind       'deck'         -- an existing LEAPR deck is the model input
           'flassh-dos'   -- a FLASSH Control + DOS pair is the model input
           'inverted-dos' -- no inputs distributed: the phonon spectrum is
                             recovered from the shipped S(alpha,beta) by
                             invert.solve (approximate, ~1e-3; own gates)
invert     (kind 'inverted-dos') overrides of invert.DEFAULTS: T_ref, nphon
           (the evaluator's phonon order, part of the model), max_iter
deck       LEAPR deck (kind 'deck')
control    FLASSH Control file (kind 'flassh-dos')
dos        FLASSH DOS file (kind 'flassh-dos')
shipped    shipped TSL ENDF file: temperature grid, B(1), sigma_b and, for the
           LTHR=3 tables, the Bragg edges are all taken from it at run time
neutron    neutron sublibrary ENDF file needed by process_thermal
lthr3      True if MT2 must be rebuilt as an LTHR=3 (mixed elastic) section
sab        (neutron basename, thermal basename) as they appear in the sab_files
           list of the matching generate_*.py, used by the --tsl-temperatures hook
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_V80 = 'endfb-viii.0-endf'
_V81 = 'endfb-viii.1-endf'
_JEFF = 'jeff-4.0-endf'
_JENDL = 'jendl-5-endf'

# release string used by the generate_*.py scripts -> library id used here
RELEASE_TO_LIB = {
    ('endf', 'viii.0'): 'endfb80',
    ('endf', 'viii.1'): 'endfb81',
    ('jeff', '4.0'):    'jeff40',
    ('jendl', '5'):     'jendl5',
}

_ENTRIES = [
    dict(lib='endfb80', name='c_H_in_ZrH', kind='deck', lthr3=False,
         deck=f'{_V80}/thermal/tsl-HinZrH.leapr',
         shipped=f'{_V80}/thermal/tsl-HinZrH.endf',
         neutron=f'{_V80}/neutron/n-001_H_001.endf',
         sab=('n-001_H_001.endf', 'tsl-HinZrH.endf')),
    dict(lib='endfb80', name='c_Zr_in_ZrH', kind='deck', lthr3=False,
         deck=f'{_V80}/thermal/tsl-ZrinZrH.leapr',
         shipped=f'{_V80}/thermal/tsl-ZrinZrH.endf',
         neutron=f'{_V80}/neutron/n-040_Zr_090.endf',
         sab=('n-040_Zr_090.endf', 'tsl-ZrinZrH.endf')),

    dict(lib='endfb81', name='c_H_in_ZrH', kind='deck', lthr3=False,
         deck=f'{_V81}/thermal/tsl-HinZrH.leapr',
         shipped=f'{_V81}/thermal/tsl-HinZrH.endf',
         neutron=f'{_V81}/neutron/n-001_H_001.endf',
         sab=('n-001_H_001.endf', 'tsl-HinZrH.endf')),
    dict(lib='endfb81', name='c_Zr_in_ZrH', kind='deck', lthr3=False,
         deck=f'{_V81}/thermal/tsl-ZrinZrH.leapr',
         shipped=f'{_V81}/thermal/tsl-ZrinZrH.endf',
         neutron=f'{_V81}/neutron/n-040_Zr_090.endf',
         sab=('n-040_Zr_090.endf', 'tsl-ZrinZrH.endf')),
    dict(lib='endfb81', name='c_H_in_ZrH2', kind='flassh-dos', lthr3=False,
         control=f'{_V81}/thermal/tsl-HinZrH2_flassh-Control.txt',
         dos=f'{_V81}/thermal/tsl-HinZrH2_flassh-DOS.txt',
         shipped=f'{_V81}/thermal/tsl-HinZrH2.endf',
         neutron=f'{_V81}/neutron/n-001_H_001.endf',
         sab=('n-001_H_001.endf', 'tsl-HinZrH2.endf')),
    dict(lib='endfb81', name='c_H_in_ZrHx', kind='flassh-dos', lthr3=False,
         control=f'{_V81}/thermal/tsl-HinZrHx_flassh-Control.txt',
         dos=f'{_V81}/thermal/tsl-HinZrHx_flassh-DOS.txt',
         shipped=f'{_V81}/thermal/tsl-HinZrHx.endf',
         neutron=f'{_V81}/neutron/n-001_H_001.endf',
         sab=('n-001_H_001.endf', 'tsl-HinZrHx.endf')),
    dict(lib='endfb81', name='c_Zr_in_ZrH2', kind='flassh-dos', lthr3=True,
         control=f'{_V81}/thermal/tsl-ZrinZrH2_flassh-Control.txt',
         dos=f'{_V81}/thermal/tsl-ZrinZrH2_flassh-DOS.txt',
         shipped=f'{_V81}/thermal/tsl-ZrinZrH2.endf',
         neutron=f'{_V81}/neutron/n-040_Zr_090.endf',
         sab=('n-040_Zr_090.endf', 'tsl-ZrinZrH2.endf')),
    dict(lib='endfb81', name='c_Zr_in_ZrHx', kind='flassh-dos', lthr3=True,
         control=f'{_V81}/thermal/tsl-ZrinZrHx_flassh-Control.txt',
         dos=f'{_V81}/thermal/tsl-ZrinZrHx_flassh-DOS.txt',
         shipped=f'{_V81}/thermal/tsl-ZrinZrHx.endf',
         neutron=f'{_V81}/neutron/n-040_Zr_090.endf',
         sab=('n-040_Zr_090.endf', 'tsl-ZrinZrHx.endf')),

    dict(lib='jeff40', name='c_H_in_ZrH', kind='deck', lthr3=False,
         deck='tsl_leapr/decks/ike_HinZrH.leapr',
         shipped=f'{_JEFF}/thermal/tsl_H_ZrH.jeff',
         neutron=f'{_JEFF}/neutron/n_1-H-001g.jeff',
         sab=('n_1-H-001g.jeff', 'tsl_H_ZrH.jeff')),
    # JSI/NCSU/CEA 2023 delta (ZrH15 -> OpenMC ZrHx) and epsilon (ZrH2) tables:
    # DFT+Phonopy spectra through NJOY LEAPR at the default nphon=100 (found by
    # scan), no deck or DOS distributed -> spectrum inverted from the file.
    dict(lib='jeff40', name='c_H_in_ZrH2', kind='inverted-dos', lthr3=False,
         invert=dict(T_ref=293.6, nphon=100, max_iter=15),
         shipped=f'{_JEFF}/thermal/tsl_H_ZrH2.jeff',
         neutron=f'{_JEFF}/neutron/n_1-H-001g.jeff',
         sab=('n_1-H-001g.jeff', 'tsl_H_ZrH2.jeff')),
    dict(lib='jeff40', name='c_H_in_ZrHx', kind='inverted-dos', lthr3=False,
         invert=dict(T_ref=293.6, nphon=100, max_iter=15),
         shipped=f'{_JEFF}/thermal/tsl_H_ZrH15.jeff',
         neutron=f'{_JEFF}/neutron/n_1-H-001g.jeff',
         sab=('n_1-H-001g.jeff', 'tsl_H_ZrH15.jeff')),
    dict(lib='jeff40', name='c_Zr_in_ZrH2', kind='inverted-dos', lthr3=False,
         invert=dict(T_ref=293.6, nphon=100, max_iter=15),
         shipped=f'{_JEFF}/thermal/tsl_Zr_ZrH2.jeff',
         neutron=f'{_JEFF}/neutron/n_40-Zr-090g.jeff',
         sab=('n_40-Zr-090g.jeff', 'tsl_Zr_ZrH2.jeff')),
    dict(lib='jeff40', name='c_Zr_in_ZrHx', kind='inverted-dos', lthr3=False,
         invert=dict(T_ref=293.6, nphon=100, max_iter=15),
         shipped=f'{_JEFF}/thermal/tsl_Zr_ZrH15.jeff',
         neutron=f'{_JEFF}/neutron/n_40-Zr-090g.jeff',
         sab=('n_40-Zr-090g.jeff', 'tsl_Zr_ZrH15.jeff')),

    dict(lib='jendl5', name='c_H_in_ZrH', kind='deck', lthr3=False,
         deck=f'{_V80}/thermal/tsl-HinZrH.leapr',
         shipped=f'{_JENDL}/thermal/jendl5-tsl/tsl_HinZrH.dat',
         neutron=f'{_JENDL}/neutron/jendl5-n/n_001-H-001.dat',
         sab=('n_001-H-001.dat', 'tsl_HinZrH.dat')),
    dict(lib='jendl5', name='c_Zr_in_ZrH', kind='deck', lthr3=False,
         deck=f'{_V80}/thermal/tsl-ZrinZrH.leapr',
         shipped=f'{_JENDL}/thermal/jendl5-tsl/tsl_ZrinZrH.dat',
         neutron=f'{_JENDL}/neutron/jendl5-n/n_040-Zr-090.dat',
         sab=('n_040-Zr-090.dat', 'tsl_ZrinZrH.dat')),
]

# Built HDF5 libraries the untouched tables are symlinked from (siblings of the
# openmc-data checkout: Models/NukeData/<lib>/{neutron,thermal,cross_sections.xml}).
BUILT_LIBRARY = {
    'endfb80': ROOT.parent / 'endfb8.0',
    'endfb81': ROOT.parent / 'endfb8.1',
    'jeff40':  ROOT.parent / 'jeff40',
    'jendl5':  ROOT.parent / 'jendl5',
}

LIBRARIES = ['endfb80', 'endfb81', 'jeff40', 'jendl5']

_PATH_KEYS = ('deck', 'control', 'dos', 'shipped', 'neutron')


def _resolve(entry):
    out = dict(entry)
    for key in _PATH_KEYS:
        if key in entry:
            p = ROOT / entry[key]
            if not p.exists():
                raise FileNotFoundError(
                    f'tsl_leapr registry: {entry["lib"]}/{entry["name"]} '
                    f'{key} missing: {p}')
            out[key] = p
    return out


def by_library(lib):
    """Resolved entries of one library (paths checked here)."""
    if lib not in LIBRARIES:
        raise KeyError(f'unknown library {lib!r}, expected one of {LIBRARIES}')
    return [_resolve(e) for e in _ENTRIES if e['lib'] == lib]


def lookup(lib, name):
    for e in _ENTRIES:
        if e['lib'] == lib and e['name'] == name:
            return _resolve(e)
    raise KeyError(f'no registry entry for {lib}/{name}')


def lookup_sab(lib, thermal_basename):
    """Registry entry for a (library, sab_files thermal basename) pair, or None."""
    for e in _ENTRIES:
        if e['lib'] == lib and e['sab'][1] == thermal_basename:
            return _resolve(e)
    return None
