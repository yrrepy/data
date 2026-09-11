#!/usr/bin/env python3
"""
Regenerate ENDF MF7 thermal scattering tables at exact temperatures with NJOY LEAPR
and convert them to OpenMC HDF5.

A bound S(alpha,beta) law cannot be broadened: temperature enters through the phonon
occupation numbers and the Debye-Waller exponent, both of which need the phonon
spectrum, which is not in the ENDF file.  The only correct way to get a table at a new
temperature is to re-run the evaluation model.  This script does that for the tables
listed in tsl_leapr/registry.py, over the union of the shipped temperature grid and the
requested temperatures, validates the result against the shipped evaluation at every
shipped temperature, and writes a parallel library directory (the shipped one is left
untouched):

    leapr-exact-temp/<library>/{decks,endf,thermal,validation,plots}/ + cross_sections.xml
"""
import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import tsl_leapr
from tsl_leapr import registry, validate
from utils import process_thermal


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=CustomFormatter)
parser.add_argument('--library',       choices=registry.LIBRARIES, required=True, help='Library to regenerate tables for')
parser.add_argument('--tables',        nargs='+',   default=None,                 help='OpenMC table names (default: every registry entry of the library)')
parser.add_argument('--temperatures',  type=float,  nargs='+', default=tsl_leapr.DEFAULT_TEMPERATURES, help='Extra temperatures in Kelvin, added to the shipped grid')
parser.add_argument('--njoy',          type=str,    default='njoy',               help='NJOY2016 executable')
parser.add_argument('--out',           type=Path,   default=None,                 help='Output directory (default: leapr-exact-temp/<library>)')
parser.add_argument('--nphon',         type=int,    default=300,                  help='Phonon expansion order, flassh-dos tables only (deck tables keep their own, inverted-dos tables the order in their registry entry)')
parser.add_argument('--validate-only', action='store_true',                       help='Run LEAPR, post-edit and validation only: no HDF5, no cross_sections.xml')
parser.add_argument('--jobs',          type=int,    default=4,                    help='Parallel tables')
parser.add_argument('--libver',        choices=['earliest', 'latest'], default='earliest', help="Output HDF5 versioning. Use 'earliest' for backwards compatibility or 'latest' for performance")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parent
out_dir = args.out if args.out is not None else ROOT / 'leapr-exact-temp' / args.library
out_dir = Path(out_dir).resolve()

entries = registry.by_library(args.library)
if args.tables:
    known = {e['name'] for e in entries}
    unknown = set(args.tables) - known
    if unknown:
        parser.error(f'no registry entry in {args.library} for: {sorted(unknown)}')
    entries = [e for e in entries if e['name'] in args.tables]

for sub in ('decks', 'endf', 'thermal', 'validation', 'plots'):
    (out_dir / sub).mkdir(parents=True, exist_ok=True)

# NJOY and the ACE intermediates of process_thermal are several GB per table; keep them
# off the (RAM-backed) default /tmp.
tmp_dir = out_dir / 'tmp'
tmp_dir.mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(tmp_dir)


def _regenerate(entry):
    return tsl_leapr.regenerate_table(entry, args.temperatures, out_dir,
                                      njoy=args.njoy, nphon=args.nphon,
                                      work_dir=tmp_dir / f'leapr_{entry["name"]}')


# =========================================================================
# LEAPR, POST-EDIT AND VALIDATION

print(f'Regenerating {len(entries)} table(s) of {args.library} into {out_dir}')
with Pool(min(args.jobs, len(entries))) as pool:
    outputs = pool.map(_regenerate, entries)

# One line per table; rows of tables not in this run are kept from the previous summary.
summary_file = out_dir / 'validation' / 'summary.txt'
names = {o['entry']['name'] for o in outputs}
kept = []
if summary_file.exists():
    kept = [l for l in summary_file.read_text().splitlines()[1:]
            if l.strip() and l.split()[1] not in names]
summary = [validate.SUMMARY_HEADER]
summary += sorted(kept + [validate.summary_line(o['result']) for o in outputs],
                  key=lambda l: l.split()[1])
summary_file.write_text('\n'.join(summary) + '\n')
print('\n'.join(summary))

failed = [o for o in outputs if not o['passed']]
for o in failed:
    print(f'FAIL {o["entry"]["name"]}: ' + '; '.join(o['result']['fails']))

if args.validate_only:
    sys.exit(1 if failed else 0)

# =========================================================================
# HDF5

good = [o for o in outputs if o['passed']]
# A table regenerated for the first time may be present as a symlink to the shipped
# HDF5 file (written by an earlier run for the then-untouched tables); h5py would
# follow it and overwrite the shipped library, so drop the link first.
for o in good:
    h5 = out_dir / 'thermal' / f'{o["entry"]["name"]}.h5'
    if h5.is_symlink():
        h5.unlink()
with Pool(min(args.jobs, max(len(good), 1))) as pool:
    results = [pool.apply_async(process_thermal,
                                (o['entry']['neutron'], o['tape'],
                                 out_dir / 'thermal', args.libver))
               for o in good]
    for r in results:
        r.get()

regenerated = set()
for o in good:
    h5 = out_dir / 'thermal' / f'{o["entry"]["name"]}.h5'
    if not h5.exists():
        raise RuntimeError(f'process_thermal produced no {h5}')
    regenerated.add(h5.name)

# =========================================================================
# SYMLINK THE UNTOUCHED TABLES AND WRITE cross_sections.xml

built = registry.BUILT_LIBRARY[args.library]
linked = 0
for src in sorted((built / 'thermal').glob('c_*.h5')):
    dst = out_dir / 'thermal' / src.name
    if src.name in regenerated or dst.exists():
        continue
    dst.symlink_to(src.resolve())
    linked += 1
print(f'symlinked {linked} untouched thermal table(s) from {built / "thermal"}')

import openmc.data                                            # noqa: E402

library = openmc.data.DataLibrary()
for particle in ('neutron', 'photon'):
    d = built / particle
    if d.is_dir():
        for p in sorted(d.glob('*.h5')):
            library.register_file(p)
for p in sorted((out_dir / 'thermal').glob('*.h5')):
    library.register_file(p)
xml = out_dir / 'cross_sections.xml'
library.export_to_xml(xml)

check = openmc.data.DataLibrary.from_xml(xml)
missing = [lib['path'] for lib in check.libraries if not Path(lib['path']).exists()]
if missing:
    raise RuntimeError(f'{xml}: {len(missing)} registered file(s) missing, '
                       f'e.g. {missing[0]}')
print(f'wrote {xml} with {len(check.libraries)} entries, all present')

try:
    tmp_dir.rmdir()
except OSError:
    pass

sys.exit(1 if failed else 0)
