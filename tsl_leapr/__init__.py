"""Model-based regeneration of ENDF MF7 thermal scattering tables at exact temperatures.

S(alpha,beta) exists only on the evaluator's temperature grid: there is no
broadening operation for a bound scattering law, so a table at a new
temperature has to come from re-running the evaluation model.  This package
re-runs NJOY2016 LEAPR from the evaluation's own inputs (a shipped LEAPR deck,
or a FLASSH Control+DOS pair) over the union of the shipped grid and the
requested temperatures, post-edits the tape to the shipped B(1)/sigma_b and,
for the mixed-elastic Zr tables, rebuilds MT2 as LTHR=3, then validates the
result against the shipped file at every shipped temperature.

Tables without distributed inputs (kind 'inverted-dos': the JEFF-4.0 JSI
ZrH15/ZrH2 tables) get their phonon spectrum recovered from the shipped
S(alpha,beta) by invert.solve (one-phonon inversion on the file's own beta
nodes, refined by Newton steps against LEAPR); they are approximate at the
1e-3 level and carry their own gates (validate.GATE_SUM_ABS_KIND, GATE_DT).

Entry points
------------
temperature_list(shipped, requested)   union grid, ascending
regenerate_table(entry, ...)           run everything, return a result dict
regenerate(entry, temperatures, out_dir, njoy, nphon)  -> validated tape Path
"""
import shutil
from pathlib import Path

from . import decks, invert, postedit, registry, validate
from .leapr import run_leapr
from .mf7 import read_mt4

__all__ = ['decks', 'invert', 'postedit', 'registry', 'validate', 'run_leapr',
           'temperature_list', 'regenerate', 'regenerate_table',
           'hook_thermal_path', 'DEFAULT_TEMPERATURES']

DEFAULT_TEMPERATURES = [293.6, 298.6, 900.0, 905.0]


def temperature_list(shipped, requested):
    """Union of the shipped MF7/MT4 temperatures and the requested ones, ascending."""
    shipped_temps = [float(t) for t in read_mt4(shipped)['temps']]
    out = {round(t, 4): t for t in shipped_temps}
    for t in requested:
        out.setdefault(round(float(t), 4), float(t))
    return [out[k] for k in sorted(out)]


def regenerate_table(entry, temperatures=None, out_dir=None, njoy='njoy',
                     nphon=300, work_dir=None):
    """Deck -> LEAPR -> post-edit -> validation for one registry entry.

    Returns a dict with 'temps', 'meta', 'tape', 'result' (validate.compare) and
    'passed'.  Nothing is raised on a validation failure; the caller decides.
    """
    out_dir = Path(out_dir)
    for sub in ('decks', 'endf', 'validation'):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    name = entry['name']

    temps = temperature_list(entry['shipped'],
                             DEFAULT_TEMPERATURES if temperatures is None
                             else temperatures)
    work = Path(work_dir) if work_dir else out_dir / 'work' / name
    if entry['kind'] == 'inverted-dos':
        inv = invert.solve(entry, work / 'invert', njoy=njoy,
                           log_path=out_dir / 'validation' / f'{name}_inversion.txt',
                           rho_path=out_dir / 'decks' / f'{name}.rho.txt')
        deck_text, meta = decks.from_uniform_dos(
            entry['shipped'], inv['delta'], inv['rho'], temps, inv['nphon'],
            kind='inverted-dos', source=entry['shipped'], extra=inv['comments'])
        meta['inversion'] = inv
    else:
        deck_text, meta = decks.build(entry, temps, nphon=nphon)
    (out_dir / 'decks' / f'{name}.leapr').write_text(deck_text)

    tape, stdout = run_leapr(deck_text, work, njoy=njoy)
    (out_dir / 'decks' / f'{name}.stdout').write_text(stdout)
    if entry['kind'] == 'inverted-dos':
        # the model's own fidelity, measured before postedit rescales the tape to
        # the shipped values at the shipped temperatures; this is what is gated
        meta['model_result'] = validate.compare(entry, tape, temps,
                                                {k: v for k, v in meta.items()
                                                 if k != 'model_result'})

    edited = out_dir / 'endf' / f'{name}.endf'
    postedit.apply(tape, entry, temps, edited)
    shutil.rmtree(work, ignore_errors=True)

    result = validate.compare(entry, edited, temps, meta)
    validate.report(result, meta, out_dir / 'validation' / f'{name}.txt')
    return dict(entry=entry, temps=temps, meta=meta, tape=edited, result=result,
                passed=result['passed'])


def regenerate(entry, temperatures, out_dir, njoy='njoy', nphon=300):
    """Regenerate one table and return the validated tape path (raises on failure)."""
    out = regenerate_table(entry, temperatures, out_dir, njoy=njoy, nphon=nphon)
    if not out['passed']:
        raise RuntimeError(
            f'tsl_leapr validation failed for {entry["lib"]}/{entry["name"]}: '
            + '; '.join(out['result']['fails'])
            + f'  (report: {out_dir}/validation/{entry["name"]}.txt)')
    return out['tape']


def hook_thermal_path(lib, path_thermal, temperatures, out_dir, njoy='njoy',
                      nphon=300):
    """generate_*.py hook: regenerated tape for `path_thermal`, else `path_thermal`.

    `lib` is a registry library id (see registry.RELEASE_TO_LIB) or None.  With
    no temperatures requested, no known library, or no registry entry for this
    sab file, the
    shipped path is returned unchanged; otherwise LEAPR is re-run and the
    validated tape is returned (a validation failure raises).
    """
    if not temperatures or lib is None:
        return path_thermal
    entry = registry.lookup_sab(lib, Path(path_thermal).name)
    if entry is None:
        return path_thermal
    print(f'tsl_leapr: regenerating {lib}/{entry["name"]} from {entry["kind"]} model')
    return regenerate(entry, temperatures, out_dir, njoy=njoy, nphon=nphon)
