"""LEAPR deck builders.

from_leapr_deck()  re-emits an existing deck with a new temperature list
from_flassh()      builds a deck from a FLASSH Control + DOS pair
from_uniform_dos() builds a deck from a phonon spectrum on a uniform grid
                   (used by invert.solve for the 'inverted-dos' kind)

Deck mechanics (verified, see claude/LEAPR_Pipeline_Plan.md section 3):
the first temperature card is positive and is followed by the rho block
(delta/ni, rho values, twt/c/tbeta, nd); every further temperature is a single
negative card that reuses the previous rho and oscillator cards.  The list is
therefore written ascending, because LEAPR writes the MT2 W'(T) table in deck
order.  The whole list is always regenerated in one run; nothing is spliced.
"""
import datetime
from pathlib import Path

import numpy as np

from .mf7 import mf1_first_text, read_mt4, section_lines


LAT_REF = 0.0253        # eV, lat=1 reference energy


# ---------------------------------------------------------------------------
# card helpers
# ---------------------------------------------------------------------------


def fmt_temp(t):
    return f'{float(t):g}'


def quote(text):
    """One card-20 comment card: Fortran list-directed quoted string."""
    body = str(text)[:66].rstrip() or ' '
    return " '" + body.replace("'", "''") + "'/"


def wrap_comment(text, width=66, indent='  '):
    """Split a long comment into cards of at most `width` characters."""
    words = str(text).split()
    lines, cur = [], ''
    for w in words:
        pre = indent if lines else ''
        cand = (cur + ' ' + w) if cur else (pre + w)
        if len(cand) > width and cur:
            lines.append(cur)
            cur = indent + w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines or ['']


def fmt_values(vals, per=6, fmt='%.7g'):
    """Format a numeric card over as many lines as needed, terminated by '/'."""
    lines = []
    for i in range(0, len(vals), per):
        lines.append(' ' + ' '.join(fmt % v for v in vals[i:i + per]))
    lines[-1] += '/'
    return '\n'.join(lines)


def _tokens(line):
    """List-directed tokens of one card line ('/' terminates the record)."""
    return line.split('/')[0].split()


class _DeckReader:
    """Line-oriented reader for a LEAPR deck.

    Value cards (alpha, beta, rho) are not reliably terminated by '/' -- the
    VIII.0 Zr deck terminates neither its alpha nor its beta grid -- so blocks
    are consumed by value count, not by looking for a slash.
    """

    def __init__(self, text):
        self.lines = [l.rstrip() for l in text.splitlines()]
        self.i = 0

    def peek(self):
        while self.i < len(self.lines) and not self.lines[self.i].strip():
            self.i += 1
        return self.lines[self.i] if self.i < len(self.lines) else None

    def card(self):
        """One single-line card, returned verbatim."""
        line = self.peek()
        if line is None:
            raise ValueError('unexpected end of LEAPR deck')
        self.i += 1
        return line

    def block(self, n):
        """As many lines as needed to supply n values, returned verbatim."""
        used, count = [], 0
        while count < n:
            line = self.peek()
            if line is None:
                raise ValueError(f'LEAPR deck ended after {count} of {n} values')
            self.i += 1
            used.append(line)
            count += len(_tokens(line))
        return used

    def rest(self):
        out = []
        while True:
            line = self.peek()
            if line is None or line.strip().lower() == 'stop':
                break
            self.i += 1
            out.append(line)
        return out


def unquote(card):
    s = card.strip()
    if s.endswith('/'):
        s = s[:-1].rstrip()
    if len(s) >= 2 and s.startswith("'") and s.endswith("'"):
        return s[1:-1].replace("''", "'")
    return s


def build_comments(shipped, kind, source, temps, extra=()):
    """Card-20 comment cards (section 2a of the build spec).

    The first card is the shipped MF1/451 first text line, so LEAPR reproduces
    the shipped ZSYMAM in columns 1-11 and OpenMC derives the same table name.
    At least five text records are emitted: openmc.data.endf.Evaluation only
    parses ZSYMAM when MF1/451 carries five or more of them.
    """
    today = datetime.date.today().isoformat()
    cards = [mf1_first_text(shipped)]
    cards += wrap_comment(f'regenerated: njoy2016.79 leapr, tsl_leapr {today}, '
                          f'model {kind} from {Path(source).name}')
    cards += wrap_comment('temperatures (K): '
                          + ' '.join(fmt_temp(t) for t in temps))
    cards += list(extra)
    while len(cards) < 5:
        cards.append(' ')
    return cards


# ---------------------------------------------------------------------------
# deck kind
# ---------------------------------------------------------------------------


def from_leapr_deck(deck_path, shipped, temps):
    """Re-emit an existing LEAPR deck for a new (ascending) temperature list.

    Cards 1-3 and 5-10 (mat/za, awr/spr/npr/iel, secondary scatterer, alpha and
    beta grids) and the whole rho block are kept verbatim; only ntempr, the
    first (positive) temperature, the negative repeat temperatures and the
    card-20 comments are rewritten.  The deck's own nphon is preserved (the
    VIII.0 decks give none and get the LEAPR default, the IKE deck gives 200).
    """
    r = _DeckReader(Path(deck_path).read_text(errors='replace'))
    c_module = r.card()                                     # ' leapr'
    c_nout = r.card()                                       # ' 20'
    c_title = r.card()
    c_ntempr = r.card()
    c_mat = r.card()
    c_awr = r.card()
    c_nss = r.card()
    c_grid = r.card()

    tok = _tokens(c_ntempr)
    nphon = int(tok[2]) if len(tok) > 2 else None
    card_ntempr = f' {len(temps)} {tok[1] if len(tok) > 1 else 1}'
    card_ntempr += f' {nphon}/' if nphon is not None else '/'

    g = _tokens(c_grid)
    nalpha, nbeta = int(g[0]), int(g[1])
    alpha = r.block(nalpha)
    beta = r.block(nbeta)

    r.card()                                                # first temperature
    c_delta = r.card()
    ni = int(_tokens(c_delta)[1])
    rho = r.block(ni)
    c_twt = r.card()
    c_nd = r.card()
    nd = int(_tokens(c_nd)[0] or 0)
    osc = (r.block(nd) + r.block(nd)) if nd > 0 else []

    tail = r.rest()
    deck_comments = [unquote(l) for l in tail if l.lstrip().startswith("'")]
    comments = build_comments(shipped, 'deck', deck_path, temps, extra=deck_comments)

    out = [c_module, c_nout, c_title, card_ntempr, c_mat, c_awr, c_nss, c_grid]
    out += alpha + beta
    out.append(f' {fmt_temp(temps[0])}/')
    out += [c_delta] + rho + [c_twt, c_nd] + osc
    out += [f' -{fmt_temp(t)}/' for t in temps[1:]]
    out += [quote(c) for c in comments]
    out += [' /', ' stop', '']

    meta = dict(kind='deck', source=Path(deck_path), nphon=nphon, beta_cut=None,
                ncomments=len(comments))
    return '\n'.join(out), meta


# ---------------------------------------------------------------------------
# flassh-dos kind
# ---------------------------------------------------------------------------


def parse_control(path):
    out = {}
    for line in open(path):
        if '/' not in line:
            continue
        val, com = line.split('/', 1)
        com = com.strip().lower()
        if com.startswith('list of temperatures'):
            out['temps'] = [float(x) for x in val.replace(',', ' ').split()]
        elif com.startswith('maximum or specified phonon order'):
            out['nphon'] = int(val)
        elif com.startswith('free atom cross section'):
            out['spr'] = float(val)
        elif com.startswith('free atom incoherent'):
            out['sinc'] = float(val)
        elif 'amu mass' in com:
            out['amu'] = float(val)
        elif com.startswith('material number'):
            out['mat'] = int(val)
    return out


def parse_dos(path):
    lines = open(path).read().splitlines()
    delta = float(lines[0])
    n = int(lines[1])
    vals = []
    for l in lines[2:]:
        if '/' in l and len(vals) >= n:
            break
        vals += [float(x) for x in l.split('/')[0].replace(',', ' ').split()]
        if len(vals) >= n:
            break
    return delta, np.array(vals[:n])


def from_flassh(shipped, control, dos, temps, nphon=300):
    """Build a LEAPR deck from a FLASSH Control + DOS pair.

    Grids, ZA, MAT and AWR come from the shipped ENDF file so that the
    regenerated tape is directly comparable; spr comes from the Control file
    ('free atom cross section') and rho from the DOS file.  iel=-1 selects
    incoherent elastic; the mixed-elastic Zr tables get their coherent part
    rebuilt afterwards by postedit.lthr3_section().
    """
    d = read_mt4(shipped)
    mat = int(section_lines(shipped, 7, 4)[0][66:70])
    c = parse_control(control)
    delta, rho = parse_dos(dos)
    temps = list(temps)

    comments = build_comments(shipped, 'flassh-dos', dos, temps)
    out = [' leapr', ' 20',
           f" 'leapr from flassh dos, mat {mat}'/",
           f' {len(temps)} 1 {nphon}/',
           f' {mat} {d["za"]:.0f}./',
           f' {d["awr"]:.7g} {c["spr"]:.7g} 1 -1/',
           ' 0/',
           f' {len(d["alpha"])} {len(d["beta"])} 1/',
           fmt_values(d['alpha']),
           fmt_values(d['beta']),
           f' {fmt_temp(temps[0])}/',
           f' {delta:g} {len(rho)}/',
           fmt_values(rho, fmt='%.8g'),
           ' 0. 0. 1./',
           ' 0/']
    out.extend(f' -{fmt_temp(t)}/' for t in temps[1:])
    out.extend(quote(x) for x in comments)
    out += [' /', ' stop', '']

    meta = dict(kind='flassh-dos', source=Path(dos), nphon=nphon,
                beta_cut=delta * (len(rho) - 1) / LAT_REF,
                ncomments=len(comments))
    return '\n'.join(out), meta


# ---------------------------------------------------------------------------
# uniform-grid rho (inverted-dos kind)
# ---------------------------------------------------------------------------


def from_uniform_dos(shipped, delta, rho, temps, nphon, kind='inverted-dos',
                     source=None, extra=(), spr=None):
    """Build a LEAPR deck from a phonon spectrum rho on a uniform grid.

    Grids, ZA, MAT and AWR come from the shipped ENDF file; spr defaults to the
    shipped B(1) (the JSI files carry the free cross section there, so LEAPR's
    sigma_b = spr ((A+1)/A)^2 reproduces the shipped MT2 sigma_b).  Continuous
    spectrum only (twt=0, no discrete oscillators), iel=-1 incoherent elastic.
    """
    d = read_mt4(shipped)
    mat = int(section_lines(shipped, 7, 4)[0][66:70])
    temps = list(temps)
    rho = np.asarray(rho, dtype=float)
    if spr is None:
        spr = float(d['B'][0])
    source = shipped if source is None else source
    comments = build_comments(shipped, kind, source, temps, extra=list(extra))
    out = [' leapr', ' 20',
           f" 'leapr from inverted rho, mat {mat}'/",
           f' {len(temps)} 1 {nphon}/',
           f' {mat} {d["za"]:.0f}./',
           f' {d["awr"]:.7g} {spr:.7g} 1 -1/',
           ' 0/',
           f' {len(d["alpha"])} {len(d["beta"])} 1/',
           fmt_values(d['alpha']),
           fmt_values(d['beta']),
           f' {fmt_temp(temps[0])}/',
           f' {delta:.8g} {len(rho)}/',
           fmt_values(rho, fmt='%.8g'),
           ' 0. 0. 1./',
           ' 0/']
    out.extend(f' -{fmt_temp(t)}/' for t in temps[1:])
    out.extend(quote(x) for x in comments)
    out += [' /', ' stop', '']
    meta = dict(kind=kind, source=Path(source), nphon=nphon, beta_cut=None,
                ncomments=len(comments))
    return '\n'.join(out), meta


# ---------------------------------------------------------------------------


def build(entry, temps, nphon=300):
    """Deck text + metadata for a registry entry and a temperature list."""
    if entry['kind'] == 'deck':
        return from_leapr_deck(entry['deck'], entry['shipped'], temps)
    if entry['kind'] == 'flassh-dos':
        return from_flassh(entry['shipped'], entry['control'], entry['dos'],
                           temps, nphon=nphon)
    if entry['kind'] == 'inverted-dos':
        raise ValueError("kind 'inverted-dos' needs LEAPR runs to recover rho: "
                         "use tsl_leapr.regenerate_table, which calls invert.solve "
                         "and then from_uniform_dos")
    raise ValueError(f'unknown model kind {entry["kind"]!r}')
