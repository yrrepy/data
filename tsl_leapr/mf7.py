"""ENDF-6 MF7 reader (MT2 elastic, MT4 inelastic) plus minimal ENDF record writers.

The reader is lifted from claude/leapr_proto/mf7.py (dependency-free, tolerant of the
JEFF habit of embedding blanks inside floats).  The writer side supplies 11-character
ENDF floats, CONT/LIST/TAB1 records with MAT/MF/MT in columns 67-75 and 5-digit
sequence numbers in columns 76-80, and a helper that replaces one MF/MT section of a
tape in place.
"""
import re
from pathlib import Path

import numpy as np

_FLOAT_RE = re.compile(r'^([+-]?\d*\.?\d*)([+-]\d+)$')

# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def ffloat(s):
    """Parse an 11-column ENDF float ('2.043634+1', ' 1.07000 +2', '1.0e-3')."""
    s = s.strip().replace(' ', '')
    if not s:
        return 0.0
    m = _FLOAT_RE.match(s)
    if m and 'e' not in s.lower() and 'd' not in s.lower():
        return float(m.group(1) + 'e' + m.group(2))
    return float(s.replace('d', 'e').replace('D', 'e'))


def fint(s):
    s = s.strip()
    return int(s) if s else 0


def tape_lines(path):
    """All lines of an ENDF tape, newline stripped, as a list of str."""
    with open(path, errors='replace') as fh:
        return [line.rstrip('\n').rstrip('\r') for line in fh]


def line_mf(line):
    return fint(line[70:72]) if len(line) >= 72 else 0


def line_mt(line):
    return fint(line[72:75]) if len(line) >= 75 else 0


def line_mat(line):
    return fint(line[66:70]) if len(line) >= 70 else 0


def section_lines(path, mf, mt):
    """Lines of one MF/MT section of a tape (as a list of str)."""
    return [l for l in tape_lines(path) if line_mf(l) == mf and line_mt(l) == mt]


def section_span(lines, mf, mt):
    """(first, last+1) indices of the MF/MT body inside an already-read tape."""
    idx = [i for i, l in enumerate(lines) if line_mf(l) == mf and line_mt(l) == mt]
    if not idx:
        raise KeyError(f'no MF{mf}/MT{mt} section on tape')
    return idx[0], idx[-1] + 1


class Rec:
    """Sequential ENDF record reader over a list of section lines."""

    def __init__(self, lines):
        self.lines = lines
        self.i = 0

    def cont(self):
        l = self.lines[self.i]
        self.i += 1
        return [ffloat(l[0:11]), ffloat(l[11:22]), fint(l[22:33]),
                fint(l[33:44]), fint(l[44:55]), fint(l[55:66])]

    def floats(self, n):
        vals = []
        while len(vals) < n:
            l = self.lines[self.i]
            self.i += 1
            for k in range(6):
                if len(vals) < n:
                    vals.append(ffloat(l[11 * k:11 * k + 11]))
        return np.array(vals)

    def ints(self, n):
        vals = []
        while len(vals) < n:
            l = self.lines[self.i]
            self.i += 1
            for k in range(6):
                if len(vals) < n:
                    vals.append(fint(l[11 * k:11 * k + 11]))
        return vals

    def list(self):
        c = self.cont()
        return c, self.floats(c[4])

    def tab1(self):
        c = self.cont()
        nr, np_ = c[4], c[5]
        interp = self.ints(2 * nr)
        xy = self.floats(2 * np_)
        return c, interp, xy[0::2], xy[1::2]

    def tab2(self):
        c = self.cont()
        interp = self.ints(2 * c[4])
        return c, interp


def read_mt4(path):
    """MF7/MT4 inelastic S(alpha,beta): grids, temperatures, B array, S[nT,nbeta,nalpha]."""
    r = Rec(section_lines(path, 7, 4))
    head = r.cont()
    za, awr, lat, lasym = head[0], head[1], head[3], head[4]
    c, B = r.list()
    lln = c[2]
    c2, _ = r.tab2()
    nb = c2[5]
    betas, temps, S, alphas = [], None, [], None
    for _ in range(nb):
        c, interp, alpha, s0 = r.tab1()
        T0, beta, lt = c[0], c[1], c[2]
        betas.append(beta)
        tlist = [T0]
        slist = [s0]
        for _ in range(lt):
            cl, sv = r.list()
            tlist.append(cl[0])
            slist.append(sv)
        if temps is None:
            temps = tlist
            alphas = alpha
        S.append(np.array(slist))              # (nT, nalpha)
    S = np.transpose(np.array(S), (1, 0, 2))   # (nT, nbeta, nalpha)
    if lln == 1:
        S = np.exp(S)
    teff = []
    while r.i < len(r.lines) - 1:
        try:
            c, interp, x, y = r.tab1()
            teff.append((x, y))
        except Exception:
            break
    return dict(za=za, awr=awr, lat=lat, lasym=lasym, lln=lln, B=B,
                temps=np.array(temps), alpha=np.array(alphas), beta=np.array(betas),
                S=S, teff=teff)


def mt4_slots(section):
    """Positions of every S(alpha,beta) and T_eff value inside an MF7/MT4 section.

    `section` is the list of the section's lines (see section_lines).  Returns
    dict(temps, nalpha, nbeta, s=[(line, field) per value] indexed [iT][ib][ia],
    teff=[(line, field) per temperature] for the first T_eff TAB1) so a value can
    be replaced in place with set_field() without touching the record layout.
    """
    r = Rec(section)
    r.cont()                                    # HEAD
    c = r.cont()                                # B LIST head
    r.floats(c[4])
    c2 = r.cont()                               # TAB2 head
    r.ints(2 * c2[4])
    nb = c2[5]
    temps, slots = None, None
    for ib in range(nb):
        c = r.cont()                            # TAB1 head: T0, beta, LT, 0, NR, NP
        lt, nr, np_ = c[2], c[4], c[5]
        r.ints(2 * nr)
        if temps is None:
            temps = [c[0]]
            slots = [[None] * nb for _ in range(lt + 1)]
        start = r.i
        slots[0][ib] = [(start + (2 * j + 1) // 6, (2 * j + 1) % 6) for j in range(np_)]
        r.floats(2 * np_)
        for it in range(1, lt + 1):
            cl = r.cont()                       # LIST head: T, beta, LI, 0, NP, 0
            if ib == 0:
                temps.append(cl[0])
            start = r.i
            slots[it][ib] = [(start + j // 6, j % 6) for j in range(cl[4])]
            r.floats(cl[4])
    teff = []
    if r.i < len(section) - 1:
        c = r.cont()                            # T_eff TAB1 head
        r.ints(2 * c[4])
        start = r.i
        teff = [(start + (2 * j + 1) // 6, (2 * j + 1) % 6) for j in range(c[5])]
    return dict(temps=np.array(temps), nalpha=len(slots[0][0]), nbeta=nb, s=slots, teff=teff)


def read_mt2(path):
    """MF7/MT2 elastic.  Returns lthr plus 'coh' (LTHR 1/3) and 'inc' (LTHR 2/3)."""
    lines = section_lines(path, 7, 2)
    if not lines:
        return None
    r = Rec(lines)
    head = r.cont()
    lthr = head[2]
    out = dict(lthr=lthr, za=head[0], awr=head[1])
    if lthr in (1, 3):
        c, interp, E, s0 = r.tab1()
        temps = [c[0]]
        svals = [s0]
        li = []
        for _ in range(c[2]):
            cl, sv = r.list()
            temps.append(cl[0])
            svals.append(sv)
            li.append(cl[2])
        out['coh'] = dict(temps=np.array(temps), E=E, S=np.array(svals),
                          interp=interp, li=li)
    if lthr in (2, 3):
        c, interp, T, W = r.tab1()
        out['inc'] = dict(sb=c[0], temps=T, W=W, interp=interp)
    return out


def mf1_text_lines(path):
    """The TEXT records of MF1/MT451 (i.e. everything after HEAD + three CONT)."""
    lines = section_lines(path, 1, 451)
    return [l[:66] for l in lines[4:]]


def mf1_first_text(path):
    """First MF1/451 text line (ZSYMAM / lab / date / author), 66 columns, rstripped."""
    txt = mf1_text_lines(path)
    if not txt:
        raise ValueError(f'{path}: no MF1/451 text records')
    return txt[0].rstrip()


def field(line, k):
    """The k-th 11-character field (0-based) of an ENDF line."""
    return line[11 * k:11 * k + 11]


def set_field(line, k, text):
    """Replace the k-th 11-character field, keeping the rest of the 80-column line."""
    if len(text) != 11:
        raise ValueError(f'field text must be 11 characters, got {len(text)!r}')
    return line[:11 * k] + text + line[11 * k + 11:]


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------


def endf_float(v):
    """Format a float as an 11-character ENDF field (7 significant digits)."""
    v = float(v)
    if v == 0.0:
        return ' 0.000000+0'
    ndig = 6
    while True:
        mant, ex = f'{v:.{ndig}E}'.split('E')
        e = int(ex)
        out = f'{mant}{"+" if e >= 0 else "-"}{abs(e)}'
        if len(out) <= 11 or ndig == 0:
            break
        ndig -= 1
    return f'{out:>11s}'


class SectionWriter:
    """Builds the 80-column lines of one MF/MT section."""

    def __init__(self, mat, mf, mt):
        self.mat, self.mf, self.mt = mat, mf, mt
        self.lines = []

    def _emit(self, body):
        seq = len(self.lines) % 99999 + 1      # NJOY endf.f90: wraps to 1 past 99999
        self.lines.append(f'{body:<66s}{self.mat:>4d}{self.mf:>2d}{self.mt:>3d}'
                          f'{seq:>5d}')

    def cont(self, c1, c2, l1, l2, n1, n2):
        self._emit(f'{endf_float(c1)}{endf_float(c2)}'
                   f'{l1:>11d}{l2:>11d}{n1:>11d}{n2:>11d}')

    def _floats(self, vals):
        for i in range(0, len(vals), 6):
            self._emit(''.join(endf_float(v) for v in vals[i:i + 6]))

    def _ints(self, vals):
        for i in range(0, len(vals), 6):
            self._emit(''.join(f'{v:>11d}' for v in vals[i:i + 6]))

    def list(self, c1, c2, l1, l2, n2, values):
        self.cont(c1, c2, l1, l2, len(values), n2)
        self._floats(values)

    def tab1(self, c1, c2, l1, l2, interp, x, y):
        self.cont(c1, c2, l1, l2, len(interp) // 2, len(x))
        self._ints(list(interp))
        xy = np.empty(2 * len(x))
        xy[0::2] = x
        xy[1::2] = y
        self._floats(xy)


def replace_section(lines, mf, mt, new_lines):
    """Return a new tape line list with the MF/MT body replaced (SEND kept)."""
    a, b = section_span(lines, mf, mt)
    return lines[:a] + list(new_lines) + lines[b:]


def write_tape(path, lines):
    Path(path).write_text('\n'.join(lines) + '\n')
