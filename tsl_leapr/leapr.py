"""Run NJOY LEAPR on a deck text in a work directory."""
import subprocess
from pathlib import Path


def run_leapr(deck_text, work_dir, njoy='njoy'):
    """Run `njoy < input` in `work_dir`; return (tape20 path, stdout text)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    deck = work_dir / 'input'
    deck.write_text(deck_text)
    tape = work_dir / 'tape20'
    tape.unlink(missing_ok=True)
    with open(deck) as fin:
        proc = subprocess.run([njoy], stdin=fin, cwd=work_dir,
                              capture_output=True, text=True)
    stdout = proc.stdout + proc.stderr
    (work_dir / 'stdout').write_text(stdout)
    if proc.returncode != 0:
        raise RuntimeError(f'{njoy} exited {proc.returncode} in {work_dir}\n{stdout[-2000:]}')
    if 'error' in stdout.lower():
        raise RuntimeError(f'{njoy} reported an error in {work_dir}\n{stdout[-2000:]}')
    if not tape.exists():
        raise RuntimeError(f'{njoy} produced no tape20 in {work_dir}\n{stdout[-2000:]}')
    return tape, stdout
