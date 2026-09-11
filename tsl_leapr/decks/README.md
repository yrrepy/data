# Decks shipped with tsl_leapr

`ike_HinZrH.leapr` is the IKE Stuttgart H(ZrH) LEAPR deck (JEFF-3.1 heritage, carried
into JEFF-4.0 as `tsl_H_ZrH.jeff`), transcribed from IAEA report INDC(NDS)-0470,
appendix 10.2.2 — 147 alpha, 185 beta, a 181-point phonon spectrum on a 1 meV grid,
`nphon = 200`, and the secondary-scatterer card 6 `1 1 90.436 6.37 1` that treats
zirconium as a free gas (B(7)=1, AWS 90.436, SPS 6.37). JEFF ships no model inputs for
this table, so the deck lives here rather than in the library tree; it reproduces the
shipped file to max rel 9e-5 at all eight shipped temperatures, with W'(T) and sigma_b
exact. The ENDF/B-VIII.0 and VIII.1 decks are *not* copied here: they are shipped
alongside the evaluations as `endfb-viii.{0,1}-endf/thermal/tsl-{H,Zr}inZrH.leapr` and
are read from there, so the regenerated tables stay tied to whatever deck the library
release actually distributes (JENDL-5 reuses the VIII.0 decks, its MF7 being a copy).
Note that these decks state 293.6 K while the shipped file's first temperature is 296 K;
the regenerated temperature list is built from the shipped MF7 grid, so both appear.
