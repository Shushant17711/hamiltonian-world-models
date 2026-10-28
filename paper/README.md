# Paper draft

`main.tex` + `refs.bib`: a 4-page negative-results write-up of this study, aimed at workshops such as
ICBINB ("I Can't Believe It's Not Better"). Figures come from `../results/figures/`.

Build (no LaTeX on the dev machine): upload `main.tex`, `refs.bib` and the two figures
`error_vs_horizon.png` and `advantage_vs_lyapunov.png` to Overleaf (put the PNGs next to `main.tex`), or run
`latexmk -pdf main.tex` locally. Before submitting, swap the preamble for the venue's style file and
check the bibliography entries.

Every number in the text is from `results/verdicts.md` or the README's exploratory return table.
