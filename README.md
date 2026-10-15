# Fair Aid QAOA

**Qompute 2026 | Track 2: Optimizing for People**

How should a limited aid budget balance overall benefit and priority for underserved communities? This project selects three of six equal-cost aid projects, models the budget as a quantum optimization constraint, and compares Qiskit QAOA with exact and heuristic classical solutions.

All inputs are synthetic teaching scores. The experiments use ideal simulation and do not establish quantum advantage or readiness for real aid allocation.
 
## Start here

- [Track 3 submission: executed notebook, figures, video and checklist](Track%203/bell_cloud/submission/README.md).

- [Track 3: Bell entanglement, hardware layout and noise experiments](Track%203/bell_cloud/README.md) — IBM QPU results, OpenQASM circuits and reproduction instructions.
- [Project guide and reproduction instructions](Track%202/fair_aid/README.md)
- [Executed submission notebook](Track%202/fair_aid/Track2_Submission.ipynb)
- [HTML presentation](Track%202/fair_aid/Track2_Submission.html) - download the file to view it in a browser; figures are embedded and formulas use the MathJax CDN.
- [Video outline and submission checklist](Track%202/fair_aid/submission/README.md)
- [Complete 80-run comparison](Track%202/fair_aid/results/comparison/summary.csv)

## What is included

The model combines net benefit, a weighted priority score, and a penalty for violating the exactly-three-project budget. Every Hamiltonian energy is checked against the direct cost for all 64 bitstrings.

The comparison runs X and XY QAOA variants across four priority weights, two circuit depths, and five initialization seeds: 80 runs with the same optimizer evaluation limit and sampling budget. Classical references include enumeration, greedy selection, one-swap improvement, and uniform feasible sampling. A further 500 synthetic score-perturbation scenarios per weight examine how input assumptions affect the chosen allocation.

| Recorded finding | Interpretation |
|---|---|
| All 40 XY runs have 100% ideal budget feasibility | The complete exchange gate pairs preserve the number of selected projects. |
| Only 3 of 8 XY weight/depth groups exceed the 5% uniform-feasible optimal probability on average | Feasibility alone does not establish better optimization or quantum advantage. |
| 48 of 80 runs did not meet the optimizer convergence criterion within the evaluation budget | All runs are retained; these are budget-limited results. |
| At priority weight 0.5, the original optimum survives 31.2% of the score-perturbation scenarios | The choice depends strongly on the assumed scores in this scenario analysis. |

The [project guide](Track%202/fair_aid/README.md) explains the objective, resource costs, fairness limitations, and interpretation of each metric. The [comparison manifest](Track%202/fair_aid/results/comparison/manifest.json) records the protocol, environment, and source/data hashes.

## Run locally

Use Python 3.13 and PowerShell from this repository's root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir 'downloads\track2\pip-cache' -r 'Track 2\fair_aid\requirements-lock.txt'
.\.venv\Scripts\python.exe -m unittest discover -s 'Track 2\fair_aid' -p 'test_*.py' -v
.\.venv\Scripts\python.exe 'Track 2\fair_aid\execute_notebook.py' --notebook Track2_Submission.ipynb
```

The submission notebook validates and reads the saved 80-run results by default. Set `RERUN_EXPERIMENTS = True` inside it, or run `benchmark.py`, to recompute the comparison. No IBM account or API token is required. Downloads and package caches stay under `downloads/track2/` and are excluded from Git.

## Project and challenge materials

The original work for this submission is under `Track 2/fair_aid/`. The original 24-run introductory experiment remains available alongside the expanded comparison. All project documentation, notebook explanations, code messages, and generated presentations are in English.

The original [challenge instructions](CHALLENGE.md) and supplied tutorials in `Track 1/`, `Track 2/`, and `Track 3/` are retained as reference materials from [sandraalhames/Qompute-2026](https://github.com/sandraalhames/Qompute-2026), checked at commit `855f3b7b7849b30cd63a07b730df97096fc8e317`. They are not claimed as this project's original work. The neighboring Max-Cut tutorial credits Benjamin Tirado / BasQ and carries its original CC BY 4.0 notice.

Team details, the 5-10 minute video, and the competition form submission are tracked in the [submission checklist](Track%202/fair_aid/submission/README.md).
