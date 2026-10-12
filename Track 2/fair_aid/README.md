# Fair Aid QAOA

**Qompute 2026 · Track 2: Optimizing for People**

How does an aid allocation change when its objective gives more weight to underserved communities? This project selects exactly three of six equal-cost projects, compares two Qiskit QAOA approaches with classical solutions, and makes the policy choices behind the objective visible.

**All project names, benefit scores, overlap penalties, and priority scores are synthetic teaching inputs.** They are not observations of real communities or beneficiary counts. The deliberately constructed tradeoff does not imply that serving underserved communities has lower real-world benefit. This is a small, ideal-simulator experiment with no claim of quantum advantage or readiness for operational aid decisions.

Start with the [submission notebook](Track2_Submission.ipynb) or [HTML presentation](Track2_Submission.html). Download the HTML file and open it locally; plots are embedded, while mathematical typesetting uses the MathJax CDN. The [original 24-run notebook](Fair_Aid_QAOA.ipynb) remains available as the first experiment. The [submission guide and video outline](submission/README.md) track the remaining presentation and submission work.

## What the experiment contributes

The original [Track 2 starter notebook](../Track2_challenge_2_tutorial.ipynb) demonstrates Max-Cut. This independent implementation instead studies a constrained allocation problem, with explicit benefit–priority tradeoffs, a mathematically checked penalty, an XY mixer that preserves the number of selected projects, classical optimization baselines, score-perturbation analysis, and a comparison with an explicit minimum-service quota. The original tutorial is retained unchanged.

| Track 2 requirement | Evidence in this project |
|---|---|
| Explain a concrete scenario and who is affected | Six candidate aid services; existing service points A–C and underserved-community services D–F; assumptions and omissions below |
| Encode the objective and constraints in a cost Hamiltonian, including a penalty | `allocation.py`; the model below; exhaustive Hamiltonian checks and `results/penalty_sweep.csv` |
| Run QAOA with Qiskit and check against a classical solution | `run_experiments.py`, `benchmark.py`, `mixers.py`, all 64 bitstrings enumerated, and every feasible allocation in `results/exact_solutions.csv` |
| Reflect on weights, penalties, and omitted considerations | Separate benefit and priority outcomes, score perturbations, a classical quota-policy comparison, and the discussion of who sets the objective |

These requirements come from the [official challenge repository](https://github.com/sandraalhames/Qompute-2026#track-2-optimizing-for-people), checked against [README commit 855f3b7](https://github.com/sandraalhames/Qompute-2026/blob/855f3b7b7849b30cd63a07b730df97096fc8e317/README.md) on 2026-10-12. Real IBM hardware is welcome but is not required by its instructions; a simulator project is eligible.

## Reproduce from a fresh checkout

The commands below use **PowerShell from the repository root**, the directory containing `Track 2`. They do not assume that the existing local virtual environment or downloaded wheels were included in the checkout. Use Python 3.13; the recorded run's exact interpreter and package versions are in [the comparison manifest](results/comparison/manifest.json).

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --cache-dir 'downloads\track2\pip-cache' -r 'Track 2\fair_aid\requirements-lock.txt'
.\.venv\Scripts\python.exe -m unittest discover -s 'Track 2\fair_aid' -p 'test_*.py' -v
.\.venv\Scripts\python.exe 'Track 2\fair_aid\execute_notebook.py' --notebook Track2_Submission.ipynb
```

If the Windows Python launcher `py` is unavailable, use a Python 3.13 executable for the first command. Package installation needs internet access; running the installed experiment requires no IBM account, API token, or external dataset. Download caches stay under the repository's `downloads/track2/` directory. The lock file records this project's environment; another operating system may need platform-appropriate wheels from PyPI.

By default, the submission notebook validates and reads the saved 80-run comparison in `results/comparison/`, including source hashes, then updates its figures and HTML export. This is result validation and presentation, not a new optimization run. Set `RERUN_EXPERIMENTS = True` in the notebook to recompute all 80 runs; when no result snapshot exists, it runs the experiment automatically.

To recompute the expanded comparison and its figures from the command line:

```powershell
.\.venv\Scripts\python.exe 'Track 2\fair_aid\benchmark.py'
.\.venv\Scripts\python.exe 'Track 2\fair_aid\plot_comparison.py'
.\.venv\Scripts\python.exe 'Track 2\fair_aid\execute_notebook.py' --notebook Track2_Submission.ipynb
```

The original 24-run notebook is separate. To recompute that experiment and its HTML export:

```powershell
.\.venv\Scripts\python.exe 'Track 2\fair_aid\execute_notebook.py' --notebook Fair_Aid_QAOA.ipynb
```

Reruns overwrite their output directories. To try a small version of the original experiment in a separate directory:

```powershell
.\.venv\Scripts\python.exe 'Track 2\fair_aid\run_experiments.py' --quick --output 'Track 2\fair_aid\results_quick'
```

Do not treat a quick run as the full submission benchmark. Generated virtual environments, package caches, and Python caches are excluded from Git; retain the input CSVs, code, notebook, HTML, and reproducibility outputs.

For an offline installation on the original Windows x64 / Python 3.13 environment, previously downloaded wheels may be available in `downloads/track2/wheels/`. These files are not part of a fresh GitHub checkout. Download compatible packages first if needed:

```powershell
.\.venv\Scripts\python.exe -m pip download --dest 'downloads\track2\wheels' --cache-dir 'downloads\track2\pip-cache' -r 'Track 2\fair_aid\requirements-lock.txt'
.\.venv\Scripts\python.exe -m pip install --no-index --find-links 'downloads\track2\wheels' -r 'Track 2\fair_aid\requirements-lock.txt'
```

## Scenario and objective

An aid organization has three budget units and six projects costing one unit each. It must use all three units. A–C represent established, more accessible service points; D–F represent mobile, remote-community, or accessible home-delivery services. These labels are scenario assumptions, not measurements.

| Project | Example service | Benefit `b` | Priority `e` |
|---|---|---:|---:|
| A | Central food hub | 9 | 0 |
| B | Established distribution center | 8.5 | 0.5 |
| C | Transit-accessible supply point | 7 | 1 |
| D | Mobile outreach team | 6.5 | 2 |
| E | Remote neighborhood delivery | 6 | 3 |
| F | Accessible home delivery | 5.5 | 4 |

The inputs are [projects.csv](data/projects.csv), with one row per project and `unit_cost=1`, and [overlaps.csv](data/overlaps.csv), with one penalty per unordered project pair. Omitted pairs have zero overlap; each listed pair is deducted once. Overlap scores are not validated counts of duplicate beneficiaries.

Let `x_i` indicate whether project `i` is selected:

```text
B(x) = Σ b_i x_i − Σ_{i<j} o_ij x_i x_j     net benefit
E(x) = Σ e_i x_i                            priority score
U_w(x) = B(x) + w E(x)                      utility to maximize
Σ x_i = K = 3                              exact budget constraint
C(x) = −U_w(x) + A(Σ x_i − K)²              cost to minimize
```

Benefit and overlap use synthetic benefit points. Priority uses synthetic priority points; `w` converts priority points into benefit points. Increasing `w` changes the policy objective. Increasing `A` strengthens the budget penalty. Neither operation establishes that an allocation is fair.

For a QUBO `c + Σ a_i x_i + Σ_{i<j} Q_ij x_i x_j`:

```text
c = A K²
a_i = −(b_i + w e_i) + A(1 − 2K)
Q_ij = o_ij + 2A
```

The substitution `x_i = (I − Z_i)/2` gives the diagonal Ising Hamiltonian:

```text
h_i = −a_i/2 − Σ_{j≠i} Q_ij/4
J_ij = Q_ij/4
offset = c + Σ a_i/2 + Σ_{i<j} Q_ij/4
H = offset I + Σ h_i Z_i + Σ_{i<j} J_ij Z_i Z_j
```

In the symmetric sum, `Q_ij` denotes the coefficient of an unordered project pair. The code stores only the upper triangle to avoid double counting. The implementation checks Hamiltonian energy against the direct objective on every bitstring. Project A is qubit 0, the rightmost bit in a Qiskit count string; displayed strings run F…A.

The penalty uses `A = 20`. Repairing an infeasible allocation by one step toward three selected projects increases its unpenalized cost by at most `max_i(max(g_i, Σ_j o_ij − g_i))`, where `g_i = b_i + w e_i`, while decreasing its squared penalty by at least `A`. Choosing `A` strictly above this bound therefore makes every exact global minimizer feasible. The largest bound over `w` in `[0,2]` is 13.5, so `A=20` is sufficient. This does not make every shallow-QAOA measurement feasible. The equal-cost, exactly-three model is narrower than a general knapsack problem; unequal costs require a new encoding, such as a budget inequality with slack variables. Editing `unit_cost` alone is not supported.

## Exact policy tradeoff

The reference enumerates all 64 states, including the 20 feasible allocations. Its unique feasible optima are:

| Priority weight `w` | Selected projects | Net benefit `B` | Priority `E` | Utility `U_w` | Selected projects from D–F |
|---:|---|---:|---:|---:|---:|
| 0 | A, B, D | 22.1 | 2.5 | 22.1 | 1 |
| 0.5 | A, B, F | 21.3 | 4.5 | 23.55 | 1 |
| 1 | A, E, F | 19.4 | 7 | 26.4 | 2 |
| 2 | D, E, F | 16.7 | 9 | 34.7 | 3 |

Source: [exact_solutions.csv](results/exact_solutions.csv). Compare benefit and priority separately: utility values in different rows use different objectives. The D–F project count is not population coverage or a fairness rate; the dataset has no population or need denominators.

## Quantum and classical comparison

The expanded comparison uses four weights `{0, 0.5, 1, 2}`, two depths `{1, 2}`, five initialization seeds `{11, 29, 47, 71, 97}`, and two quantum approaches: **80 runs**. Each run allows at most 300 COBYLA objective evaluations and then takes 2,048 samples with Qiskit's `StatevectorSampler`. All runs, including those reaching the evaluation limit, are retained. The precise configuration is recorded in [comparison/manifest.json](results/comparison/manifest.json); per-run outcomes are in [comparison/summary.csv](results/comparison/summary.csv).

The original 24-run X-mixer experiment remains in `results/`, with three seeds `{11, 29, 47}`, a 160-evaluation limit, and 2,048 final samples per run. Its plots show the minimum-to-maximum range over those three initializations; depth 2 does not outperform depth 1 in every setting. Treat it as a separate experiment; use the expanded comparison for the matched two-mixer protocol.

- **X mixer with a penalty:** starts from a uniform superposition of all 64 bitstrings and uses the budget penalty to discourage invalid allocations.
- **XY ring mixer:** starts from the uniform superposition of the 20 feasible states. Each ring edge applies an `RXX(beta)` / `RYY(beta)` exchange pair. The ordered product preserves Hamming weight in an ideal circuit after each complete pair. It is one ring sweep, not the exact exponential of the sum of all ring interactions.

The XY starting state depends only on the number of projects and the budget, not the known optimum. It is prepared with a dense state vector and Qiskit's general `StatePreparation` synthesis. That preparation is included in circuit resource counts; it is not an efficient, scalable Dicke-state preparation claim. Hardware noise could violate the ideal feasibility property.

Both approaches retain the same full cost Hamiltonian with `A=20` and the same normalization. The penalty is zero on the feasible subspace explored by the ideal XY circuit. This compares complete variants with different initial states and mixers; it does not isolate the effect of changing only the mixer.

Training uses exact ideal expectation values from `StatevectorEstimator`. The cost is shifted and scaled for optimization and restored for reporting. Final statevector probabilities and finite-shot frequencies are recorded separately. Reported circuit depths and two-qubit gate counts include initial-state preparation, use an all-to-all logical model, and do not include routing or noise from a particular IBM device.

Classical comparisons include complete enumeration, marginal-utility greedy selection, greedy followed by one-swap local improvement, uniform sampling of all states, and uniform sampling of feasible states. See [classical_baselines.csv](results/comparison/classical_baselines.csv) and the per-run random-reference fields. The random baselines have exact optimal probabilities of `1/64 = 1.5625%` and `1/20 = 5%` for a unique optimum. On this tiny instance, classical enumeration already solves the problem exactly.

At `w=1`, both greedy methods return B+D+F with utility 26.2, a regret of 0.2 relative to the exact A+E+F solution at 26.4. They reach an exact optimum at the other tested weights. These small-instance results do not establish general performance guarantees for any method.

Read the result columns together:

| Column | Interpretation |
|---|---|
| `feasible_probability` | Exact probability that one measurement satisfies the budget |
| `optimal_probability` | Exact probability of a globally optimal feasible allocation in one measurement |
| `feasible_rate`, `optimal_rate` | Observed frequencies in the 2,048 sampled measurements |
| `conditional_regret` | Expected utility loss relative to the optimum, conditional on a feasible result; read alongside feasible probability |
| `best_sampled_regret` | Utility loss of the best feasible sampled allocation; not the quality of a typical measurement |
| `optimizer_success` | The optimizer's stopping status; not a certificate of global optimality |

Finding an optimum somewhere in 2,048 samples is a weak criterion when only 20 allocations are feasible. The charts therefore expose per-measurement quality and variation across initialization seeds. Error bars show the minimum-to-maximum seed range, not confidence intervals. Equal optimizer evaluation limits do not imply equal circuit cost or runtime.

In the recorded comparison, all 40 XY runs have 100% ideal feasibility, but only 3 of the 8 XY weight/depth groups have a mean optimal probability above uniform feasible sampling's 5%. Their compiled circuits use 111 and 165 CX gates at depths 1 and 2, versus 30 and 60 for X. Across both variants, 48 of 80 runs did not meet the optimizer's convergence criterion within the evaluation budget. These are budget-limited outcomes, not evidence of quantum advantage. See [grouped_summary.csv](results/comparison/grouped_summary.csv).

The score-perturbation experiment re-solves 500 perturbed synthetic scenarios per weight classically to examine how sensitive the allocation is to its inputs. Every benefit score, priority score, and unordered overlap penalty receives an independent uniform multiplier between 0.9 and 1.1; zeros remain zero. The same 500 scenarios are reused across weights. [policy_stability.csv](results/comparison/policy_stability.csv) records how often the original optimum remains optimal and its regret under perturbed scores; [selection_stability.csv](results/comparison/selection_stability.csv) records each project's selection frequency, splitting credit across tied optima. This is a scenario analysis, not an empirical error model or evidence about real aid outcomes, and does not simulate quantum hardware noise.

The original allocation remains optimal in 70.6%, 31.2%, 42.2%, and 93.2% of scenarios for weights 0, 0.5, 1, and 2 respectively. In particular, the weight-0.5 choice is sensitive to these assumed score changes: defending the inputs deserves as much attention as solving the stated objective.

A separate [quota_policy.csv](results/comparison/quota_policy.csv) compares classical optima when a minimum number of D–F projects must be selected. This makes the distinction between a soft priority weight and a hard service requirement concrete. It reports utility loss relative to the same-weight optimum without the quota. The additional quota is a classical policy analysis; it is not an additional constraint implemented in the QAOA benchmark, and a project-count quota is still not a population-coverage guarantee.

For example, at `w=0`, requiring at least two projects from D–F changes the optimum from A+B+D to A+D+E, reducing net benefit from 22.1 to 20.5. The quota makes an explicit project-selection commitment; increasing a priority weight alone provides no such guarantee.

## Who sets the objective?

The humanitarian context is informed by the IFRC code's emphasis on needs and recipient participation, particularly principles 2 and 7. [IFRC Code of Conduct](https://www.ifrc.org/sites/default/files/2021-07/code-of-conduct-movement-ngos-english.pdf).

The project author chose every score. In a real process, affected communities, delivery organizations, and budget holders would need to establish the objective together, document score evidence, and provide a route to challenge exclusions. Human approval should remain responsible for actual allocations.

A weighted priority score can still disadvantage low-scoring groups, hide access barriers, or concentrate resources unevenly within a community. Missing factors include actual unmet need, transport time, project success, changing conditions, eligibility rules, disability access, and language barriers. Explicit minimum-service constraints might be useful, but require defensible group definitions and real demand data. Cloud cost, technical expertise, and access to compute also determine who can benefit from quantum tools.

Real data, unequal costs, larger instances, and noise or hardware studies are possible extensions. They are not prerequisites imposed by the current challenge instructions, and they should not be presented as completed evidence.

## Files and sources

| File | Purpose |
|---|---|
| `allocation.py` | Input validation, objective, QUBO/Ising conversion, exact reference |
| `mixers.py` | Feasible starting state and XY ring circuit |
| `run_experiments.py` | QAOA solver, original 24-run experiment, sampling, and run manifest |
| `benchmark.py` | Expanded 80-run comparison, classical baselines, score stability, and quota-policy analysis |
| `test_*.py` | Mathematical, encoding, input, and circuit checks; 34 tests passed in the local validation |
| `plots.py`, `plot_comparison.py` | Editable Matplotlib figures for the original experiment and expanded comparison, exported to PNG and SVG |
| `Track2_Submission.ipynb`, `Track2_Submission.html` | Primary submission notebook and browser presentation |
| `Fair_Aid_QAOA.ipynb`, `Fair_Aid_QAOA.html` | Preserved original 24-run explanation and browser presentation |
| `results/summary.csv`, `results/runs.json` | Original experiment's metrics, parameters, trajectories, state probabilities, and sampled counts |
| `results/comparison/summary.csv`, `results/comparison/runs.json` | Expanded comparison's complete run records |
| `results/comparison/manifest.json` | Comparison configuration, package versions, UTC execution time, and source/data hashes |
| `results/comparison/figures/` | Mixer-comparison and score-stability charts |
| `results/exact_solutions.csv`, `results/penalty_sweep.csv` | Original experiment's feasible allocations and penalty sensitivity |
| `results/manifest.json` | Original experiment's configuration and provenance |
| `execute_notebook.py` | Execute the selected notebook with the current Python and export HTML |
| `build_notebook.py` | Rebuild the original notebook structure, clearing previous execution outputs |
| `reference_manifest.json` | Downloaded references, source URLs, timestamps, and hashes |
| `submission/README.md` | Video outline and final submission checklist |

The [official repository](https://github.com/sandraalhames/Qompute-2026) supplies the task and rubric. The neighboring Max-Cut tutorial is credited to B. Tirado, BasQ Qiskit Fall Fest 2026, CC BY 4.0; this project's code is independently implemented. Method and API references are the [IBM QAOA tutorial](https://quantum.cloud.ibm.com/docs/tutorials/quantum-approximate-optimization-algorithm), [QAOAAnsatz](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.QAOAAnsatz), [StatevectorEstimator](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorEstimator), [StatevectorSampler](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.StatevectorSampler), and the [alternating operator ansatz paper](https://arxiv.org/abs/1709.03489). These sources do not supply or validate our synthetic priority scores.

Downloaded reference copies are stored under the repository's `downloads/track2/references/`; [reference_manifest.json](reference_manifest.json) in this project folder records their source URLs and SHA256 hashes. The download directory is excluded from Git; the public source links above remain available to reviewers. The IFRC reference supports the context, not the fabricated benefit or priority values.

## Remaining submission work

The computational materials, challenge mapping, and video outline are prepared. GitHub publication is being prepared; the video and event submission are still pending. Complete the team information, record and publish a 5–10 minute video, confirm reviewer access to the repository and video, and have one team member submit the official form. See the [submission guide](submission/README.md).

The official README gives the deadline as **October 14, 2026, 11:59PM PST**. Preserve that timezone label and confirm its intended meaning with the organizers; October 18 is the separate in-person check-in date. Real hardware and real aid data are optional extensions, not mandatory challenge requirements.
