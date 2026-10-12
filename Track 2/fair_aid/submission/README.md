# Track 2 Submission and Video Preparation

Fair Aid QAOA selects three of six equal-cost aid projects, examines how a policy priority weight changes the allocation, and compares Qiskit QAOA with classical methods. The [project guide](../README.md) contains the implementation and results. Use the [submission notebook](../Track2_Submission.ipynb) and [HTML presentation](../Track2_Submission.html) for the presentation.

**GitHub publication is being prepared. The video and event submission are still pending.** Every input is a synthetic teaching score; do not present the results as real relief data or validated social benefits.

## Official requirements

Source: the [official Qompute 2026 repository](https://github.com/sandraalhames/Qompute-2026), checked on 2026-10-12 against [README commit 855f3b7](https://github.com/sandraalhames/Qompute-2026/blob/855f3b7b7849b30cd63a07b730df97096fc8e317/README.md). Later organizer announcements take precedence.

| Official requirement | Project evidence | Final presentation check |
|---|---|---|
| A concrete allocation scenario and the people affected | Established services A-C and services for underserved communities D-F, explicitly presented as fictional examples | Explain who is affected before introducing the algorithm |
| A cost Hamiltonian with a constraint penalty | `C(x) = -U_w(x) + A(sum(x_i)-3)^2`; QUBO/Ising conversion and exhaustive energy validation | Show the budget constraint, penalty strength, and a feasible allocation |
| Qiskit implementation checked against brute force or a classical solution | X and XY approaches; all 64 states enumerated; greedy, local-improvement, and random baselines | Use the complete experiment outputs and retain every initialization |
| Reflection on weights, penalties, and omissions | Weight sweep, score perturbations, classical minimum-project quota analysis, and missing-demand discussion | Explain who should choose the weights and when a human must decide |
| A 5-10 minute video and the team's GitHub repository | Seven-minute outline below; English README, notebooks, HTML, and reproducible outputs | Publish accessible repository and video links, then submit the form |

The official deadline is written as **"11:59PM PST, 14th Oct (Wednesday)"**, referring to October 14, 2026. Preserve the published timezone label and verify its intended meaning with the organizers; do not silently replace it with local time or assume an unconfirmed conversion. Submit through the [official form](https://docs.google.com/forms/d/e/1FAIpQLSfafnjAv0nPT38lwS_pa4oWEN3R26bjPpya_I5AeTX5MIz09A/viewform?usp=dialog). **Only one team member should submit once.**

In-person participants must also check in on **October 18**, a separate requirement from the October 14 submission deadline. A member who cannot attend should contact both the organizers and teammates; the official contact is `qee@usc.edu`. Real hardware is optional, and the instructions do not require real aid data.

## Seven-minute presentation outline

These are presenter notes that can be divided among team members. Use the final `Track2_Submission.ipynb`, `results/comparison/`, and its manifest for all reported numbers; do not select only the best-performing seed. The original 24-run X-mixer experiment remains in `Fair_Aid_QAOA.ipynb` and `results/`. The expanded 80-run comparison uses different evaluation limits and more seeds, so use it for the X/XY comparison.

| Time | On screen | Presenter notes |
|---|---|---|
| 0:00-0:40 | Title and notebook problem statement | Introduce the limited budget: only three projects can be funded. Ask who defines the optimum and how the allocation changes when underserved communities receive more priority. State immediately that these are synthetic teaching inputs and simulator experiments. |
| 0:40-1:25 | Six-project input table | Explain cost, benefit, priority, and service overlap, including the A-C and D-F scenario groups. Scores are not people counts, priority is not a fairness guarantee, and equal costs and full budget use are assumptions. |
| 1:25-2:20 | Objective, budget penalty, and Hamiltonian-validation table | Distinguish policy weight `w` from penalty `A`. Show binary variables mapped to qubits. All 64 states are checked, of which 20 select exactly three projects. `A=20` guarantees feasibility of the exact global minimum, not of every shallow-QAOA measurement. |
| 2:20-3:15 | Initial-state and mixer explanations in the notebook; `mixers.py` | X explores all bitstrings; XY begins in a uniform superposition of the 20 feasible states. Complete exchange pairs preserve the selected-project count. The initial state uses only the budget structure, not the known optimum. Count generic state-preparation costs. |
| 3:15-4:30 | `results/comparison/figures/mixer_comparison.png`; `classical_baselines.csv` | Explain four weights, two depths, five seeds, and two approaches: 80 runs, up to 300 objective evaluations and 2,048 final shots each. Read feasibility, optimal probability, conditional regret, and resources together. Error bars show the seed range. Point to actual outcomes; finding one optimal sample is not evidence of quantum advantage. |
| 4:30-5:25 | `results/figures/allocation_tradeoff.png`; `results/comparison/figures/score_stability.png`; `quota_policy.csv` | Raising the weight from 0 to 2 changes the optimum from A+B+D to D+E+F. Net benefit drops from 22.1 to 16.7 while priority rises from 2.5 to 9. These reflect a changed synthetic objective, not measured social outcomes. Perturbations test input sensitivity; the additional quota is a classical policy analysis, is not enforced in QAOA, and is not population coverage. |
| 5:25-6:20 | Societal-impact and limitations discussion | Affected communities, delivery organizations, and budget holders should help choose the objective. Missing factors include transport, disability and language access, real demand, and differences within groups. Explain documented evidence, appeals, and human approval. Compute cost and expertise also affect who benefits. |
| 6:20-7:00 | README, file structure, and reproduction commands | Show the entry points and complete outputs. Real data, unequal costs, larger instances, and noise studies are extensions. Close with supported findings: objective choices change allocations, encoding affects feasibility, classical enumeration suffices for this instance, and quantum advantage has not been demonstrated. |

Rehearse with `Track2_Submission.html` and check that formulas, legends, labels, and numbers remain readable at the recording resolution. The submission notebook normally validates the saved 80-run snapshot and source hashes without re-optimizing. Set `RERUN_EXPERIMENTS=True` to recompute; it also runs when the result snapshot is missing. Distinguish validation of saved results from a new computation during the demonstration.

The humanitarian context draws on the [IFRC Code of Conduct](https://www.ifrc.org/sites/default/files/2021-07/code-of-conduct-movement-ngos-english.pdf), particularly principles 2 and 7 on needs and recipient participation. This source supplies neither the synthetic scores nor validation of the model's fairness. Reference files are under the repository's `downloads/track2/references/`; [reference_manifest.json](../reference_manifest.json) records their URLs and SHA256 hashes.

## Evidence for the judging criteria

| Criterion | Weight | Evidence to make visible |
|---|---:|---|
| Technical Content | 35% | Explicit constraint and Hamiltonian; encoding and mixer checks; Qiskit runs; complete classical comparisons; feasibility and solution quality; initial-state and circuit costs |
| Societal Impacts | 20% | Affected people, responsibility for weights, proxy limitations, possible exclusions, appeals, human accountability, and access to quantum tools |
| Presentation | 20% | A coherent problem-model-experiment-policy narrative, readable figures, and a 5-10 minute recording |
| Originality | 20% | A constrained aid model beyond the Max-Cut starter, two quantum variants, score sensitivity, policy comparisons, and clear attribution |
| Clean files and attendance/communication | 5% | Clear GitHub entry point, dependencies and reproduction steps, sources and complete outputs; required attendance and communication for in-person members |

## Final checklist

### Local deliverables

- [x] The 34 local mathematical, encoding, input, circuit, and analysis tests passed.
- [x] The existing Python 3.13.7 environment executed all 16 code cells across the two notebooks without errors; HTML formulas and charts were checked in a browser.
- [ ] In a fresh environment, follow the [reproduction instructions](../README.md#reproduce-from-a-fresh-checkout) to install, test, and execute the submission notebook. Set `RERUN_EXPERIMENTS=True` for a full recomputation.
- [x] The comparison manifest matches its source and inputs, its `summary.csv` retains all 80 runs including unsuccessful optimizations, and the original 24 rows remain separate.
- [x] The submission notebook has no error outputs, its HTML opens, its charts agree with the CSVs, and relative documentation links were checked.
- [ ] Publish code, input CSVs, tests, dependency files, executed notebooks, HTML, results, and figures. Exclude `.venv/`, `downloads/`, caches, and temporary experiment directories.
- [x] The event instructions and starter tutorial retain their attribution, and `Track 2/README.md` links to this project.

### Team and external steps

- [ ] Fill in the team name, members, and responsibilities; confirm who will attend in person.
- [ ] Publish the team's GitHub repository and check reviewer access in a signed-out browser.
- [ ] Record a 5-10 minute video using the outline, publish it, and check playback permissions.
- [ ] Confirm the intended `PST` deadline label and appoint one member to submit the official form on time.
- [ ] Save the submission confirmation and avoid duplicate submissions. In-person participants must check in on October 18.

| Information to complete | Status |
|---|---|
| Team name, members, and responsibilities | To be supplied by the team |
| Team GitHub repository | Publication in progress; add the verified URL after publication |
| 5-10 minute video | To be recorded and published |

The current instructions do not require paid compute or a hardware experiment. Prioritize verified results, clear explanations, and accessible submission materials. Real demand data, unequal project costs, larger instances, or noise experiments can follow as explicitly scoped extensions.
