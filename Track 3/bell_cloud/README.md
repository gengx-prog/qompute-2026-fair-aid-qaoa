# Bell correlations on IBM quantum hardware

This starter implements the Bell-state experiment in the Track 3 **Geometry-aware quantum cloud challenge** PDF. It prepares two entangled data qubits, measures XX, YY and ZZ, and compares an unprotected circuit with a circuit that detects odd ZZ parity using an auxiliary qubit.

Python and Qiskit construct and submit the quantum circuits. The `H`, `CX`, `Sdg` and measurement instructions are quantum operations. Each run exports **OpenQASM 3** files so the quantum instructions can be read independently of the Python controller.

This is an introductory experiment and optional real-QPU extension. The PDF's complete benchmark requires three specified processor models and nine core cases; the common graphs, numbered output ports and numerical noise definitions are not supplied in the local materials. This starter does not invent those definitions or claim to complete that benchmark.

## Run locally

Use PowerShell from the repository root. The separate environment below preserves the existing Track 2 environment. Setup is already completed in this workspace; on a fresh checkout run:

```powershell
py -3.13 -m venv downloads/track3/venv
& 'downloads/track3/venv/Scripts/python.exe' -m pip install --cache-dir downloads/track3/pip-cache -r 'Track 3/bell_cloud/requirements.txt'
```

Run the ideal and synthetic-noise experiments without any IBM account:

```powershell
& 'downloads/track3/venv/Scripts/python.exe' 'Track 3/bell_cloud/run_bell.py' local --shots 4096
```

The script prints its output directory. `--output PATH` can select a new directory; it must not already exist. Each variant has three basis settings, and `--shots` is the number of measurements **per setting**, not the total budget. Local mode runs six ideal circuits and six noisy circuits.

The synthetic model applies depolarization after `h`/`sdg` with parameter 0.002 and after `cx` with parameter 0.02, plus independent symmetric readout flips with probability 0.02. These are teaching assumptions, not IBM calibrations or official benchmark values. All noise parameters and seeds are recorded. To change them, use `--one-qubit-error`, `--two-qubit-error`, and `--readout-error`.

## Run on a real IBM QPU

1. Sign in to [IBM Quantum Platform](https://quantum.cloud.ibm.com/) and create or obtain access to a compute instance. Use the instance assigned by the organizers if one is provided.
2. Obtain the instance CRN and your API key as described in the [official account guide](https://quantum.cloud.ibm.com/docs/en/guides/initialize-account). Enter them locally when the script prompts. The API key is entered invisibly and is not saved by this script; do not put it in Git or chat.
3. Submit the six circuits:

```powershell
& 'downloads/track3/venv/Scripts/python.exe' 'Track 3/bell_cloud/run_bell.py' hardware --shots 2048 --submit
```

This selects an available real QPU in the specified instance, compiles the circuits into its native instructions, submits **6 x 2,048 = 12,288 shots**, records the job ID, then returns without waiting for the queue. The selected instance's plan and allocation apply. A completed real-QPU run is recorded below.

For the available ten-minute allocation, this starter limits each submission to **one job with six circuits**, at most 4,096 shots per circuit, and **120 seconds of Runtime execution time**. The default is 2,048 shots and a 120-second limit; `--max-execution-time` can lower the limit. The program rejects a planning estimate above the selected cap, records an attempt before submission, and never retries automatically. Inspect the platform before retrying a submission whose response was lost. Separate manual invocations can still spend more quota, so check remaining instance usage before each real experiment.

The limit applies to QPU/system execution, not queue time or total elapsed wall time. It controls consumption but cannot guarantee successful completion by a wall-clock deadline. A timeout may return no usable results. Fetch saves `hardware-usage.json` with actual provider metrics; usage can remain pending after measurement results are ready. See [IBM execution limits](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time) and [usage accounting](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time).

To choose an accessible device explicitly, add `--backend YOUR_BACKEND_NAME`. To choose the initial data/data/ancilla placement, add `--layout Q0 Q1 Q2` using actual integer IDs. With no layout specified, the script selects a connected three-qubit path and uses the same initial placement for both variants.

Omit `--submit` to authenticate, compile and inspect the ISA circuits without sending a QPU job. Running the command again with `--submit` creates a new compilation and output directory; it does not submit the previously saved QPY file. Specify the backend and layout explicitly when comparing runs.

An explicitly named saved Qiskit account can be used through `--account NAME`. Alternatively, the script reads `IBM_QUANTUM_API_KEY` and `IBM_QUANTUM_INSTANCE` from the local environment. Without these settings, it prompts interactively. No credential is included in the results.

When IBM reports the job as completed, retrieve it using the manifest path printed by the submission command:

```powershell
& 'downloads/track3/venv/Scripts/python.exe' 'Track 3/bell_cloud/run_bell.py' fetch --manifest 'PATH_TO_OUTPUT/manifest.json'
```

Fetch authenticates again but never submits another job. It returns promptly if the job is still queued or running. The results are saved as `hardware-results.json`.

The tested environment pins Runtime 0.50.0 and uses its current client-side `executor_sampler.Sampler`. It returns a single job ID that can be retrieved through `service.job`. The interface was validated locally with a fake backend before the real-account run below.

## Completed real-QPU experiment

After all 18 local checks passed, one job ran on **ibm_marrakesh**, using physical qubits 0, 1 and 2. Job **db852e6il99s73fpamcg** completed on 2026-10-15 UTC (2026-10-14 in Los Angeles). Evidence is saved in [`results/ibm-20261015/`](results/ibm-20261015/).

- Six circuits, 2,048 measurements each: **12,288 total**; no repeated submission.
- Server execution limit: **120 seconds**, also present in returned Runtime metadata.
- IBM's completed usage record: **5 seconds charged**, from a starting allowance of 600 seconds. The instance row subsequently showed **9 minutes 55 seconds remaining** and zero pending workloads.
- Server creation to completion: **9.44 seconds**; this differs from charged QPU usage and excludes local setup.

| Real-hardware measurement | Baseline | Parity check with postselection |
|---|---:|---:|
| Raw Bell fidelity estimate | 96.85% | 95.13% |
| Approximate 95% shot-noise interval | 95.50–97.80% | 93.51–96.35% |
| Retained measurements | 6,144 / 6,144 | 5,982 / 6,144 |
| Retention rate | 100% | 97.36% |
| Native two-qubit gates per circuit | 1 CZ | 6 CZ |

In plain language, the two data qubits produced strong Bell correlations, but adding the parity check did **not improve the measured fidelity in this run**. Extra gates and routing are possible contributors, not an isolated causal finding. The intervals describe finite sampling only; they do not remove readout or gate bias. These results are an optional hardware experiment, not completion of the PDF's nine-case benchmark.

The compiler changed the protected circuit's output mapping: `meas[0]`, `meas[1]`, `meas[2]` read physical wires 0, 2, 1. Counts still use classical-register order `meas[2] meas[1] meas[0]`, so the leftmost bit remains the auxiliary result. See `isa/resources.json` for all mappings, `hardware-results.json` for raw counts and correlations, and `hardware-usage.json` for finalized IBM usage.

The temporary API key used for this run was revoked after retrieval and its local download copies were removed. Existing results require no credential to inspect; future cloud retrieval or experiments require a valid key.

## Optimization campaign

### Confirmed results on 2026-10-15 UTC

The completed campaign found a repeatable improvement by selecting a better physical placement. All **176 connected data-qubit pairs** were covered in exploration, with **228 candidate observations including repeated controls**. The frozen candidate uses physical data qubits **7 and 6**, auxiliary wire **5**, and an extra reset before Bell preparation. Its two new confirmation jobs both passed the registered criterion. This is a supported choice among the tested configurations, not proof of a global optimum.

| Independent replication | Original layout `[0,1,2]` | Selected layout `[7,6,5]` + reset | Improvement | Same new layout without extra reset |
|---|---:|---:|---:|---:|
| A: `db8627eil99s73fpco30` | 95.8069% | 97.6349% | +1.8280 percentage points | 97.7875% |
| B: `db8627gr1c5c738lqgr0` | 90.5304% | 97.7325% | +7.2021 percentage points | 97.6501% |

Each candidate used **16,384 shots per basis**, with all shots retained. The conservative attempt-adjusted difference intervals versus the original were **[+0.8131, +2.8409]** and **[+5.9193, +8.4771] percentage points**, respectively. These are finite-shot intervals, not protection against hardware drift or systematic measurement error.

**The extra reset itself was not shown to help.** Against the ordinary Bell circuit on the same new layout, its changes were −0.1526 and +0.0824 percentage points, with both secondary intervals including zero. The benefit should be described as selecting a better-performing placement/combined configuration. The original placement also changed substantially between jobs, so the larger improvement in B is not a fixed gain that can be promised in future runs. An earlier audit found byte-identical original logical and ISA QASM across the drop seen during refinement; the cause of that performance change was not isolated.

The exact confirmed configuration, quantum instructions and source-indexed evidence are in [`best-confirmed/`](results/optimization-20261015/best-confirmed/). Start with [`best-config.json`](results/optimization-20261015/best-confirmed/best-config.json), or inspect the actual logical [XX](results/optimization-20261015/best-confirmed/confirm-01-a/logical/top0_dir0_baseline_reset_XX.qasm), [YY](results/optimization-20261015/best-confirmed/confirm-01-a/logical/top0_dir0_baseline_reset_YY.qasm), and [ZZ](results/optimization-20261015/best-confirmed/confirm-01-a/logical/top0_dir0_baseline_reset_ZZ.qasm) OpenQASM 3 files. The neighboring `isa/` folder contains the device-native versions; the spec fixes physical placement, whereas logical OpenQASM alone does not.

The separate controlled-X-error experiment also passed both replications:

| Fixed 10% injected X-error mixture | Ordinary circuit, all shots | Bridge detector, retained shots | Retention | Improvement, 99% bootstrap interval |
|---|---:|---:|---:|---:|
| `db86286r6oqs7388qudg` | 88.2813% | 97.2299% | 88.8672% | +8.9486 pp [8.3553, 9.5204] |
| `db8628eil99s73fpco50` | 88.1055% | 97.3183% | 88.8997% | +9.2129 pp [8.6156, 9.7835] |

This establishes filtering for the declared X-error mixture, with about **11.1% of shots discarded**. With no injected errors, the bridge gave 97.4744% and 97.5949%, against ordinary-circuit results of 97.9492% and 97.7973%; its descriptive difference intervals included zero. Therefore the controlled-noise result must not be presented as a general improvement under natural noise, arbitrary-error correction, or increased unconditional output yield. See [`controlled-noise-report.json`](results/optimization-20261015/controlled-noise-report.json).

Including the original starter run, **13 successful hardware jobs** executed **828 circuits and 1,155,072 shots**, charging **338 seconds (5 minutes 38 seconds)** in total. **262 seconds (4 minutes 22 seconds) remain** of the 600-second allowance; the IBM instance row independently showed these values and zero pending workloads. The final four jobs charged 41, 41, 10 and 10 seconds. Queue time is additional wall time. Full accounting, candidate observations and the coverage audit are in [`analysis-summary/`](results/optimization-20261015/analysis-summary/); failed local preparation attempts are also preserved and consumed no QPU time.

After all results and final usage were retrieved, the campaign's temporary API key was revoked and its local secret file removed. The recorded evidence can be read and reanalyzed without credentials. The cleanup receipt is `results/optimization-20261015/credential-cleanup.json`.

### Method and reproduction

The follow-up search is recorded in [`results/optimization-20261015/`](results/optimization-20261015/). Its total budget is **600 charged QPU seconds including the initial 5-second experiment**, rather than ten minutes of wall-clock time. Pending jobs reserve their full execution caps. Completed jobs release that reservation only after IBM reports finalized usage. Every submission requires successful local ideal-state validation and matching source/circuit hashes; an ambiguous submission is never automatically retried.

The exploratory search compares all **176 unordered connected data-qubit pairs** on the captured Marrakesh topology, using a prescribed orientation for each new pair. Repeated original and anchor controls expose changes between jobs. A follow-up compares the leading pairs in both directions, omits the unused auxiliary measurement (`baseline_quiet`), and tests an equivalent nearest-neighbor parity circuit (`bridge`). The bridge uses **4 native CZ gates**, compared with **6** for the originally routed parity circuit; ordinary Bell preparation uses **1**. Fewer gates alone do not establish better measured fidelity.

The refinement also tests an extra reset of both data qubits (`baseline_reset`) and a 200-microsecond wait before preparation (`baseline_relax_200us`). The compiled circuits retain these operations. Neither operation is assumed to help: the service already initializes qubits, and a wait can relax residual excitation or increase it toward the thermal population.

Screening maxima are exploratory. A candidate is frozen before **two new confirmation jobs**, with its original baseline in each job. A protected candidate also needs a same-layout unprotected control. A modified baseline includes that control as a secondary check, so a gain from selecting a better pair is not automatically attributed to reset, waiting, or removing auxiliary readout. The raw Bell estimate includes analysis and readout errors; it is not an isolated measurement of preparation fidelity.

The registered criterion requires both replications to show a point improvement of at least **0.005** and a positive lower endpoint of the conservative Bonferroni-Wilson difference interval, at alpha **0.05/3** per candidate attempt and at most three attempts. Before any candidate was frozen or confirmation submitted, an offline power review increased confirmation sampling from 8,192 to **16,384 shots per basis**; the timestamped amendment preserves the original success criterion. All unsuccessful attempts remain in the evidence.

The optimization tools use the same local authentication environment variables as `run_bell.py`:

```powershell
# Compile and validate the exact spec locally before submitting it.
& 'downloads/track3/venv/Scripts/python.exe' 'Track 3/bell_cloud/optimize_bell.py' prepare 'PATH_TO_SPEC.json'
& 'downloads/track3/venv/Scripts/python.exe' 'Track 3/bell_cloud/optimize_bell.py' submit 'NEW_STAGE_NAME'
# Fetch is read-only and does not create another job.
& 'downloads/track3/venv/Scripts/python.exe' 'Track 3/bell_cloud/optimize_bell.py' fetch 'NEW_STAGE_NAME'
# Inspect all available results, including incomplete campaigns, without credentials.
& 'downloads/track3/venv/Scripts/python.exe' 'Track 3/bell_cloud/summarize_optimization.py' --campaign 'Track 3/bell_cloud/results/optimization-20261015'
```

Use a fresh stage name when reproducing an experiment. The existing campaign folders are evidence and must not be overwritten or blindly resubmitted. `batch_optimization.py` can submit several independently validated stages together, reserving every job's full cap before the first submission; batching reduces repeated queue overhead without changing the total budget rule. `confirmation_optimization.py` freezes and evaluates the two holdouts without account access. `export_confirmed.py` packages OpenQASM and evidence only when both independent replications pass.

`native_twirl_candidate.py` is an additional locally tested fallback: it constructs equivalent fixed-seed native-CZ Pauli-twirled circuits, with matching preparation ensembles across XX/YY/ZZ. Its presence does not mean it was run on hardware. Consult actual job records for which methods were tested.

`stress_protocol.py` implements a separate, frozen controlled-noise experiment on physical layout `[6,7,8]`. Each of two jobs compares ordinary Bell preparation with the bridge detector, using nine no-fault circuit copies and one copy with an injected X error after preparation. The resulting fixed 9:1 mixture models a 10% X-error channel; raw counts are combined **before** postselection. Each job uses 60 circuits with 512 shots each. Both jobs must meet the declared improvement threshold with a positive 99% stratified-bootstrap lower bound. This tests conditional error filtering, not improvement under naturally occurring noise. Report the discarded fraction as well as retained-shot fidelity; higher conditional fidelity does not imply more correct outputs per attempted shot. The no-fault comparison shares samples with the mixture and is descriptive.

## What the quantum circuit does

Data qubits are `q[0]` and `q[1]`; `q[2]` is the auxiliary qubit. First apply `h q[0]` and `cx q[0], q[1]` to create Phi+ = (00 + 11)/sqrt(2). The protected circuit then applies `cx q[0], q[2]` and `cx q[1], q[2]`. Analysis rotations are:

| Measurement | Before ordinary Z readout on each data qubit |
|---|---|
| XX | H |
| YY | S-dagger, then H |
| ZZ | No rotation |

All three qubits are measured at the end; no dynamic circuits or mid-circuit feedback are needed. The deferred auxiliary measurement gives the same conditional statistics as measuring it just before the data rotations.

The baseline retains every shot. The protected result retains shots whose auxiliary measurement is zero. It detects odd ZZ parity, including a single bit-flip error before the check, but does not detect a phase-flip error. This is error detection with postselection, not full error correction. Extra gates and readout errors can outweigh its benefit.

Counts are displayed in `q2 q1 q0` order, so the leftmost bit is the auxiliary measurement. Conditional data parity gives the three correlations. Fidelity is:

```text
F(Phi+) = (1 + <XX> - <YY> + <ZZ>) / 4
```

The minus sign before YY matters. Seeing only 00 and 11 in the ZZ measurement is insufficient evidence of entanglement. Report fidelity, the three correlations, accepted and total shots, and circuit cost together. An unavailable basis after postselection produces an unavailable fidelity, never a fabricated zero or one.

Intervals combine three Bonferroni-adjusted Wilson bounds and represent approximate finite-shot uncertainty only. They exclude drift, gate/readout bias and other systematic errors. Raw fidelity estimates and intervals are not clipped to [0, 1]. On hardware, basis-change errors affect the raw estimate; this starter does not perform measurement mitigation or tomography corrections.

## Files and evidence

- `bell_core.py`: six logical circuits and count analysis.
- `run_bell.py`: local experiment, hardware compilation/submission, and result retrieval.
- `test_bell_core.py`: ideal-state, X/Z fault, classical-mixture, bit-order and postselection checks.
- `results/local-demo/`: executed ideal and synthetic-noise example, raw counts, resource records, manifests, drawings, QPY and OpenQASM.
- Each hardware output includes logical and ISA circuits, actual measured output wires, a compilation-time backend snapshot, a manifest and, after fetch, raw counts and Runtime metadata.

For resource accounting, `allocated_qubits` can equal the whole device width after transpilation. Use `used_physical_qubits` for the union of gate-active and measured wires. Both variants measure three qubits, including the otherwise idle auxiliary in the baseline. The protected logical circuit adds two CNOTs; device routing can add more operations. A zero count of explicit SWAP instructions in native output does not mean zero routing overhead.

The hardware compiler fixes the initial placement but may move data during routing. Actual measured physical output wires are recorded. Enforcing the PDF's **fixed output ports**, constructing its three common graphs, isolating routing overhead, and preparing the full nine-case notebook remain separate benchmark work after the organizer specifications are obtained. Real-QPU results must remain separate from the common benchmark (PDF page 9).

## Validation

```powershell
& 'downloads/track3/venv/Scripts/python.exe' -m unittest discover -s 'Track 3/bell_cloud' -p 'test_*.py' -v
```

The local suite passed **76 tests**, followed by **8 additional export-evidence checks**, recorded in `results/optimization-20261015/local-suite-validation.json`. The original starter has eight physics checks and ten budget/interface checks covering the state, postselection, strict work limits, cap propagation and single submission behavior. Optimization checks additionally cover the full bridge unitary, quiet-readout bit order, independent confirmation, pending-budget reservations, batch submission failures, initialization variants, X/Z fault validation, controlled-noise aggregation, and native-twirl equivalence. Export checks bind the packaged evidence to the frozen plan, exact job, raw records, submitted QPY hash and corresponding OpenQASM. The six initial hardware ISA circuits were also compiled and executed through Runtime's local fake-backend mode with network access blocked, exported/imported as OpenQASM 3, and saved/reloaded as QPY. These checks do not establish that a real QPU job has run.

References: the neighboring challenge PDF, [IBM hardware workflow](https://quantum.cloud.ibm.com/docs/en/guides/hello-world), [Sampler guide](https://quantum.cloud.ibm.com/docs/en/guides/get-started-with-sampler), and [OpenQASM export](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qasm3).
