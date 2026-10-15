"""Reanalyze saved QPU counts and build a portable, executed submission.

Offline only: no IBM account access, cloud calls, or QPU submissions.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
from nbconvert import HTMLExporter
import numpy as np

from bell_core import analyze_counts, build_circuits
from optimization_analysis import analyze_records, compare_candidates, confirm_candidate, confirm_replicates
from stress_protocol import confirm_jobs
from summarize_optimization import build_summary

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "results" / "optimization-20261015"
COLORS = ["#64748b", "#148a83", "#d48d26"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_report():
    plan = read(CAMPAIGN / "confirmation-plan-01.json")
    confirmations = []
    input_files = [CAMPAIGN / "confirmation-plan-01.json"]
    for stage in plan["stage_names"]:
        path = CAMPAIGN / stage / "hardware-results.json"
        input_files.append(path)
        result = read(path)
        summaries = analyze_records(result["records"])
        test = confirm_candidate(
            summaries, plan["candidate_id"], plan["original_baseline_id"],
            screening_job_ids=plan["excluded_confirmation_job_ids"],
            matching_baseline_id=plan["matching_baseline_id"],
            alpha=plan["alpha_per_attempt"], minimum_improvement=plan["minimum_improvement"],
        )
        test.update(stage=stage, summaries=summaries)
        test["attribution_comparison"] = compare_candidates(
            summaries[plan["candidate_id"]], summaries[plan["attribution_baseline_id"]],
            alpha=plan["alpha_per_attempt"], comparison_count=2,
        )
        confirmations.append(test)
    assert confirm_replicates(confirmations)["confirmed_improvement"]

    stress_plan = read(CAMPAIGN / "controlled-noise-plan.json")
    input_files.append(CAMPAIGN / "controlled-noise-plan.json")
    stress_inputs = []
    for stage in stress_plan["stage_names"]:
        paths = [CAMPAIGN / stage / filename for filename in
                 ("spec.json", "hardware-results.json", "manifest.json")]
        input_files.extend(paths)
        stress_inputs.append(tuple(read(path) for path in paths))
    stress = confirm_jobs(stress_plan, stress_inputs)
    assert stress["confirmed"]
    campaign_summary, _, _ = build_summary(CAMPAIGN)
    assert campaign_summary["coverage"]["complete"]
    assert not campaign_summary["warnings"]

    local = {}
    circuits = build_circuits()
    for label in ("ideal", "synthetic_noisy"):
        path = HERE / "results" / "local-demo" / f"{label}.json"
        input_files.append(path)
        local[label] = analyze_counts(circuits, read(path)["raw_counts"])
    starter_path = HERE / "results" / "ibm-20261015" / "hardware-results.json"
    input_files.append(starter_path)
    starter = analyze_counts(circuits, read(starter_path)["raw_counts"])
    hashes = {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in input_files}
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "Offline reanalysis of recorded counts, not a new QPU run.",
        "packages": {name: version(name) for name in
                     ("qiskit", "numpy", "matplotlib", "nbformat", "nbconvert")},
        "confirmation": confirmations, "controlled_noise": stress,
        "local": local, "starter_hardware": starter,
        "coverage": campaign_summary["coverage"], "usage": campaign_summary["usage"],
        "input_sha256": hashes,
        "scope": "Qompute 2026 general Track 3. The optional official three-model geometry benchmark is incomplete.",
        "bootstrap_note": "Recomputed 5,000-draw percentile intervals can vary slightly across numerical-library builds/platforms. Raw counts, point estimates and the pass rule are unchanged.",
    }


def figures(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.titleweight": "bold"})
    files = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            path = output / f"{name}.{ext}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
        plt.close(fig)
        files.append(name)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    keys = ["original_baseline", "top0_dir0_baseline", "top0_dir0_baseline_reset"]
    labels = ["Original [0,1,2]", "Selected [7,6,5]", "Selected + reset"]
    for index, (key, label) in enumerate(zip(keys, labels)):
        rows = [row["summaries"][key] for row in report["confirmation"]]
        means = np.array([row["fidelity"] for row in rows]) * 100
        intervals = np.array([row["fidelity_interval"] for row in rows]) * 100
        x = np.arange(2) + (index - 1) * .24
        ax.bar(x, means, width=.23, color=COLORS[index], label=label)
        ax.errorbar(x, means, yerr=[means - intervals[:, 0], intervals[:, 1] - means],
                    fmt="none", color="#172033", capsize=4)
        for xx, value in zip(x, means):
            ax.text(xx, value + 1.7, f"{value:.2f}", ha="center", fontsize=11)
    ax.set(ylim=(0, 108), ylabel="Raw Bell fidelity (%)", xticks=[0, 1],
           xticklabels=["Confirmation A", "Confirmation B"],
           title="Selected placement improves both new confirmation runs")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=3, frameon=False, fontsize=11)
    fig.text(.07, -.13, "All shots retained. Error bars: approximate 95% finite-shot intervals.\nReset-specific benefit is unproven; hardware drift is outside these intervals.", fontsize=10)
    save(fig, "placement_confirmation")

    fig, (ax, ay) = plt.subplots(1, 2, figsize=(12, 5.4))
    group_labels, rows = [], []
    for index, job in enumerate(report["controlled_noise"]["jobs"]):
        for scenario in ("p0", "p10"):
            group_labels.append(f"{'AB'[index]} / {'0%' if scenario == 'p0' else '10%'} X")
            rows.append(job["scenarios"][scenario])
    for vi, variant in enumerate(("baseline", "bridge")):
        x = np.arange(4) + (vi - .5) * .34
        values = np.array([row[variant]["fidelity"] for row in rows]) * 100
        ci = np.array([row[variant]["fidelity_interval_99"] for row in rows]) * 100
        label = "Ordinary / all shots" if vi == 0 else "Detector / retained shots"
        ax.bar(x, values, width=.33, color=COLORS[vi], label=label)
        ax.errorbar(x, values, yerr=[values-ci[:, 0], ci[:, 1]-values], fmt="none", color="#172033", capsize=3)
    ax.set(ylim=(0, 105), xticks=range(4), xticklabels=group_labels,
           ylabel="Raw Bell fidelity (%)", title="Filtering helps the declared X-error mixture")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.15), frameon=False, fontsize=10)
    retention = [row["bridge"]["retention"] * 100 for row in rows]
    ay.bar(range(4), retention, color=COLORS[1])
    for x, value in enumerate(retention):
        ay.text(x, value + 1.5, f"{value:.2f}%", ha="center", fontsize=11)
    ay.set(ylim=(0, 108), xticks=range(4), xticklabels=group_labels,
           ylabel="Accepted attempted shots (%)", title="Higher conditional quality has a cost")
    fig.text(.5, -.19, "Error bars: 99% stratified bootstrap intervals. 0% is descriptive and shares samples with 10%.", ha="center", fontsize=10)
    save(fig, "controlled_noise_and_yield")

    fig, (ax, ay) = plt.subplots(1, 2, figsize=(11, 5.2))
    sources = [HERE/"results/ibm-20261015/isa/resources.json",
               HERE/"results/ibm-20261015/isa/resources.json",
               CAMPAIGN/"stress-01/isa/resources.json"]
    prefixes = ["baseline_", "protected_", "stress_bridge_nofault_"]
    counts = [sorted({r["two_qubit_gates"] for r in read(path) if r["name"].startswith(prefix)})
              for path, prefix in zip(sources, prefixes)]
    assert all(len(row) == 1 for row in counts)
    ax.bar(["Ordinary", "Routed check", "Bridge check"], [row[0] for row in counts], color=COLORS)
    ax.set(ylabel="Native two-qubit CZ gates per circuit", ylim=(0, 7), title="Detection adds hardware operations")
    for i, row in enumerate(counts): ax.text(i, row[0]+.15, str(row[0]), ha="center")
    usage = report["usage"]
    ay.barh([0], [usage["total_used_seconds"]], color=COLORS[1], label="Charged")
    ay.barh([0], [usage["remaining_seconds"]], left=[usage["total_used_seconds"]], color="#dce6e8", label="Remaining at recorded closeout")
    ay.set(xlim=(0, 600), yticks=[], xlabel="Recorded charged QPU seconds", title="338 of 600 allocated seconds used")
    ay.legend(loc="lower center", bbox_to_anchor=(.5, -.36), frameon=False, fontsize=10)
    ay.text(169, 0, "338 s", ha="center", va="center", color="white", fontsize=17)
    ay.text(469, 0, "262 s", ha="center", va="center", fontsize=17)
    save(fig, "hardware_cost")
    return files


def build_notebook(output):
    """Execute the notebook's Python cells directly, saving real cell outputs.

    A single Python namespace and stdout capture avoid requiring a running
    Jupyter server. The saved notebook also runs normally in Jupyter.
    """
    output = Path(output)
    cells = []
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md("""# Can we trust a noisy quantum computer?
## Bell correlations, placement and error detection on IBM hardware

**Qompute 2026, Track 3 / team-12**<br>
HUNG-YU LIN, XINCHEN GENG, Daniel Cai

HUNG-YU LIN: acceptance review and video. XINCHEN GENG: experiments and method development. Daniel Cai: report/presentation.

This executed notebook reanalyzes saved raw counts. It never submits cloud jobs.
The project follows the general Track 3 requirements. It does **not** claim completion of the optional three-architecture geometry benchmark.

**Question:** When do placement and parity detection improve a result, and what do they cost?
""")
    code("""from pathlib import Path
import sys
root = Path.cwd().resolve()
for candidate in [root, root.parent, root / 'Track 3' / 'bell_cloud']:
    if (candidate / 'build_submission.py').is_file():
        root = candidate
        break
else:
    raise FileNotFoundError('Open from repository root, bell_cloud, or its submission directory.')
sys.path.insert(0, str(root))
from build_submission import compute_report, figures
report = compute_report()
print(report['execution'])
print('Both frozen placement replications pass:', all(r['confirmed_improvement'] for r in report['confirmation']))
print('Both controlled-X replications pass:', report['controlled_noise']['confirmed'])
print('Input files hashed:', len(report['input_sha256']))""")
    md(r"""## Verification and baselines

The target is $|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2$.
Measure three settings, with
$$F=(1+\langle XX\rangle-\langle YY\rangle+\langle ZZ\rangle)/4.$$
The YY minus sign is essential. A classical 00/11 mixture can pass a ZZ-only check.
The ancilla detects odd ZZ parity. A single X fault before the check can be detected; a Z fault cannot. Postselection keeps ancilla-zero shots. This does not implement full error correction.

Saved ideal and synthetic noisy simulator counts provide the baselines below. Synthetic noise uses depolarizing parameters 0.002 / 0.02 on the declared one-/two-qubit gates and readout flip probability 0.02. These are teaching assumptions, not calibrated device noise.
""")
    code("""for experiment, variants in report['local'].items():
    for variant, row in variants.items():
        print(f\"{experiment:18} {variant:10} F={row['fidelity']:.6f}, retained={row['acceptance_fraction']:.3%}\")
for variant, row in report['starter_hardware'].items():
    print(f\"Initial QPU {variant:10} F={row['fidelity']:.6f}, retained={row['acceptance_fraction']:.3%}\")""")
    md("""## Placement: screening followed by two new confirmations

The search covered 176 unordered connected pairs, one prescribed initial orientation each. The candidate was frozen before two new jobs. Each setting has 16,384 shots. Both replications must improve by at least 0.005 and have a positive conservative, attempt-adjusted difference bound. This controls selection from screening more carefully than reporting its maximum.

Compare the selected layout plus reset with the **same layout without reset** to interpret the added operation. No reset-specific improvement was established. Different original-baseline performance across jobs limits any promise of a fixed future gain.
""")
    code("""for row in report['confirmation']:
    print(row['stage'])
    for name, result in row['summaries'].items():
        print(f\"  {name}: F={result['fidelity']:.6%}; retained={result['retention']:.2%}\")
    primary = row['comparisons']['original_baseline']
    print('  Improvement (percentage points):', round(100*primary['improvement'], 4))
    print('  Adjusted difference interval (pp):', [round(100*v, 4) for v in primary['improvement_interval']])
    print('  Reset-specific difference interval (pp):', [round(100*v, 4) for v in row['attribution_comparison']['improvement_interval']])""")
    code("""image_names = figures(report, root / 'submission' / 'figures')
from IPython.display import display, Image
display(Image(filename=str(root / 'submission' / 'figures' / 'placement_confirmation.png')))""")
    md("""## Controlled X-error experiment and yield

Each of two new jobs uses nine no-fault circuit copies and one X-fault copy for each variant and basis, with 512 shots per circuit. Counts combine in a fixed 9:1 allocation before filtering. This models a declared 10% X-error mixture, rather than random per-shot Bernoulli fault allocation.

Both jobs must pass the frozen improvement criterion with a positive 99% stratified-bootstrap lower bound. The 0% comparison shares samples and is descriptive. Confidence intervals cover finite-shot variation, excluding drift and systematic readout bias.
""")
    code("""for job in report['controlled_noise']['jobs']:
    print(job['stage'], job['job_id'])
    for name, scenario in job['scenarios'].items():
        baseline, bridge = scenario['baseline'], scenario['bridge']
        print(f\"  {name}: ordinary={baseline['fidelity']:.4%}, filtered={bridge['fidelity']:.4%}, retained={bridge['retention']:.4%}\")
    print('  99% difference interval (pp):', [round(100*v, 4) for v in job['primary']['confidence_interval_99']])
display(Image(filename=str(root / 'submission' / 'figures' / 'controlled_noise_and_yield.png')))""")
    md("""## Cost and practical interpretation

Ordinary Bell preparation uses one native CZ gate; the initial routed detector uses six; the bridge detector uses four in the measured layouts. The bridge preserves the complete detector unitary while changing connectivity requirements. Fewer gates alone do not prove better noisy performance.

The recorded campaign used 13 completed hardware jobs, 828 circuits and 1,155,072 shots, charging 338 seconds. The recorded remaining allowance was 262 seconds, not a live account balance.
""")
    code("""print('Coverage:', report['coverage']['observed_planned_pairs'], '/', report['coverage']['planned_unique_pairs'])
print('Final recorded charge:', report['usage']['total_used_seconds'], 'seconds')
print('Recorded remaining:', report['usage']['remaining_seconds'], 'seconds')
display(Image(filename=str(root / 'submission' / 'figures' / 'hardware_cost.png')))""")
    md("""## Recommendation and societal impact

Cloud platforms should expose calibrated placement choices with repeated same-job reference controls and uncertainty. Users with small compute budgets need raw-count exports, explicit resource costs, and failure results to compare methods fairly.

Use the observed placement benefit as evidence among tested configurations. Use the injected-error experiment as evidence for conditional X-error filtering. Neither result establishes quantum advantage, general fault tolerance, or readiness for high-stakes decisions. Report accepted shots alongside fidelity so discarded outputs do not disappear from the assessment.

## Scope and remaining architecture work

The optional geometry PDF asks for three specified models, fixed output ports, common numerical noise assumptions and nine core cases. Those complete definitions are absent from the supplied materials. This submission does not substitute invented definitions or claim architecture superiority. The present scope is a verified small experiment with ideal/noisy references, suppression through placement, error detection, resource accounting and limitations under the general Track 3 prompt.

## Sources and reproduction

- [Official Track 3 requirements](https://github.com/sandraalhames/Qompute-2026#track-3-can-we-trust-a-noisy-quantum-computer)
- [Project code and raw evidence](https://github.com/gengx-prog/qompute-2026-fair-aid-qaoa/tree/main/Track%203/bell_cloud)
- [IBM error mitigation and suppression guide](https://quantum.cloud.ibm.com/docs/en/guides/error-mitigation-and-suppression-techniques)
- The neighboring geometry challenge PDF is credited to Steven Rayan / quanTA, University of Saskatchewan.

Run `python build_submission.py` from `Track 3/bell_cloud` after installing `requirements-submission.txt`. A single Python namespace executes all code cells and records their real outputs. The notebook also runs in Jupyter. This rebuilds local analyses and figures only. Saved QPU evidence and frozen plans stay unchanged. Bootstrap percentile endpoints may vary slightly across numerical builds and platforms; point estimates and conclusions should agree.
""")
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
                             language_info={"name": "python", "version": sys.version.split()[0]})
    namespace = {"__name__": "__main__"}
    import os
    previous = Path.cwd()
    os.chdir(HERE)
    try:
        number = 0
        for cell in notebook.cells:
            if cell.cell_type != "code": continue
            number += 1
            cell.execution_count = number
            capture = StringIO()
            outputs = []
            def capture_display(*objects, **kwargs):
                for obj in objects:
                    if hasattr(obj, "_repr_png_"):
                        data = obj._repr_png_()
                        if isinstance(data, tuple): data = data[0]
                        if isinstance(data, bytes): data = base64.b64encode(data).decode()
                        outputs.append(nbf.v4.new_output("display_data", data={"image/png": data}))
                    else:
                        outputs.append(nbf.v4.new_output("display_data", data={"text/plain": str(obj)}))
            # Imports of display use this wrapper only while executing a cell.
            import IPython.display
            original_display = IPython.display.display
            IPython.display.display = capture_display
            try:
                with redirect_stdout(capture): exec(compile(cell.source, "Track3_Submission.ipynb", "exec"), namespace)
            finally:
                IPython.display.display = original_display
            if capture.getvalue(): outputs.insert(0, nbf.v4.new_output("stream", name="stdout", text=capture.getvalue()))
            cell.outputs = outputs
    finally:
        os.chdir(previous)
    nbf.validate(notebook)
    nbf.write(notebook, output / "Track3_Submission.ipynb")
    exporter = HTMLExporter()
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(notebook)
    (output / "Track3_Submission.html").write_text(html, encoding="utf-8")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "submission")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = compute_report()
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    cells = build_notebook(args.output)
    print(json.dumps({"output": str(args.output), "executed_code_cells": cells,
                      "placement_confirmed": True, "controlled_noise_confirmed": True}))


if __name__ == "__main__":
    main()
