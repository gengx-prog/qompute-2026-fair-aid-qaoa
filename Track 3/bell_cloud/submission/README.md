# Track 3 submission

## Start here

- [Executed results notebook](Track3_Submission.ipynb)
- [Standalone HTML results](Track3_Submission.html), including embedded figures
- [Captioned English presentation video](https://github.com/gengx-prog/qompute-2026-fair-aid-qaoa/releases/download/track3-submission-2026-10-14/Track3_Presentation.mp4)
- [English captions](Track3_Presentation.srt) and [editable narration](video_script.json)
- [Recomputed numerical summary and input hashes](summary.json)
- [Methods and original evidence](../README.md)

The HTML is a local/browser presentation. GitHub displays its source, so download it before opening. The notebook's saved outputs are also readable on GitHub.

## Team information supplied by the team

| Member | Responsibility |
|---|---|
| HUNG-YU LIN | Acceptance review and video |
| XINCHEN GENG | Experiments and method development |
| Daniel Cai | Report/presentation |

Team name: **team-12**. Project title: **Can we trust a noisy quantum computer? Bell correlations, placement and error detection.** The team confirmed that no member has submitted yet.

## Scientific scope

This submission follows the general Qompute 2026 Track 3 prompt: a known-answer experiment, ideal/noisy baselines, a suppression/detection strategy, and an improvement-versus-cost assessment. It includes real IBM QPU evidence.

The optional geometry PDF's full benchmark remains incomplete. Its three common processor definitions, numbered fixed output ports and numerical noise assumptions were not supplied in the available materials. We do not invent those specifications or claim completion of its nine cases. If that subchallenge is required for the chosen judging category, obtain the definitions and complete it before making that claim.

The main measured benefit is from the selected physical configuration. Extra reset alone was not shown to help. The bridge detector improves retained-shot fidelity for the declared fixed 10% X-error mixture, with about 11.1% discarded shots. It did not establish an improvement under natural noise, arbitrary-error correction, or increased unconditional yield. Intervals exclude calibration drift and systematic measurement bias.

## Reproduce without spending QPU quota

From `Track 3/bell_cloud`, on macOS/Linux:

```sh
python3 -m venv .venv-submission
.venv-submission/bin/python -m pip install -r requirements-submission.txt
.venv-submission/bin/python -m unittest discover -s . -p 'test_*.py' -v
.venv-submission/bin/python build_submission.py
```

On Windows PowerShell, replace `.venv-submission/bin/python` with `.venv-submission/Scripts/python.exe` and use the intended installed Python version. The scripts resolve the evidence from their own location. No IBM key is needed for saved-count analysis.

The builder executes six notebook code cells in a single Python namespace and records actual outputs and embedded images. The notebook also runs in Jupyter. The saved original QPU records and frozen plans remain unchanged. Its 5,000-draw bootstrap percentile endpoints may differ slightly between numerical builds/platforms, while point estimates and conclusions should agree.

To verify portable export, use a **new** output directory:

```sh
python export_confirmed.py results/optimization-20261015/confirmation-attempt-01.json NEW_EXPORT_DIRECTORY
```

Historical Windows paths now resolve by the plan filename beside the report; the frozen canonical plan hash is still required. New confirmation reports store a relative plan filename.

Video rebuild requires macOS `say` with Samantha, `ffmpeg`, `ffprobe`, and Pillow:

```sh
python build_video.py
```

The narration uses the system's synthetic voice, with no imitation of any team member. `video-manifest.json` records duration, scene timing and file hashes. This command creates local files only.

## Submission entry and external checklist

Use the [submission form confirmed by the team](https://docs.google.com/forms/d/e/1FAIpQLSffK6jWlW7L8uXGhQ_336lzX1qnckdqiC-M593KWLag3hKDBw/viewform).

The older form ending `...fafnjAv0nPT38lwS_pa4oWEN3R26bjPpya_I5AeTX5MIz09A` displayed "no longer accepting responses" when inspected on October 14, 2026. After Google reauthentication, the form required email, team name, member names, a GitHub link, a code ZIP and a compressed video. Both required files were uploaded and one response was submitted. The form displayed: "Your submission has been received and a copy has been sent to you. Good luck and see you Sunday!"

The official repository specifies a 5-10 minute video and the team GitHub link, submitted once by one team member. It prints the deadline as **11:59PM PST, October 14, 2026**. Preserve this published label rather than assuming a timezone correction.

- [x] Record supplied member names and known roles.
- [x] Reanalyze hardware counts and create executed notebook, HTML and figures.
- [x] Fix relocatable export and pass its regression checks.
- [x] Confirm team name and Daniel Cai's role.
- [ ] Team reviews the presentation and verifies all credits.
- [x] Publish the submission branch and video release asset. [PR #1](https://github.com/gengx-prog/qompute-2026-fair-aid-qaoa/pull/1) is merged into main.
- [x] Confirm with the team that nobody has already submitted.
- [x] Submit once using the contact email confirmed by the team.
- [x] Save the actual submission confirmation locally; public status is in `submission-status.json`.
- [ ] Confirm attendance arrangements for October 18.

GitHub repository: https://github.com/gengx-prog/qompute-2026-fair-aid-qaoa

Video file: `Track3_Presentation.mp4`, with synthetic English narration and burned-in captions. The MP4 is distributed as the release asset linked above, rather than duplicated in Git history. Publishing files does not establish a completed competition submission.

The prepared submission branch is [track3-submission-20261014](https://github.com/gengx-prog/qompute-2026-fair-aid-qaoa/tree/track3-submission-20261014). PR #1 was merged into `main` at commit `b074e86`. The current project files are available on [main](https://github.com/gengx-prog/qompute-2026-fair-aid-qaoa/tree/main/Track%203/bell_cloud/submission). The submitted branch remains available because the competition form references it. The competition form was successfully submitted once. The uploaded ZIP snapshots commit `dfd36e5`; subsequent updates record submission status only.
