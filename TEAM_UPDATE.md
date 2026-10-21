# Team update — Wed 21 Oct

Presentations are **Tue 27 Oct**, six days away. Repo (4%), report (8%) and presentation + individual Q&A (6%) are all due then.

## Done
- `project.ipynb` is built in the lab format: sections 0–12, numbered questions, *Answer:* cells, a W&B section and a submit checklist. All the code is written. **The full run on real GTSRB (T4, all 60 runs) finished on 21 Oct with no errors.** `project_results_with_answers.ipynb` has the outputs plus draft answers for every question.
- `README.md` and `requirements.txt` cover the repo marks.

## Headline result (test macro-F1, mean ± std over 3 seeds)
| N | classical (full bank) | frozen probe | fine-tuned | gap ft − classical (paired) |
|---|---|---|---|---|
| 200 | 0.317 ± 0.026 | 0.303 ± 0.015 | 0.478 ± 0.035 | +0.161 ± 0.009 |
| 500 | 0.457 ± 0.010 | 0.428 ± 0.024 | 0.789 ± 0.019 | +0.331 ± 0.014 |
| 1000 | 0.562 ± 0.007 | 0.477 ± 0.033 | 0.925 ± 0.021 | +0.363 ± 0.028 |
| 2000 | 0.659 ± 0.022 | 0.545 ± 0.021 | 0.963 ± 0.006 | +0.304 ± 0.026 |
| 5000 | 0.753 ± 0.006 | 0.603 ± 0.008 | 0.981 ± 0.006 | +0.227 ± 0.011 |

Majority floor: macro-F1 0.0025, accuracy 0.057.

- **Hypothesis, part 1 (CNN higher at every size): supported.** Every seed, every N.
- **Hypothesis, part 2 (gap widest at low data): contradicted.** The gap is smallest at N=200, peaks at N=1000 and shrinks after.
- Fine-tuning on 500 images (0.789) beats the classical pipeline on 5000 (0.753).
- Frozen ImageNet features are *worse* than the filter bank from N≥500. The gain comes from fine-tuning, not from the features alone.
- The classical pipeline fails on rare classes and on speed-limit digits (20→30 km/h 56%, 120→80 24%).
- Cost: the classical model is 0.43 MB vs 44.8 MB, but our implementation is *slower* per image (98 vs 25 ms on CPU), because of its large unoptimised filters.
- Tuning: full bank > MR8 (0.548 vs 0.367 val). The best fine-tune lr is at the top of the grid (1e-3), so the CNN numbers may be conservative.

Limitations to state in the report: grayscale only (no colour); linear classifier; seed mixes subset and initialisation variance; N=200 gives some classes only one image.

## Changes from the proposal (we must be able to defend these)
1. **Track-level validation split.** GTSRB images come in ~30-frame tracks of the same sign, so a random split leaks. Validation is used only for tuning, once, at N=1000.
2. **Two extra methods.** A majority-class floor (the brief wants something trivial beside every number), and a frozen-ResNet linear probe, which shows whether the gain comes from ImageNet features or from fine-tuning.
3. **Classical tuned fairly.** The pipeline tries MR8 (orientation discarded) vs. the full 38-filter bank, × 4 values of C. Fine-tuning tries 3 learning rates. Same budget, so the classical side is not a straw man.
4. **Fixed step budget (1,500 steps) for fine-tuning,** not fixed epochs, so the curve does not just measure compute. No augmentation, for either method.
5. **Data source.** The data comes from the Kaggle copy of the full original release, which has the proposal's 39,209 training and 12,630 test images. Q1.1 checks the counts.

## Split (Niel's A/B/C) and what each part hands over
Each part only needs to agree on the files it reads and writes, so the parts can be worked on separately.

| part | notebook sections | reads | hands over |
|---|---|---|---|
| **A: data & preprocessing** | §0–3: load, 64×64 cache, track-level val split, nested subsets, macro-F1, majority floor | Kaggle GTSRB | `gtsrb64.npz`, `results/subsets.json`, `macro_f1()` |
| **B1: classical baseline** | §4 + its row of §6: MR8 / full filter bank, logistic regression, tuning of features × C | A's arrays and subsets | a function `indices → test predictions`, `results/tuning.csv` |
| **B2: transfer learning** | §5 + its row of §6: ResNet-18 fine-tune + frozen probe, tuning of lr | A's arrays and subsets | same function shape as B1 |
| **C: comparison** | §7–10: the 60-run loop, curve + paired gap, cost, per-class failures | B1/B2 functions | `results/results.csv` (method, N, seed, macro_f1, accuracy, seconds), `results/preds/*.npy`, `figures/` |
| **Report** | Overleaf, 6 pp max: problem, method, experiment, results, next | all of the above | PDF |

The code for every part already runs (tested on synthetic data). Owning a part now means: run it on real data, check it, fill its *Answer:* cells, write its report section, and be able to explain it in the Q&A.

## Next
| when | what | who |
|---|---|---|
| ✅ 21 Oct | Full GPU run, draft answers | Khuselo |
| ✅ 21 Oct | results/, figures/, draft answers pushed to `khuselo/full-pipeline` | Khuselo |
| now | Reword the draft answers for your part; merge the PR | split by part |
| ~22 Oct | Report draft (6 pp): problem, method, experiment, results, next steps | all |
| ~25 Oct | Slides + Q&A rehearsal; each person explains every section | all |

**Proposed split for owning sections in the Q&A:** A: §1–3 (data, protocol, metric) · B: §4 + §10 (classical, failure analysis) · C: §5–6 + §9 (CNN, tuning, cost). Everyone owns §8, the main result.
