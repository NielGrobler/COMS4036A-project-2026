# Team update — Sun 18 Oct

Presentations are **Tue 27 Oct**, nine days away. Repo (4%), report (8%) and presentation + individual Q&A (6%) are all due then.

## Done
- `project.ipynb` is built in the lab format: sections 0–12, numbered questions, *Answer:* cells, a W&B section and a submit checklist. All the code is written. It has been run end to end on a small synthetic stand-in, so every cell executes. **It has not run on real GTSRB on a GPU yet.**
- `README.md` and `requirements.txt` cover the repo marks.

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
| now | Run all on Colab GPU (~1 h); fix anything real data breaks | Khuselo |
| after run | Fill the *Answer:* cells | split by section |
| ~22 Oct | Report draft (6 pp): problem, method, experiment, results, next steps | all |
| ~25 Oct | Slides + Q&A rehearsal; each person explains every section | all |

**Proposed split for owning sections in the Q&A:** A: §1–3 (data, protocol, metric) · B: §4 + §10 (classical, failure analysis) · C: §5–6 + §9 (CNN, tuning, cost). Everyone owns §8, the main result.
