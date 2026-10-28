# Labelled-data size vs. classical and transfer-learning traffic-sign classifiers

COMS4036A Honours project · Khuselo Sofohlo (2729931), Wakhiwe Ndzimandze (2694083), Niel Grobler (2690034)

**Question:** How does the amount of labelled training data change the advantage of an ImageNet-pretrained ResNet-18 over a classical MR8 filter-bank pipeline, on GTSRB traffic-sign classification?

**Hypothesis:** The pretrained CNN keeps a higher macro-F1 at every size, and the gap widens most in the low-data regime.

## What is compared

| method | what it is | role |
|---|---|---|
| `majority` | predict the most frequent training class | trivial floor |
| `classical` | grayscale → MR8 / 38-filter bank → 4×4 spatial pooling → logistic regression | baseline (Approach 1) |
| `probe` | frozen ImageNet ResNet-18 features → logistic regression | control: features vs. fine-tuning |
| `finetune` | ImageNet ResNet-18, new 43-way head, all weights fine-tuned | Approach 2 |

Training sizes N ∈ {200, 500, 1000, 2000, 5000}, seeds {0, 1, 2}, giving 60 runs. Metric: macro-F1 (primary) and accuracy on the official 12,630-image GTSRB test set.

**Held fixed:** test set, the 64×64 canonical resolution, a track-level validation split, nested class-stratified subsets (identical for every method at a given seed), the optimiser-step budget for fine-tuning, the tuning budget (done once at N=1000, on validation only), and the evaluation code.

## Setup

The project runs on Google Colab with a GPU (a T4 is enough).

1. Open `project.ipynb` in Colab and set **Runtime ▸ Change runtime type ▸ GPU**.
2. Locally: `pip install -r requirements.txt` (Python ≥ 3.10).
3. W&B: log in when the notebook prompts. For no logging, set `CFG["wandb_mode"] = "disabled"`.

## Data

The source is the [Kaggle copy of the original GTSRB release](https://www.kaggle.com/datasets/aakcodebreaker/german-traffic-sign-recognition-benchmark) (~490 MB, Apache 2.0): 39,209 training images (`GTSRB/Final_Training/Images/000NN/`) and 12,630 test images (`GTSRB/Final_Test/Images` + `GT-final_test.csv`). The notebook downloads it with `kagglehub` on first run. If Kaggle asks for credentials, add `KAGGLE_USERNAME` / `KAGGLE_KEY` as Colab secrets.

Other sources, set in `CFG`:
- `data_source="local", data_dir=...` uses a copy you already have, in the same layout.
- `data_source="torchvision"` uses torchvision's mirror. Note that its training archive (`GTSRB-Training_fixed`) is smaller than 39,209.

Every image is resized once to 64×64 and cached as `gtsrb64.npz`.

## Running

1. Open `project.ipynb` in Colab and set **Runtime ▸ Change runtime type ▸ T4 GPU**.
2. Choose **Runtime ▸ Run all**. The full run takes about one hour on a T4: most of that is the 15 fine-tuning runs, plus tuning.
3. The first cell mounts Google Drive and works from `MyDrive/coms4036a-run`, so the data cache and `results/` survive a disconnect. Outside Colab it does nothing.

The notebook in this repository is saved **with the outputs of our run** (21 Oct 2026), so you can read the results without running it.

The experiment loop (Section 7) is resumable: it appends each run to `results/results.csv` and skips runs already there. After a disconnect, choose **Run all** again. To reproduce from scratch, delete `results/` first; every seed is fixed.

## Outputs

| file | contents |
|---|---|
| `results/subsets.json` | exact training indices for every (seed, N) |
| `results/tuning.csv` | validation scores for every hyperparameter tried |
| `results/results.csv` | one row per run: method, N, seed, macro-F1, accuracy, seconds |
| `results/preds/*.npy` | test predictions for every run |
| `results/summary.csv` | mean ± std per method and N (the report's main table) |
| `results/gap_*.csv` | paired gap per N |
| `results/cost.csv` | parameters, size, latency, training time |
| `figures/*.png` | every figure in the report |

## Reproducibility

All randomness goes through `seed_everything(seed)`, with cuDNN in deterministic mode. The validation split is fixed by `CFG["split_seed"]`. Subsets come from `np.random.RandomState(seed)`. Mixed precision on the GPU can still move fine-tuning results in the 3rd–4th decimal between machines.
