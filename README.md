# DecoyOpt

Enzyme/non-enzyme discrimination and EC number prediction using frozen
[ProtT5-XL-UniRef50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) features.
A binary classifier learns from enzymes, unannotated proteins and D0/D5/D10 decoys;
an independent CLEAN encoder predicts EC labels from training-enzyme centers.

## Install

Run from the repository root:

```bash
mamba create -n decoyopt_codex python=3.10 pip -y
mamba activate decoyopt_codex
python -m pip install -e .
```

## Download data

Download [decoyopt_data.tar.gz](https://drive.google.com/file/d/1FvPlTQCJO3pOgX74LrWNOkRELiThT-o4/view?usp=drive_link)
from Google Drive:

```bash
mkdir -p dist
gdown 1FvPlTQCJO3pOgX74LrWNOkRELiThT-o4 -O dist/decoyopt_data.tar.gz
tar -xzf dist/decoyopt_data.tar.gz
```

The archive creates `data/`:

```text
data/
  sequences.csv                   # one sequence per protein/decoy
  decoys.csv                      # decoy ID, parent ID, category
  five_fold/graphpart_{40,60,80}/fold_{1..5}.csv
  manifest.json                   # sample counts and SHA-256 hashes
```

The fold tables contain `enzyme_id`, `label`, `EC number` and `split`
(`train`, `val`, `test`). Each decoy inherits its parent's partition.
Within each split, the five models share one fixed test set.

## Generate features

```bash
CUDA_VISIBLE_DEVICES=0 decoyopt cache --device cuda:0
```

The command downloads the ProtT5 model to `cache/models/` and writes
`cache/shared_ids.txt`, `cache/ProtT5-XL-UniRef50.f32` and `cache/metadata.json`.
Features have 1024 dimensions, with mean pooling over residue tokens, excluding
padding and special tokens. Preprocessing matches the original cache: uppercase,
UZOBJ → X and a 1024-token limit. GPU inference uses bfloat16 when supported,
otherwise float16; the matrix is stored as float32. All 15 models share it.
Run the same command to resume interrupted generation.

Use `--batch-size 2` to reduce GPU memory, or `--model /path/to/prot_t5_xl_uniref50`
to load an existing local model.

## Train

Train all three splits and five folds:

```bash
CUDA_VISIBLE_DEVICES=0 decoyopt train --device cuda:0
```

Train a selected fold:

```bash
CUDA_VISIBLE_DEVICES=0 decoyopt train --splits 40 --folds 1 --device cuda:0
```

Each fold trains two independent models with seed 1234 and Adam (lr 5e-4).
The binary model runs for 20,000 epochs with source-balanced BCE plus
0.5 × ranking loss (P > D0 > D5 > D10, margin 1). The CLEAN encoder runs for
2,000 epochs using enzyme-only triplets, refreshing nearest-class negatives and
resetting Adam every 100 epochs. Final-epoch weights and training EC centers
are saved to `results/<split>/fold_<n>/`.

Completed models are skipped when the same command is repeated. Use
`--overwrite` to retrain, or `--results-dir` to select another output directory.

## Test

```bash
CUDA_VISIBLE_DEVICES=0 decoyopt test --device cuda:0
```

Select models with `--splits 40 60 80` and `--folds 1 2 3 4 5`.
The binary threshold maximizes validation MCC and is frozen for testing.
EC labels use CLEAN maximum separation over training EC centers.

| Evaluation | Samples | Metrics |
|---|---|---|
| Enzyme/non-enzyme discrimination | Enzymes + all negatives | AUROC, AP, macro F1, MCC, CEN |
| Negative FPR | Unannotated proteins, D0, D5, D10 separately | FPR per source |
| EC prediction, full test set | Enzymes + all negatives | Weighted precision/recall/F1, macro F1, overall MCC |
| EC prediction, enzyme subset | All true test enzymes | Same five EC metrics |

EC metrics are reported at levels 1–4. Full-test evaluation includes an explicit
`non_enzyme` class for negative truth and rejected predictions. Enzyme-subset
evaluation keeps rejected enzymes as empty predictions, contributing false
negatives. Overall MCC pools the sample × label indicators. `ec_direct` also
reports enzyme-only CLEAN predictions without the binary gate.

Each fold produces `metrics.json`, `thresholds.json` and `predictions.csv`.
Once all five folds finish, `results/<split>/five_fold_summary.csv` reports
mean ± sample SD. Regenerate summaries with:

```bash
decoyopt summary
```

## Test the provided checkpoints

Download [decoyopt_checkpoints.tar.gz](https://drive.google.com/file/d/1ZYhVuXcdrlmYNKQ5GGEe9mc4WyNxix29/view?usp=drive_link)
from Google Drive:

```bash
mkdir -p dist
gdown 1ZYhVuXcdrlmYNKQ5GGEe9mc4WyNxix29 -O dist/decoyopt_checkpoints.tar.gz
tar -xzf dist/decoyopt_checkpoints.tar.gz
CUDA_VISIBLE_DEVICES=0 decoyopt test --checkpoint-dir checkpoints --results-dir results/pretrained --device cuda:0
```

Generate the dataset cache first. The checkpoint archive contains all 15 pairs
of binary/EC weights, EC centers, vocabularies, training settings, validation
thresholds and original reference metrics. Evaluation recalibrates thresholds on
the corresponding validation partition and writes results to `results/pretrained/`.

All commands accept `--data-dir` and `--cache-dir` to use other data locations.
