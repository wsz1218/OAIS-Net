# OAIMS on EM datasets

These entry points adapt the author-provided binary brain implementation to EM
images. The original source files are unchanged. This is a reproduction of the
OAIMS architecture and MI/PI adaptation on EM, not a reproduction of the original
fundus/brain benchmark scores, and not OAIS-Net with a replacement loss.

## Environment

Use a dedicated environment with the original `requirements.txt` (Python 3.10 is
recommended by the author). Do not install these pins over the SAM2 environment.

```bash
conda create -n oaims_em python=3.10
conda activate oaims_em
pip install -r requirements.txt
```

Commands below run from `OAIMS-main`. `DATA` is the same data root used for the
other baselines; replace it with the real server path. Each domain must contain
`train/input`, `train/target`, `test/input`, and `test/target`. Images and masks
must share a stem. Masks already encoded as 0/1 are preserved. Other 8-bit
grayscale targets (including 0/255 masks with intermediate gray values) are
binarized with `mask > 127`, matching the existing SAM2 and Click U-Net EM
helpers. The same conversion is used for training, validation and evaluation;
source mask files are never modified. This loader expects binary semantic
targets, not arbitrary multiclass or instance-ID maps.

Domains: `domain_2_cells`, `domain_3_muscle`, `domain_4_fuzzy_blobs`,
`domain_6_Lucchi`. Verify the actual specimen identity of each domain before
labelling tables: folder names alone do not establish PdCu/Au provenance.

## Training

```bash
DATA=/path/to/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle
python train_oaims_em_domains.py --dataset-root "$DATA" --dry-run

CUDA_VISIBLE_DEVICES=1 python train_oaims_em_domains.py \
  --dataset-root "$DATA" \
  --output-root experiments/oaims_em_domains \
  --epochs 600 --batch-size 4 --crop-size 256 --num-points 10 \
  --workers 0 --seed 42
```

Four domains train sequentially; each receives its own initialization, optimizer
and checkpoint. Add `--domains domain_2_cells` for a first single-domain run.
The source training default is batch size 8; the example reduces it to 4 for
memory. Source defaults otherwise retained include Adam lr=0.0005, 600 epochs,
10 training points, DF+CCG, and the optimizer-step learning-rate schedule. The
EM loader adds fixed-size random crops (256 by default) and uses the author's
grayscale augmentation. Per-image z-score normalization is retained.

No validation/test data are used for checkpoint selection by default. The main
checkpoint is `DOMAIN/checkpoints/last_checkpoint.pth`. Existing runs cannot be
silently overwritten. Use `--resume` with unchanged training settings to resume,
or select a new output root. For independent validation, explicitly supply
`--val-root /path/to/data --val-split val`; this requires `DOMAIN/val/input,target`.
It additionally saves `best_checkpoint.pth` by frozen validation Dice at 6 clicks.
Do not pass the test split as validation. Validation masks are never used in
online optimizer losses.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate_oaims_em_domains.py \
  --dataset-root "$DATA" \
  --checkpoint-root experiments/oaims_em_domains \
  --output-root results \
  --max-clicks 15 --target-clicks 1,3,5,10,15 \
  --target-ious 0.85,0.88,0.90 \
  --modes mi_pi frozen --click-policy distance --runs 1 --seed 42
```

CUDA visibility is controlled only by the shell. Under
`CUDA_VISIBLE_DEVICES=1`, the default `cuda` selects the one visible GPU. There
is no DataParallel or distributed training. `--workers 0` avoids data-loader
worker processes. Do not use `--device cuda:1` with only physical GPU 1 exposed.

`mi_pi` is full OAIMS; `frozen` is its own offline model without online parameter
updates (not OAIS-Net's Base). Optional `pi` disables MI but keeps PI. Within
each domain/mode/run, the weights AND Adam state persist across images. Each new
domain, mode and run starts fresh from the offline checkpoint. Seed is fixed and
recorded per domain/run; increasing `--runs` repeats online evaluation, not offline
training. It does not constitute multiple independent training runs.

### Protocol

- Test image order is natural filename order, matching the existing EM helpers.
  Use `--image-list-dir /path/to/lists` for explicit ordered lists named
  `domain_2_cells.txt`, etc. Each line is an exact filename. This is useful when
  historical baselines evaluated different subsets; equal folder names are not
  proof of equal test samples.
- Default external clicks use the same largest-error-region, distance-transform
  rule as the existing SAM2/Click U-Net evaluators. First click is generated from
  an empty foreground mask. `--min-error-area` defaults to 5.
- `--click-policy reference` retains the author's random point selection inside
  the largest class-specific error component. Neither policy fabricates a
  `(0,0)` background click when there is no remaining error. Later fixed-click
  columns then retain the last prediction and record the actual click count.
- Preserve the author's MI timing: prediction with k clicks is recorded FIRST;
  then the next click generates a detached pseudo-target, and one MI step
  updates the k-click logits. Thus IoU@k never uses click k+1. The last click
  has no MI update. All network parameters are eligible for updating.
- PI runs once after the fixed click budget, using the final predicted mask.
  It reuses the original function: localization-click update, then an internally
  sampled correction-click update (10 internal points by default). These internal
  points are generated against the final prediction, not test GT, and are not
  user clicks. Optimizer cost is included in PI time. PI is not empty-click LTM.
- Test GT is used only by the external simulator and metric calculation. There
  is no GT-based rollback, best-iterate selection, or GT target in MI/PI losses.
- Fixed-click IoU and Dice are computed from the SAME binary prediction. Each
  dataset is summarized with equal weight per image. No cross-dataset pooled
  score is computed. Identical empty prediction/GT has IoU=Dice=1, consistent
  with the existing EM evaluation helpers.
- NoC is the first target crossing on the SAME fixed-budget trajectory, capped
  at 15 for failures; success and failure rate are exported separately. It is
  not a separate target-stopped sequence. For adaptive models, stopping PI at
  each target could change later images; do not combine those two protocols.
- HD95 is in pixels and follows the existing evaluator's empty-mask penalty of
  50 (including two empty masks), rather than claiming a physical-distance metric.
- The author's `models/losses.py` CCG call has unconventional cross-entropy
  argument order. This integration deliberately retains it, records source
  hashes, and does not silently substitute a different loss. Any corrected-loss
  experiment must be separately identified and retrained.

### Outputs

Each evaluation creates `results/OAIMS_EM_TIMESTAMP/` without overwriting prior runs:

- `evaluation_config.json`: exact settings, protocol and original source hashes.
- `oaims_all_domains_fixed_clicks.csv`, `oaims_all_domains_noc.csv`: all domain summaries.
- `DOMAIN/sequence_manifest.csv`: actual image/mask paths and sequence order.
- `DOMAIN/oaims_per_image.csv`: IoU, Dice, HD95 at reported click budgets.
- `DOMAIN/oaims_click_trajectories.csv`: every actual click result.
- `DOMAIN/oaims_per_image_noc.csv`: capped NoC AND success flag per target/image/run.
- `DOMAIN/oaims_summary_fixed_clicks.csv`, `oaims_summary_noc.csv`.
- `DOMAIN/oaims_timing.csv`: inference, MI (including its extra target forward),
  PI and compute wall times per image. CUDA is synchronized. This excludes data
  loading and visualization and is NOT actual human annotation time.
- `DOMAIN/MODE/visuals/`: image+clicks, GT and prediction/error overlays. Colors:
  green=true positive, red=false positive, blue=false negative.

IoU/Dice CSV values are fractions in [0,1]; multiply by 100 for paper tables.
`*_std` is population SD over image-run observations, not training-run SD.
`*_run_mean_std` separately describes dataset-mean variability across evaluation
runs. It is normal for deterministic distance-click evaluation to have very
small or zero run variability. Keep raw statistics even if paper tables show means.

## CPU protocol checks

```bash
python test_oaims_em_protocol.py --oaims-root . -v
```

This uses synthetic images, not your test dataset. It checks paired metrics,
GT separation, update counts, filename pairing, padded image sizes, and numerical
agreement with the upstream MI+PI trajectory using its original click policy.
No full training experiment is launched by this test.
