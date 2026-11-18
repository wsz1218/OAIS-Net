# SAM2-EM Semantic Fine-Tuning for DeepPI/OAIS Comparison

This folder contains self-contained scripts for training and evaluating SAM2 on
the DeepPI/OAIS EM semantic segmentation data.

The comparison target is the same as OAIS-Net: whole-image binary foreground
segmentation. Each `target` mask is thresholded to one semantic foreground object
with palette label id `1`; background is `0`.

## Server Layout

Recommended layout on the server:

```text
/path/to/sam2-main/
  em_tools/
  checkpoints/sam2.1_hiera_base_plus.pt

/path/to/DeepPI-EM-main/
  pi_seg/data/datasets/
    Skeletal_muscle/
      domain_2_cells/train/input,target
      domain_2_cells/test/input,target
      ...
```

The scripts do not require the OAIS helper files. They only need the dataset
folders and the SAM2 code.

## Install

Run inside the SAM2 repo:

```bash
cd /path/to/sam2-main
pip install -e ".[dev]"
```

Put the official SAM2.1 base-plus checkpoint at:

```text
checkpoints/sam2.1_hiera_base_plus.pt
```

or pass another path with `--checkpoint`.

## Train One SAM2 Model per EM Domain on Server

From the SAM2 repo:

```bash
cd /path/to/sam2-main
bash em_tools/run_em_domain_finetune.sh \
  --dataset-root /path/to/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --epochs 40 \
  --batch-size 1 \
  --num-workers 4 \
  --num-gpus 1
```

This trains four independent SAM2 comparison models:

```text
sam2_logs/em_bplus_semantic_domains/domain_2_cells/checkpoints/checkpoint.pt
sam2_logs/em_bplus_semantic_domains/domain_3_muscle/checkpoints/checkpoint.pt
sam2_logs/em_bplus_semantic_domains/domain_4_fuzzy_blobs/checkpoints/checkpoint.pt
sam2_logs/em_bplus_semantic_domains/domain_6_Lucchi/checkpoints/checkpoint.pt
```

If GPU memory is limited, reduce `--resolution` to `768` or `512`.

If you only want to train one domain, pass `--domains`:

```bash
bash em_tools/run_em_domain_finetune.sh \
  --dataset-root /path/to/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --checkpoint checkpoints/sam2.1_hiera_base_plus.pt \
  --domains domain_6_Lucchi
```

## Evaluate All EM Test Domains

Evaluate the four fine-tuned domain models with the same automatic click rule
used by the OAIS/RITM/SimpleClick comparison: the next click is generated from
the largest connected error region between current prediction and GT.

```bash
cd /path/to/sam2-main
bash em_tools/run_em_eval.sh \
  --dataset-root /path/to/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --checkpoint-root sam2_logs/em_bplus_semantic_domains \
  --merge-strategy prompt_set
```

For disconnected foreground targets, also run:

```bash
bash em_tools/run_em_eval.sh \
  --dataset-root /path/to/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --checkpoint-root sam2_logs/em_bplus_semantic_domains \
  --merge-strategy positive_union
```

Use the better, explicitly reported strategy as the SAM2 comparison setting.

## Outputs

Evaluation writes results to:

```text
test_result/SAM2_EM/
```

Each domain output contains:

- `sam2_per_image.csv`
- `sam2_summary_fixed_clicks.csv`
- `sam2_summary_noc.csv`
- `visuals/*.png`

The evaluation wrapper also writes paper-ready combined summaries:

- `sam2_all_domains_fixed_clicks_prompt_set.csv`
- `sam2_all_domains_noc_prompt_set.csv`

Metrics:

- IoU at fixed clicks: `1,3,5,10,15`
- Dice at fixed clicks
- HD95 at fixed clicks
- NoC for target IoU: `0.85,0.88,0.90,0.93`

## Patent-Style Black-White Figures

If your Base/OAIS patent npz results are already saved, pass their root folder
with `--patent-reference-root`. The expected layout is:

```text
<patent-reference-root>/<domain>/patent_npz/Base/*.npz
```

Run SAM2 evaluation and save patent figures at click 10:

```bash
bash em_tools/run_em_eval.sh \
  --dataset-root /path/to/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --checkpoint-root sam2_logs/em_bplus_semantic_domains \
  --merge-strategy prompt_set \
  --save-patent \
  --patent-step 10 \
  --patent-reference-root /path/to/base_oais_patent_results \
  --patent-method-label "SAM2"
```

Outputs:

```text
test_result/SAM2_EM/<domain>_<strategy>/patent_npz/SAM2_<strategy>/*.npz
test_result/SAM2_EM/<domain>_<strategy>/patent_figures/SAM2_<strategy>/*.png
```

## Windows Local Smoke Test

From the SAM2 repo:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File em_tools\run_em_finetune.ps1 `
  -DatasetRoot F:\A\DeepPI-EM-main\DeepPI-EM-main\pi_seg\data\datasets `
  -SkipTrain
```

This checks data conversion and config generation without starting training.
