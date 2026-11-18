# Click U-Net EM Baseline

This folder contains a plain U-Net interactive segmentation baseline for the
DeepPI/OAIS EM comparison experiments.

The method is intentionally simple:

- Input: grayscale EM image + positive-click Gaussian map + negative-click Gaussian map.
- Output: whole-image binary foreground mask.
- No online parameter update, no STA, and no LTM.
- Training clicks are simulated from the GT mask.
- Testing clicks follow the same automatic rule used in the OAIS comparison:
  compare current prediction with GT, find the largest connected error region,
  and click the distance-transform center.

## Train Four Domain Models

Run on the server:

```bash
cd /home/dell/wszwork/ams/A/DeepPI-EM-main/othermodel/U-Net

bash run_train.sh \
  --dataset-root /home/dell/wszwork/ams/A/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --epochs 300 \
  --batch-size 4 \
  --workers 4 \
  --crop-size 512 \
  --device cuda
```

Expected checkpoints:

```text
experiments/click_unet_em_domains/domain_2_cells/checkpoints/last_checkpoint.pth
experiments/click_unet_em_domains/domain_3_muscle/checkpoints/last_checkpoint.pth
experiments/click_unet_em_domains/domain_4_fuzzy_blobs/checkpoints/last_checkpoint.pth
experiments/click_unet_em_domains/domain_6_Lucchi/checkpoints/last_checkpoint.pth
```

Train a single domain:

```bash
bash run_train.sh \
  --dataset-root /home/dell/wszwork/ams/A/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --domains domain_6_Lucchi
```

## Evaluate

```bash
bash run_eval.sh \
  --dataset-root /home/dell/wszwork/ams/A/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --checkpoint-root experiments/click_unet_em_domains \
  --device cuda
```

Outputs:

```text
test_result/ClickUNet_EM/<domain>/click_unet_per_image.csv
test_result/ClickUNet_EM/<domain>/click_unet_summary_fixed_clicks.csv
test_result/ClickUNet_EM/<domain>/click_unet_summary_noc.csv
test_result/ClickUNet_EM/<domain>/visuals/*.png
test_result/ClickUNet_EM/click_unet_all_domains_fixed_clicks.csv
test_result/ClickUNet_EM/click_unet_all_domains_noc.csv
```

Metrics:

- IoU/Dice at fixed clicks: `1,3,5,10,15`
- NoC targets: `0.85,0.88,0.90,0.93`

## Recommended Settings

For final paper experiments, use at least:

```bash
--epochs 300 --crop-size 512 --batch-size 4
```

If the model underfits, increase epochs to `500`. If GPU memory is limited,
reduce `--batch-size` to `2`.

## Patent-Style Black-White Figures

If your Base/OAIS patent npz results are already saved, pass their root folder
with `--patent-reference-root`. The expected layout is:

```text
<patent-reference-root>/<domain>/patent_npz/Base/*.npz
```

Run Click U-Net evaluation and save patent figures at click 10:

```bash
bash run_eval.sh \
  --dataset-root /home/dell/wszwork/ams/A/DeepPI-EM-main/pi_seg/data/datasets/Skeletal_muscle \
  --checkpoint-root experiments/click_unet_em_domains \
  --save-patent \
  --patent-step 10 \
  --patent-reference-root /path/to/base_oais_patent_results \
  --patent-method-label "Interactive U-Net"
```

Outputs:

```text
test_result/ClickUNet_EM/<domain>/patent_npz/ClickUNet/*.npz
test_result/ClickUNet_EM/<domain>/patent_figures/ClickUNet/*.png
```
