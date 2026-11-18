# OAIS-Net

OAIS-Net combines short-term adaptation (STA) within an image with long-term
memory (LTM) updates across images for interactive segmentation. STA uses click
feedback to update the decoder and prediction head. LTM uses the final prediction
after interaction to update the encoder and retain the updated parameters for
subsequent images.

## Datasets

The following **electron microscopy datasets** are available from their original
providers or the project team's shared repository. Please cite the corresponding
publications and follow each provider's data-use terms.

### Lucchi++

- [Dataset page](https://casser.io/connectomics/)
- [Alternative project page and download](https://sites.google.com/view/connectomics/)

A mitochondria segmentation benchmark with revised annotations of the EPFL
Hippocampus electron microscopy image stacks. Open the dataset page and select
the **Lucchi++** download, rather than Kasthuri or Kasthuri++.

### Skeletal Muscle TEM

- [Download images and annotations from Google Drive](https://drive.google.com/drive/folders/1n2ZqbJEHPyMB_6a6OTBBACt5Jct2PZJc?usp=sharing)

A custom dataset developed by our team for mitochondria segmentation in skeletal
muscle transmission electron microscopy (TEM) images.

### HeLa Cell Electron Microscopy

- [Raw electron microscopy images: EMPIAR-10094](https://www.ebi.ac.uk/empiar/EMPIAR-10094/)
- [Four-class annotations: Zenodo 3874949](https://zenodo.org/records/3874949)
- [Alternative multiple-nuclei annotations: Zenodo 6355622](https://zenodo.org/records/6355622)
- [Related publication: Impact of Training Data, Ground Truth and Shape Variability in the Deep Learning-Based Semantic Segmentation of HeLa Cells Observed with Electron Microscopy](https://doi.org/10.3390/jimaging9030059)

These links provide the serial block-face scanning electron microscopy HeLa data
and annotation resources described in the publication cited by the OAIS-Net
manuscript. The two annotation releases differ in their labeling scope; they
should not be treated as interchangeable. This is the HeLa **electron microscopy**
dataset, not the similarly named optical microscopy datasets in the Cell Tracking
Challenge.

The original annotations distinguish nuclear envelope, nucleus, the rest of the
cell, and background. Before using them with OAIS-Net, select the required target
classes and export a binary foreground mask. Do not pass the original multiclass
label values directly to the binary-mask loader. Match the annotated region of
interest to the corresponding raw image region and slice.

## Data Preparation

Prepare paired images and binary masks in the following layout:

```text
data/<dataset>/
  train/
    input/image001.tif
    target/image001.png
  val/
    input/image101.tif
    target/image101.png
  test/
    input/image201.tif
    target/image201.png
```

- Each image and mask must have the same filename stem and spatial dimensions.
- Use 0 for background and 1 or 255 for the selected foreground.
- Supported input formats include PNG, single-image TIF/TIFF, JPG/JPEG and BMP.
  Export volume slices and MATLAB annotations into paired 2D image files first.
- Separate training, validation and test source images before extracting patches.
  Keep related slices and image regions grouped appropriately to avoid leakage.
- The links above provide source data, not a prepackaged OAIS-Net split. Exact
  reproduction requires the same image selection, label mapping and sequence
  order as the reported experiment; downloading the source dataset alone does
  not define those settings.

## Installation

Use Python 3.11 in an isolated environment. Install the PyTorch 2.6.0 and
torchvision 0.21.0 build suitable for your CPU or GPU, following the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/),
then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Training

Run from the repository root and replace the example paths with your own:

```bash
python main.py --train-dir data/hela/train --val-dir data/hela/val --output runs/hela --epochs 1000 --batch-size 4
```

Add `--encoder-checkpoint PATH_TO_ENCODER.pth.tar` to use compatible CEM500K
encoder weights. Without this option, training starts from random weights.
The validation split is optional and must be independent of the final test set.
Training saves `model_latest.pth`; with validation it also saves the checkpoint
with the best validation IoU as `model_best_iou.pth`.

## Testing

Fixed-click evaluation of Base and OAIS-Net:

```bash
python test.py --test-dir data/hela/test --checkpoint runs/hela/model_latest.pth --mode base oais --protocol fixed --max-clicks 15 --output runs/hela_fixed --save-masks
```

Evaluation of the number of clicks needed to reach a target IoU:

```bash
python test.py --test-dir data/hela/test --checkpoint runs/hela/model_latest.pth --mode base oais --protocol noc --target-iou 0.90 --max-clicks 15 --output runs/hela_noc90
```

Outputs include per-image and per-click CSV files, configuration records and
summary metrics. IoU and Dice are reported in [0,1]. NoC is capped at the click
budget, with target attainment and success rate reported separately.
Use `--mode base sta ltm oais` for branch comparisons and `--order reverse` or
`--order random` for sequence-order experiments. Use a new output directory for
each experiment. Run `python main.py --help` or `python test.py --help` for all
options.

Ground-truth masks are used by the evaluation simulator to generate clicks,
calculate metrics and apply target-IoU stopping. The public online adaptation
workflow uses model predictions, rather than ground-truth masks, as its learning
targets. These commands perform simulated-click evaluation, not a real-user
annotation-time study.

## Reproducibility

The supported public entry points are `main.py` and `test.py`. Historical
experiment scripts and the GUI are separate and may require additional setup.
Trained weights are not downloaded automatically; provide a compatible
checkpoint or train one locally. Matching published numerical results requires
the corresponding data splits, weights and experimental settings.

