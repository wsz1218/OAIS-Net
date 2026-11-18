import torch

# --- Device Selection ---
# Automatically selects the best available device for PyTorch operations.
# It prioritizes MPS (for Apple Silicon GPUs), then CUDA (for NVIDIA GPUs), and falls back to CPU.
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'


# --- Pre-training Configuration ---
# Specifies whether to load weights from a pre-trained model before starting training.
PRE_TRAINED = False
# Path to the .pth.tar file of the pre-trained model.
PRE_TRAINED_PATH = 'PRE_TRAINED_PATH'


# --- Resume Training Configuration ---
# If True, the script will look for the latest checkpoint in 'SAVE_DIR' and resume training from there.
# If False, or if no checkpoint is found, training will start from scratch (or from the pre-trained model if enabled).
RESUME_TRAINING = False


# --- Core Training Hyperparameters ---
BATCH_SIZE = 4              # Number of samples per batch during training.
VAL_BATCH_SIZE = 1          # Number of samples per batch during validation.
NUM_EPOCHS = 2000           # Total number of epochs to train the model.
LEARNING_RATE = 2e-5        # The learning rate for the Adam optimizer.
WEIGHT_DECAY = 0            # Weight decay (L2 penalty) for the optimizer.
MIN_OBJECT_AREA = 0         # Minimum pixel area for an object to be considered valid (0 to disable).


# --- Validation and Logging ---
# Defines how often to run a validation cycle.
VALIDATION_STEP = 10        # Run validation every 10 epochs.


# --- Point Sampling for Interactive Segmentation ---
# Parameters related to simulating user clicks for interactive segmentation models.
NUM_MAX_POINTS = 20         # The maximum number of points to be sampled per image during training/validation.


# --- Inference Parameters ---
# Settings used during model evaluation or inference.
SAMPLE_CNT = 16             # Number of samples to draw from the model's probabilistic output.


# --- Dataset Configuration ---
# Specifies which dataset to use and where it is located.
# Dataset selection
DATASET_NAME = "SkeletalMuscle"  # Choose "Lucchi" or "SkeletalMuscle"
DATASET_DIR = 'pi_seg/data/datasets/Skeletal_muscle'

SAVE_DIR = 'SAVE_DIR'


# Test configuration
TEST_SAVE_DIR = 'SAVE_DIR'
TEST_MODEL_PATH = 'save/model_best_iou.pth'    # Path to the final trained model to be used for testing.
