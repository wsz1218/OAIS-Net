param(
    [Parameter(Mandatory=$true)]
    [string]$DatasetRoot,
    [string]$Checkpoint = "",
    [string]$ConfigName = "sam2.1_hiera_b+_EM_semantic_finetune.yaml",
    [string]$DataDir = "",
    [string]$ExperimentLogDir = "",
    [int]$Epochs = 40,
    [int]$NumWorkers = 4,
    [int]$NumFrames = 1,
    [int]$BatchSize = 1,
    [int]$Resolution = 1024,
    [int]$NumGpus = 1,
    [switch]$SkipDataPrepare,
    [switch]$SkipConfig,
    [switch]$SkipTrain
)

$ErrorActionPreference = "Stop"

$Sam2Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ($Checkpoint -eq "") {
    $Checkpoint = Join-Path $Sam2Root "checkpoints\sam2.1_hiera_base_plus.pt"
}
if ($DataDir -eq "") {
    $DataDir = Join-Path $Sam2Root "em_data\oais_semantic_train"
}
if ($ExperimentLogDir -eq "") {
    $ExperimentLogDir = Join-Path $Sam2Root "sam2_logs\em_bplus_semantic_finetune"
}

if (-not (Test-Path $DatasetRoot)) {
    throw "Dataset root not found: $DatasetRoot"
}
if ((-not $SkipConfig) -and (-not (Test-Path $Checkpoint))) {
    throw "SAM2 checkpoint not found: $Checkpoint"
}

if (-not $SkipDataPrepare) {
    python (Join-Path $PSScriptRoot "prepare_em_data.py") `
        --scan-root $DatasetRoot `
        --output-dir $DataDir
}

if (-not $SkipConfig) {
    python (Join-Path $PSScriptRoot "make_em_config.py") `
        --data-dir $DataDir `
        --checkpoint $Checkpoint `
        --config-name $ConfigName `
        --experiment-log-dir $ExperimentLogDir `
        --epochs $Epochs `
        --num-workers $NumWorkers `
        --num-frames $NumFrames `
        --batch-size $BatchSize `
        --resolution $Resolution `
        --gpus-per-node $NumGpus
}

if (-not $SkipTrain) {
    Push-Location $Sam2Root
    try {
        python training/train.py `
            -c "configs/sam2.1_training/$ConfigName" `
            --use-cluster 0 `
            --num-gpus $NumGpus
    }
    finally {
        Pop-Location
    }
}

$FineTunedCheckpoint = Join-Path $ExperimentLogDir "checkpoints\checkpoint.pt"

Write-Host ""
Write-Host "SAM2-EM semantic fine-tuning pipeline finished or training was skipped."
Write-Host "Expected fine-tuned checkpoint:"
Write-Host "  $FineTunedCheckpoint"
