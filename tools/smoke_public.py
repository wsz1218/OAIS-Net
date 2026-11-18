"""CPU end-to-end smoke test on synthetic data, never on the user's datasets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", help="A new directory for synthetic images and generated checkpoints")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="oais_smoke_"))
    if args.work_dir:
        work.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(123)
    for split, count in (("train", 2), ("val", 1), ("test", 2)):
        for folder in ("input", "target"):
            (work / split / folder).mkdir(parents=True)
        for i in range(count):
            mask = np.zeros((32, 32), np.uint8)
            cv2.circle(mask, (14+i, 16), 7, 255, -1)
            image = np.clip(mask.astype(float) * .6 + rng.normal(60, 15, mask.shape), 0, 255).astype(np.uint8)
            cv2.imencode(".png", image)[1].tofile(str(work / split / "input" / f"image{i}.png"))
            cv2.imencode(".png", mask)[1].tofile(str(work / split / "target" / f"image{i}.png"))
    def run(script, options):
        command = [sys.executable, "-B", str(root / script), *map(str, options)]
        print("RUN:", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=root, check=True)
    train_args = ["--train-dir", work / "train", "--val-dir", work / "val", "--output", work / "trained",
                  "--device", "cpu", "--threads", 2, "--batch-size", 2, "--crop-size", 32,
                  "--train-max-clicks", 2, "--val-clicks", 1, "--val-every", 1]
    run("main.py", train_args + ["--epochs", 1])
    checkpoint = work / "trained" / "model_latest.pth"
    assert checkpoint.is_file()
    # Exercise checkpoint reload and Windows worker spawning as well as fresh training.
    run("main.py", train_args + ["--epochs", 2, "--resume", checkpoint, "--workers", 1])
    for protocol in ("fixed", "noc"):
        run("test.py", ["--test-dir", work / "test", "--checkpoint", checkpoint,
                        "--output", work / protocol, "--device", "cpu", "--threads", 2,
                        "--max-clicks", 2, "--mode", "base", "oais", "--protocol", protocol,
                        "--save-masks"])
        results = json.loads((work / protocol / "summary.json").read_text())
        assert len(results) == 2 and all(r["images"] == 2 for r in results)
        for r in results:
            assert 0 <= r["mean_final_iou"] <= 1
            if protocol == "noc":
                assert 0 <= r["capped_noc"] <= 2 and 0 <= r["success_rate"] <= 1
    print(f"PASS: train, resume with worker, checkpoint reload, fixed and NoC Base/OAIS evaluation. Artifacts: {work}")


if __name__ == "__main__":
    main()
