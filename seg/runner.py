"""Command-line training and evaluation; legacy experiment scripts are separate."""
import argparse
import csv
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import initialize_model, load_model_weights, read_checkpoint, resolve_device
from seg.dataio import PatchDataset, discover_pairs
from seg.interaction import add_click, empty_points, evaluate_sequence, next_click
from seg.inference.online import OnlineAdaptor, OnlineSettings


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_output(path, resume=False):
    path = Path(path)
    if path.exists() and any(path.iterdir()) and not resume:
        raise ValueError(f"Output directory is not empty: {path}. Choose a new directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def common_options(parser):
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--max-size", type=int, default=1024, help="Longest image side; 0 disables resizing")
    parser.add_argument("--threads", type=positive_int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--output", required=True, help="New output directory")


def check_common(args):
    if args.max_size < 0:
        raise ValueError("max-size must be >= 0")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    return resolve_device(args.device)


def train_parser():
    parser = argparse.ArgumentParser(description="Train OAIS-Net offline on paired binary masks.")
    common_options(parser)
    parser.add_argument("--train-dir", required=True, help="Directory containing input/ and target/")
    parser.add_argument("--val-dir", help="Optional independent validation split, never the final test split")
    parser.add_argument("--epochs", type=positive_int, default=1000)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--crop-size", type=positive_int, default=256)
    parser.add_argument("--workers", type=int, default=0, help="0 is portable; increase for parallel loading")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-max-clicks", type=positive_int, default=20)
    parser.add_argument("--val-clicks", type=positive_int, default=5)
    parser.add_argument("--val-every", type=positive_int, default=10)
    parser.add_argument("--encoder-checkpoint", help="Optional CEM500K/MoCo ResNet-50 encoder weights")
    parser.add_argument("--resume", help="Resume a checkpoint created by this training entry point")
    return parser


def train_main(argv=None):
    args = train_parser().parse_args(argv)
    device = check_common(args)
    if args.crop_size < 32 or args.crop_size % 32 or args.workers < 0 or args.lr <= 0:
        raise ValueError("crop-size must be a multiple of 32; workers >= 0; lr > 0")
    pairs = discover_pairs(args.train_dir)
    val_pairs = discover_pairs(args.val_dir) if args.val_dir else []
    if val_pairs:
        fingerprints = {hashlib.sha256(p.read_bytes()).digest() for p, _ in pairs}
        if any(hashlib.sha256(p.read_bytes()).digest() in fingerprints for p, _ in val_pairs):
            raise ValueError("Training and validation contain identical image files. Use disjoint source images.")
    output = prepare_output(args.output, resume=bool(args.resume))
    save_json(output / "train_config.json", vars(args))
    save_json(output / "data_manifest.json", {
        "train": [[str(p.resolve()), str(m.resolve())] for p, m in pairs],
        "validation": [[str(p.resolve()), str(m.resolve())] for p, m in val_pairs]})
    model = initialize_model(device, None if args.resume else args.encoder_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start, best = 1, -1.0
    if args.resume:
        checkpoint = read_checkpoint(args.resume)
        load_model_weights(model, checkpoint)
        if "optimizer" not in checkpoint or "epoch" not in checkpoint:
            raise ValueError("Resume requires a training checkpoint with optimizer and epoch.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["epoch"] + 1, checkpoint.get("best_iou", -1.0)
    dataset = PatchDataset(pairs, args.crop_size, args.max_size)
    generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        worker_init_fn=seed_worker, generator=generator)
    print(f"Training: {len(pairs)} images; validation: {len(val_pairs)}; device: {device}", flush=True)
    for epoch in range(start, args.epochs + 1):
        seed_everything(args.seed + epoch)
        generator.manual_seed(args.seed + epoch)
        losses = []
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            points = empty_points(len(image), args.train_max_clicks, device)
            histories = [[] for _ in range(len(image))]
            model.eval()
            with torch.no_grad():
                for _ in range(random.randint(0, args.train_max_clicks)):
                    predicted = torch.sigmoid(model(image, points, training=False)["instances"]) > 0.5
                    for i in range(len(image)):
                        click = next_click(predicted[i, 0].cpu().numpy(),
                                           target[i, 0].cpu().numpy(), histories[i])
                        if click is not None:
                            histories[i].append(click)
                            add_click(points, click, i)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(image, points, training=True)["instances"]
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; check inputs and learning rate.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_iou = None
        improved = False
        if val_pairs and (epoch % args.val_every == 0 or epoch == args.epochs):
            rows = evaluate_sequence(model, val_pairs, device, args.val_clicks, max_size=args.max_size)
            val_iou = float(np.mean([r["final_iou"] for r in rows]))
            improved = val_iou > best
            best = max(best, val_iou)
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "epoch": epoch, "best_iou": best, "config": vars(args)}
        torch.save(checkpoint, output / "model_latest.pth")
        if improved:
            torch.save(checkpoint, output / "model_best_iou.pth")
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_iou": val_iou}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(record, flush=True)


def evaluate_parser():
    parser = argparse.ArgumentParser(description="Sequential interactive evaluation with prediction-supervised online updates.")
    common_options(parser)
    parser.add_argument("--test-dir", required=True, help="Held-out directory with input/ and target/")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", nargs="+", choices=["base", "sta", "ltm", "oais"], default=["base", "oais"])
    parser.add_argument("--protocol", choices=["fixed", "noc"], default="fixed")
    parser.add_argument("--target-iou", type=float, default=0.90, help="Stopping threshold for the noc protocol")
    parser.add_argument("--max-clicks", type=positive_int, default=15)
    parser.add_argument("--order", choices=["original", "reverse", "random"], default="original")
    parser.add_argument("--order-seed", type=int, default=20261106)
    parser.add_argument("--num-runs", type=positive_int, default=1)
    parser.add_argument("--save-masks", action="store_true")
    for field in fields(OnlineSettings):
        parser.add_argument("--" + field.name.replace("_", "-"), type=field.type, default=field.default)
    return parser


def evaluate_main(argv=None):
    args = evaluate_parser().parse_args(argv)
    device = check_common(args)
    if not 0 < args.target_iou <= 1:
        raise ValueError("target-iou must be in (0, 1]")
    settings = OnlineSettings(**{f.name: getattr(args, f.name) for f in fields(OnlineSettings)})
    if (settings.sta_lr <= 0 or settings.ltm_lr <= 0 or settings.intensity_tau <= 0
            or settings.sta_steps < 1 or settings.ltm_steps < 1 or settings.min_component < 0
            or min(settings.click_weight, settings.negative_boost, settings.kd_weight,
                   settings.sta_anchor, settings.ltm_anchor) < 0):
        raise ValueError("Invalid online learning rates, steps, weights or thresholds.")
    pairs = discover_pairs(args.test_dir)
    if args.order == "reverse":
        pairs.reverse()
    elif args.order == "random":
        random.Random(args.order_seed).shuffle(pairs)
    checkpoint = read_checkpoint(args.checkpoint)
    output = prepare_output(args.output)
    save_json(output / "evaluation_config.json", {
        **vars(args), "online_settings": asdict(settings),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "sequence": [[str(p.resolve()), str(m.resolve())] for p, m in pairs],
        "run_seeds": [args.seed + i for i in range(args.num_runs)]})
    summaries = []
    for mode in dict.fromkeys(args.mode):
        for run in range(args.num_runs):
            seed_everything(args.seed + run)
            model = initialize_model(device)
            load_model_weights(model, checkpoint)
            adaptor = None if mode == "base" else OnlineAdaptor(model, settings)
            run_dir = output / f"{mode}_run{run+1}"
            run_dir.mkdir()
            def save_mask(stem, mask):
                cv2.imencode(".png", mask.astype(np.uint8) * 255)[1].tofile(str(run_dir / f"{stem}_mask.png"))
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            rows = evaluate_sequence(model, pairs, device, args.max_clicks, mode, adaptor,
                                     args.target_iou if args.protocol == "noc" else None,
                                     args.max_size, save_mask if args.save_masks else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            save_csv(run_dir / "per_image.csv", [{k: v for k, v in r.items() if k != "curve"} for r in rows])
            save_csv(run_dir / "per_click.csv", [{"image": r["image"], **c} for r in rows for c in r["curve"]])
            summary = {"mode": mode, "run": run + 1, "seed": args.seed + run, "images": len(rows),
                       "protocol": args.protocol, "runtime_seconds_including_io": elapsed,
                       "mean_final_iou": float(np.mean([r["final_iou"] for r in rows])),
                       "mean_final_dice": float(np.mean([r["final_dice"] for r in rows])),
                       "mean_subsequent_zero_iou": float(np.mean([r["zero_iou"] for r in rows[1:]])) if len(rows) > 1 else None}
            if args.protocol == "noc":
                summary.update(target_iou=args.target_iou,
                               capped_noc=float(np.mean([r["noc"] for r in rows])),
                               success_rate=float(np.mean([r["target_attained"] for r in rows])))
            else:
                for step in range(args.max_clicks + 1):
                    for metric in ("iou", "dice"):
                        summary[f"{metric}@{step}"] = float(np.mean([
                            r["curve"][min(step, len(r["curve"])-1)][metric] for r in rows]))
            save_json(run_dir / "summary.json", summary)
            summaries.append(summary)
            del adaptor, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    save_json(output / "summary.json", summaries)
    print(f"Evaluation complete: {output.resolve()}", flush=True)
