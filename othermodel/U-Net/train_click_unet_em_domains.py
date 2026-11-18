import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from em_utils import EMImageMaskDataset, RobotClicker, encode_click_maps
from model import ClickUNet, bce_dice_loss


DEFAULT_DOMAINS = [
    "domain_2_cells",
    "domain_3_muscle",
    "domain_4_fuzzy_blobs",
    "domain_6_Lucchi",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Train one plain Click U-Net model per EM domain for OAIS comparison."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root containing domain_xxx/train/input,target and domain_xxx/test/input,target.",
    )
    parser.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--output-root", default="experiments/click_unet_em_domains")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=15)
    parser.add_argument("--click-sigma", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_domain(dataset_root, domain):
    domain_dir = dataset_root / domain
    required = [
        domain_dir / "train" / "input",
        domain_dir / "train" / "target",
        domain_dir / "test" / "input",
        domain_dir / "test" / "target",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{domain} is missing required folders: {missing}")
    return domain_dir


def save_checkpoint(path, model, optimizer, epoch, args, domain, loss):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "domain": domain,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "train_loss": float(loss),
        },
        path,
    )


def build_click_input(images, clicks_per_sample, sigma, device):
    images_np = images.detach().cpu().numpy()
    click_channels = []
    for sample_idx, clicks in enumerate(clicks_per_sample):
        _, h, w = images_np[sample_idx].shape
        pos_map, neg_map = encode_click_maps((h, w), clicks, sigma=sigma)
        click_channels.append(np.stack([pos_map, neg_map], axis=0))
    click_tensor = torch.from_numpy(np.stack(click_channels).astype(np.float32)).to(device)
    return torch.cat([images, click_tensor], dim=1)


def simulate_interactive_clicks(model, images, masks, max_points, sigma, device):
    batch_size = images.shape[0]
    target_clicks = random.randint(1, max_points)
    gt_masks = (masks.detach().cpu().numpy()[:, 0] > 0.5).astype(np.uint8)
    pred_masks = np.zeros_like(gt_masks, dtype=np.uint8)
    clickers = [RobotClicker(gt_masks[i]) for i in range(batch_size)]
    clicks_per_sample = [[] for _ in range(batch_size)]

    model.eval()
    with torch.no_grad():
        for click_idx in range(target_clicks):
            for sample_idx, clicker in enumerate(clickers):
                click = clicker.get_next_click(pred_masks[sample_idx])
                if click is not None:
                    clicks_per_sample[sample_idx].append(click)

            if click_idx < target_clicks - 1:
                model_input = build_click_input(images, clicks_per_sample, sigma, device)
                logits = model(model_input)
                probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
                pred_masks = (probs >= 0.5).astype(np.uint8)

    model.train()
    return clicks_per_sample


def train_domain(domain, domain_dir, output_root, device, args):
    train_set = EMImageMaskDataset(
        image_dir=domain_dir / "train" / "input",
        mask_dir=domain_dir / "train" / "target",
        crop_size=args.crop_size,
        augment=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = ClickUNet(in_channels=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    exp_dir = output_root / domain
    ckpt_dir = exp_dir / "checkpoints"
    print("\n" + "=" * 80)
    print(f"Training Click U-Net domain: {domain}")
    print(f"Train samples: {len(train_set)}")
    print(f"Batches/epoch: {len(train_loader)}")
    print(f"Output: {exp_dir.resolve()}")
    print("=" * 80)

    if args.dry_run:
        return

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"{domain} Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            raw_images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            clicks = simulate_interactive_clicks(
                model=model,
                images=raw_images,
                masks=masks,
                max_points=args.num_points,
                sigma=args.click_sigma,
                device=device,
            )
            images = build_click_input(raw_images, clicks, args.click_sigma, device)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        avg_loss = float(np.mean(losses)) if losses else 0.0
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{domain}] Epoch {epoch:04d}/{args.epochs} | train_loss={avg_loss:.6f}")

        save_checkpoint(ckpt_dir / "last_checkpoint.pth", model, optimizer, epoch, args, domain, avg_loss)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(ckpt_dir / f"{epoch:04d}.pth", model, optimizer, epoch, args, domain, avg_loss)

    print(f"[DONE] {domain} checkpoint: {ckpt_dir / 'last_checkpoint.pth'}")


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    domain_dirs = {domain: validate_domain(dataset_root, domain) for domain in args.domains}

    print(f"Device: {device}")
    for domain in args.domains:
        train_domain(domain, domain_dirs[domain], output_root, device, args)


if __name__ == "__main__":
    main()
