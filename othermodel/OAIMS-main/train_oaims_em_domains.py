"""Train the author OAIMS U-Net on each EM domain, without test-set selection."""
import argparse
from datetime import datetime
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from oaims_em_common import (DOMAINS, EMTrainingDataset, PaddedModel, checkpoint_load,
                             collect_pairs, domain_seed, load_pair, load_reference,
                             resolve_device, run_image, seed_worker, set_seed,
                             source_hashes, write_csv, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oaims-root', default=str(Path(__file__).resolve().parent))
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--domains', nargs='+', default=DOMAINS)
    p.add_argument('--output-root', default='experiments/oaims_em_domains')
    p.add_argument('--train-split', default='train')
    p.add_argument('--val-root', help='Optional independent validation root; never use the held-out test root.')
    p.add_argument('--val-split', default='val')
    p.add_argument('--val-every', type=int, default=3)
    p.add_argument('--val-clicks', type=int, default=6)
    p.add_argument('--epochs', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--num-points', type=int, default=10, help='Offline training click limit; independent of test budget.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='Check data only; does not create/train a model.')
    return p


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda']])


def lr_multiplier(step):
    # Author schedule is based on optimizer steps, not epochs.
    return 1.0 if step < 45000 else 0.5 if step < 55000 else 0.25 if step < 65000 else 0.1 if step < 75000 else 0.01


def train_domain(args, domain, device, refs):
    training, online, _ = refs
    seed = domain_seed(args.seed, domain)
    set_seed(seed)
    pairs = collect_pairs(args.dataset_root, domain, args.train_split)
    val_pairs = collect_pairs(args.val_root, domain, args.val_split) if args.val_root else []
    train_paths = {p.resolve() for p, _ in pairs}
    if any(p.resolve() in train_paths for p, _ in val_pairs):
        raise ValueError('Training and validation paths overlap')
    output = Path(args.output_root).resolve() / domain
    checkpoint_dir = output / 'checkpoints'
    last_path = checkpoint_dir / 'last_checkpoint.pth'
    if output.exists() and not args.resume:
        raise FileExistsError(f'{output} exists. Use --resume or choose a new output root.')
    if args.resume and not last_path.exists():
        raise FileNotFoundError(f'Resume checkpoint not found: {last_path}')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    from data.transforms import get_brain_interactive_transforms
    transform = get_brain_interactive_transforms()
    dataset = EMTrainingDataset(pairs, args.crop_size, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, worker_init_fn=seed_worker,
                        pin_memory=device.type == 'cuda')
    model = PaddedModel(training.UNet(in_c=3, out_c=2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    criterion = training.build_loss()
    start_epoch, best = 0, -1.0
    hashes = source_hashes(args.oaims_root)
    if args.resume:
        saved = checkpoint_load(last_path, device)
        if saved['domain'] != domain or saved['reference_sha256'] != hashes:
            raise ValueError('Resume domain/reference code mismatch')
        for key in ['crop_size', 'num_points', 'lr', 'seed', 'batch_size', 'workers', 'dataset_root']:
            if saved['args'][key] != vars(args)[key]:
                raise ValueError(f'Resume setting changed: {key}')
        model.network.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        start_epoch, best = saved['epoch'], saved['best_val_dice']
        restore_rng(saved['rng_state'])
    manifest = [{'split': 'train', 'image': str(p), 'mask': str(m)} for p, m in pairs]
    manifest += [{'split': 'val', 'image': str(p), 'mask': str(m)} for p, m in val_pairs]
    write_csv(output / 'data_manifest.csv', manifest)
    write_json(output / 'training_config.json', {'args': vars(args), 'domain_seed': seed,
               'reference_sha256': hashes, 'loss': 'author DF+CCG, including original CCG argument order',
               'selection': 'best validation Dice@6 (or configured val clicks)' if val_pairs else 'last epoch; no validation set'})
    history_path = output / 'training_history.csv'
    if args.resume and history_path.exists():
        import csv
        with history_path.open(encoding='utf-8-sig', newline='') as file:
            history = list(csv.DictReader(file))
        history = [r for r in history if int(r['epoch']) <= start_epoch]
    else:
        history = []
    print(f'[{domain}] train={len(pairs)}, val={len(val_pairs)}, seed={seed}, start={start_epoch}', flush=True)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                points, point_labels = training.generate_training_points_batch(
                    images, labels, model, device, args.num_points)
            loss = training.train_step(images, labels, points, point_labels,
                                       model, optimizer, criterion, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss in {domain}, epoch {epoch}')
            scheduler.step()
            losses.append(float(loss.detach()))
        val_dice, improved = None, False
        if val_pairs and (epoch % args.val_every == 0 or epoch == args.epochs):
            state = rng_state()
            set_seed(seed)
            values = []
            for image_path, mask_path in val_pairs:
                image, gt = load_pair(image_path, mask_path)
                result = run_image(model, image, gt, device, args.val_clicks, [], 'frozen',
                                   None, None, online, click_policy='reference', min_error_area=1)
                values.append(result['states'][args.val_clicks]['dice'])
            val_dice = float(np.mean(values))
            improved = val_dice > best
            best = max(best, val_dice)
            restore_rng(state)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        'val_dice': val_dice, 'lr': optimizer.param_groups[0]['lr']})
        checkpoint = {'model_state_dict': model.network.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch,
                      'best_val_dice': best, 'domain': domain, 'args': vars(args),
                      'reference_sha256': hashes, 'rng_state': rng_state(),
                      'created_at': datetime.now().isoformat()}
        temporary = checkpoint_dir / 'last_checkpoint.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(last_path)
        if improved:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:04d}.pth')
        write_csv(history_path, history)
        print(f'[{domain}] {epoch}/{args.epochs} loss={np.mean(losses):.6f} val_dice={val_dice}', flush=True)


def main():
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.num_points, args.save_every, args.val_every, args.val_clicks) < 1:
        raise ValueError('Epoch/batch/click/save/validation settings must be positive')
    if args.crop_size < 32 or args.crop_size % 16:
        raise ValueError('--crop-size must be a multiple of 16 and >= 32')
    if args.workers < 0 or args.lr <= 0 or len(args.domains) != len(set(args.domains)):
        raise ValueError('Invalid workers, lr or duplicated domains')
    if args.train_split == 'test' or (args.val_root and args.val_split == 'test'):
        raise ValueError('Do not train/select checkpoints on the held-out test split')
    if args.dry_run:
        for domain in args.domains:
            pairs = collect_pairs(args.dataset_root, domain, args.train_split)
            for pair in pairs:
                load_pair(*pair)
            val = collect_pairs(args.val_root, domain, args.val_split) if args.val_root else []
            for pair in val:
                load_pair(*pair)
            print(f'{domain}: train={len(pairs)}, validation={len(val)}; OK')
        return
    refs = load_reference(args.oaims_root)
    device = resolve_device(args.device)
    for domain in args.domains:
        train_domain(args, domain, device, refs)


if __name__ == '__main__':
    main()
