"""Evaluate original OAIMS MI+PI with the existing EM benchmark's outer protocol."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from oaims_em_common import (DOMAINS, PaddedModel, checkpoint_load, collect_pairs,
                             domain_seed, hd95, load_pair, load_reference, mean_std,
                             resolve_device, run_image, save_visual, set_seed,
                             source_hashes, write_csv, write_json)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oaims-root', default=str(Path(__file__).resolve().parent))
    p.add_argument('--dataset-root', required=True)
    p.add_argument('--test-split', default='test')
    p.add_argument('--domains', nargs='+', default=DOMAINS)
    p.add_argument('--image-list-dir', help='Optional DOMAIN.txt with exact ordered test filenames per domain.')
    p.add_argument('--checkpoint-root', default='experiments/oaims_em_domains')
    p.add_argument('--checkpoint-pattern', default='{domain}/checkpoints/last_checkpoint.pth')
    p.add_argument('--output-root', default='results')
    p.add_argument('--modes', nargs='+', choices=['mi_pi', 'frozen', 'pi'], default=['mi_pi', 'frozen'])
    p.add_argument('--max-clicks', type=int, default=15)
    p.add_argument('--target-clicks', default='1,3,5,10,15')
    p.add_argument('--target-ious', default='0.85,0.88,0.90')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--pi-correction-clicks', type=int, default=10)
    p.add_argument('--click-policy', choices=['distance', 'reference'], default='distance')
    p.add_argument('--min-error-area', type=int, default=5)
    p.add_argument('--runs', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--visualize-max-images', type=int, default=3)
    p.add_argument('--dry-run', action='store_true', help='Check image/mask pairs and checkpoint paths only.')
    return p


def method_name(mode):
    return {'mi_pi': 'OAIMS', 'frozen': 'OAIMS-frozen', 'pi': 'OAIMS-PI-only'}[mode]


def evaluate_domain(args, domain, pairs, checkpoint_path, output, device, refs):
    training, online, _ = refs
    checkpoint = checkpoint_load(checkpoint_path, 'cpu')
    weights = checkpoint.get('model_state_dict')
    if weights is None:
        raise ValueError(f'Expected author-compatible model_state_dict in {checkpoint_path}')
    if checkpoint.get('domain', domain) != domain:
        raise ValueError(f'Checkpoint domain mismatch for {domain}')
    if 'domain' not in checkpoint:
        print(f'WARNING: {checkpoint_path} has no EM domain metadata; verify training provenance.', flush=True)
    saved_hashes = checkpoint.get('reference_sha256')
    if saved_hashes and saved_hashes != source_hashes(args.oaims_root):
        raise ValueError('Upstream source differs from training; use the recorded source version')
    domain_dir = output / domain
    domain_dir.mkdir()
    write_csv(domain_dir / 'sequence_manifest.csv', [
        {'index': i, 'image': str(p), 'mask': str(m)} for i, (p, m) in enumerate(pairs, 1)])
    fixed, noc_rows, trajectories, times = [], [], [], []
    for mode in args.modes:
        method = method_name(mode)
        for run in range(1, args.runs + 1):
            seed = domain_seed(args.seed + run - 1, domain)
            set_seed(seed)
            model = PaddedModel(training.UNet(in_c=3, out_c=2)).to(device)
            model.network.load_state_dict(weights, strict=True)
            criterion = online.build_loss() if mode != 'frozen' else None
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if mode != 'frozen' else None
            print(f'[{domain}] {method} run={run} images={len(pairs)} seed={seed}', flush=True)
            for index, (image_path, mask_path) in enumerate(pairs, 1):
                image, gt = load_pair(image_path, mask_path)
                result = run_image(model, image, gt, device, args.max_clicks, args.target_ious,
                                   mode, criterion, optimizer, online, args.click_policy,
                                   args.min_error_area, args.pi_correction_clicks)
                identity = {'method': method, 'dataset': domain, 'run': run, 'seed': seed,
                            'sequence_index': index, 'image': image_path.name}
                for k in args.target_clicks:
                    state = result['states'][k]
                    fixed.append({**identity, 'clicks': k, 'actual_clicks': len(state['clicks']),
                                  'iou': state['iou'], 'dice': state['dice'],
                                  'hd95': hd95(state['pred'], gt)})
                    if run == 1 and index <= args.visualize_max_images:
                        title = f'{method} | {k} clicks | IoU {state["iou"]*100:.2f}% | Dice {state["dice"]*100:.2f}%'
                        save_visual(domain_dir / mode / 'visuals' / f'{image_path.stem}_step{k}.png',
                                    image, gt, state, title)
                for target, value in result['noc'].items():
                    noc_rows.append({**identity, 'target_iou': target, 'noc': value['clicks'],
                                     'success': int(value['success'])})
                for row in result['trace']:
                    trajectories.append({**identity, **row})
                timing = {k: v for k, v in result.items() if k not in ['states', 'trace', 'noc']}
                times.append({**identity, **timing})
                final = result['states'][args.max_clicks]
                print(f'  {index}/{len(pairs)} {image_path.name}: IoU@{args.max_clicks}={final["iou"]:.4f}, '
                      f'Dice={final["dice"]:.4f}, MI={result["mi_updates"]}, PI={result["pi_updates"]}', flush=True)
                # Save after each image so an interrupted sequence is inspectable.
                write_csv(domain_dir / 'oaims_per_image.csv', fixed)
                write_csv(domain_dir / 'oaims_per_image_noc.csv', noc_rows)
                write_csv(domain_dir / 'oaims_click_trajectories.csv', trajectories)
                write_csv(domain_dir / 'oaims_timing.csv', times)
            del model, optimizer, criterion
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    summaries, noc_summaries = [], []
    for mode in args.modes:
        method = method_name(mode)
        for k in args.target_clicks:
            rows = [r for r in fixed if r['method'] == method and r['clicks'] == k]
            row = {'method': method, 'dataset': domain, 'clicks': k,
                   'num_images': len(pairs), 'num_runs': args.runs, 'num_observations': len(rows)}
            for metric in ['iou', 'dice', 'hd95']:
                row[metric + '_mean'], row[metric + '_std'] = mean_std([r[metric] for r in rows])
                run_means = [np.mean([r[metric] for r in rows if r['run'] == run]) for run in range(1, args.runs + 1)]
                row[metric + '_run_mean_std'] = float(np.std(run_means))
            summaries.append(row)
        for target in args.target_ious:
            rows = [r for r in noc_rows if r['method'] == method and r['target_iou'] == target]
            mean, std = mean_std([r['noc'] for r in rows])
            failures = sum(not r['success'] for r in rows)
            noc_summaries.append({'method': method, 'dataset': domain, 'target_iou': target,
                                 'noc_mean': mean, 'noc_std': std, 'num_images': len(pairs),
                                 'num_runs': args.runs, 'num_observations': len(rows),
                                 'num_failures': failures, 'failure_rate': failures / len(rows)})
    write_csv(domain_dir / 'oaims_summary_fixed_clicks.csv', summaries)
    write_csv(domain_dir / 'oaims_summary_noc.csv', noc_summaries)
    return summaries, noc_summaries


def main():
    args = parser().parse_args()
    args.target_clicks = [int(s) for s in args.target_clicks.split(',')]
    args.target_ious = [float(s) for s in args.target_ious.split(',')]
    if args.max_clicks < 1 or not args.target_clicks or any(k < 1 or k > args.max_clicks for k in args.target_clicks):
        raise ValueError('Target clicks must be in [1, max-clicks]')
    if not args.target_ious or any(not 0 < t <= 1 for t in args.target_ious):
        raise ValueError('Target IoUs must be in (0, 1]')
    for values in [args.domains, args.modes, args.target_clicks, args.target_ious]:
        if len(values) != len(set(values)):
            raise ValueError('Duplicated domain, mode, click count or target')
    if args.runs < 1 or args.lr <= 0 or args.pi_correction_clicks < 1 or args.min_error_area < 1:
        raise ValueError('Invalid run/lr/correction-click/min-area setting')
    inputs = []
    for domain in args.domains:
        image_list = Path(args.image_list_dir) / f'{domain}.txt' if args.image_list_dir else None
        pairs = collect_pairs(args.dataset_root, domain, args.test_split, image_list)
        checkpoint = Path(args.checkpoint_root).resolve() / args.checkpoint_pattern.format(domain=domain)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for pair in pairs:
            load_pair(*pair)
        inputs.append((domain, pairs, checkpoint))
        print(f'{domain}: {len(pairs)} test pairs; checkpoint={checkpoint}', flush=True)
    if args.dry_run:
        return
    refs = load_reference(args.oaims_root)
    device = resolve_device(args.device)
    output = Path(args.output_root).resolve() / datetime.now().strftime('OAIMS_EM_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'evaluation_config.json', {
        'args': vars(args), 'reference_sha256': source_hashes(args.oaims_root),
        'protocol': 'fixed-budget trajectory; metrics recorded before the next-click MI; PI once after final prediction',
        'noc': 'first observed target crossing on fixed-budget sequence; failed cases capped at max_clicks',
        'std': 'population std across image-run observations; run_mean_std separately reported',
        'hd95': 'pixel distances; empty-mask penalty 50 for compatibility with existing EM evaluators',
        'loss': 'original author CCG argument order retained',
        'checkpoint_selection': 'explicit checkpoint path, never selected on test metrics'})
    fixed, noc = [], []
    for domain, pairs, checkpoint in inputs:
        a, b = evaluate_domain(args, domain, pairs, checkpoint, output, device, refs)
        fixed.extend(a)
        noc.extend(b)
    write_csv(output / 'oaims_all_domains_fixed_clicks.csv', fixed)
    write_csv(output / 'oaims_all_domains_noc.csv', noc)
    print(f'DONE: {output}', flush=True)


if __name__ == '__main__':
    main()
