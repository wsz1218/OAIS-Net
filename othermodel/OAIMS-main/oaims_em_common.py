"""EM data/evaluation glue. The upstream OAIMS implementation is not rewritten."""
import csv
import hashlib
import importlib
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from torch.utils.data import Dataset

DOMAINS = ['domain_2_cells', 'domain_3_muscle', 'domain_4_fuzzy_blobs', 'domain_6_Lucchi']
EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}
REFERENCE_FILES = ['models/unet_brain.py', 'models/losses.py', 'utils_brain.py',
                   'train_brain.py', 'online_adapt_brain.py', 'data/transforms.py']


def natural_key(value):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', str(value))]


def load_reference(root):
    root = Path(root).resolve()
    for name in REFERENCE_FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(f'Missing original OAIMS source: {root / name}')
    for name in ['models', 'data', 'utils_brain', 'train_brain', 'online_adapt_brain']:
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, '__file__', None):
            if root not in Path(existing.__file__).resolve().parents:
                raise RuntimeError(f'Conflicting module {name}: {existing.__file__}; use a fresh process.')
    sys.path.insert(0, str(root))
    try:
        training = importlib.import_module('train_brain')
        online = importlib.import_module('online_adapt_brain')
        utils = importlib.import_module('utils_brain')
    except ModuleNotFoundError as error:
        raise RuntimeError('Install the upstream requirements.txt in the OAIMS environment: '
                           + str(error)) from error
    return training, online, utils


def source_hashes(root):
    return {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest()
            for name in REFERENCE_FILES}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def domain_seed(seed, domain):
    return (seed + int(hashlib.sha256(domain.encode()).hexdigest()[:8], 16)) % (2 ** 32)


def resolve_device(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Select --device cpu for a smoke test.')
    return device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def checkpoint_load(path, device):
    # Only load your own training checkpoints or trusted author checkpoints.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def collect_pairs(root, domain, split, image_list=None):
    image_dir = Path(root) / domain / split / 'input'
    mask_dir = Path(root) / domain / split / 'target'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f'Required directories: {image_dir} and {mask_dir}')
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS],
                    key=lambda p: natural_key(p.name))
    if image_list is not None:
        names = [s.strip() for s in Path(image_list).read_text(encoding='utf-8-sig').splitlines()
                 if s.strip() and not s.lstrip().startswith('#')]
        if len(names) != len(set(names)) or any(Path(n).name != n for n in names):
            raise ValueError('Image list must contain unique file names, one per line.')
        lookup = {p.name: p for p in images}
        missing = set(names) - set(lookup)
        if missing:
            raise FileNotFoundError(f'Image-list entries not found in {image_dir}: {sorted(missing)}')
        images = [lookup[name] for name in names]
    masks = {}
    for path in mask_dir.iterdir():
        if path.suffix.lower() in EXTENSIONS:
            masks.setdefault(path.stem, []).append(path)
    pairs = []
    for path in images:
        candidates = masks.get(path.stem, [])
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one mask for {path.name}, found {candidates}')
        pairs.append((path, candidates[0]))
    if not pairs:
        raise ValueError(f'No image-mask pairs in {image_dir}')
    if len({p.stem for p, _ in pairs}) != len(pairs):
        raise ValueError(f'Duplicate image stems in {image_dir}')
    return pairs


def load_pair(image_path, mask_path):
    """Read a binary EM target, matching the SAM2/Click U-Net gray threshold."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError(f'Cannot read {image_path} / {mask_path}')
    if image.shape != mask.shape:
        raise ValueError(f'Image-mask size mismatch: {image_path}')
    if mask.max() <= 1:
        mask = (mask > 0).astype(np.uint8)
    else:
        # Stored 8-bit targets can contain intermediate gray levels at boundaries.
        mask = (mask > 127).astype(np.uint8)
    return image, mask


def normalize_image(image):
    array = image.astype(np.float32)
    return (array - array.mean()) / (array.std() + 1e-8)


class PaddedModel(torch.nn.Module):
    """Pad only for U-Net geometry; crop logits before any metric or loss."""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, images, points, labels, train_attention=True):
        h, w = images.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16
        if h + ph < 32 or w + pw < 32:
            raise ValueError('Images must be at least 32 pixels per dimension for InstanceNorm.')
        padded = F.pad(images, (0, pw, 0, ph), mode='replicate') if ph or pw else images
        return self.network(padded, points, labels, train_attention=train_attention)[..., :h, :w]


class EMTrainingDataset(Dataset):
    def __init__(self, pairs, crop_size, transform):
        self.pairs, self.crop_size, self.transform = pairs, crop_size, transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image, mask = load_pair(*self.pairs[index])
        n = self.crop_size
        h, w = image.shape
        if h < n or w < n:
            image = cv2.copyMakeBorder(image, 0, max(0, n-h), 0, max(0, n-w), cv2.BORDER_REPLICATE)
            mask = cv2.copyMakeBorder(mask, 0, max(0, n-h), 0, max(0, n-w), cv2.BORDER_CONSTANT, value=0)
        h, w = image.shape
        top, left = random.randint(0, h-n), random.randint(0, w-n)
        image, mask = image[top:top+n, left:left+n], mask[top:top+n, left:left+n]
        if self.transform is not None:
            result = self.transform(image=image[..., None], mask=mask)
            image, mask = result['image'].reshape(n, n), result['mask']
        return (torch.from_numpy(np.ascontiguousarray(normalize_image(image)[None])),
                torch.from_numpy(np.ascontiguousarray(mask[None].astype(np.int64))))


def metrics(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return {'iou': float(intersection / union) if union else 1.0,
            'dice': float(2 * intersection / denom) if denom else 1.0}


def hd95(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return 50.0  # Compatibility with the existing EM evaluators; unit: pixels.
    kernel = np.ones((3, 3), bool)
    pa = pred ^ binary_erosion(pred, structure=kernel, border_value=0)
    ga = gt ^ binary_erosion(gt, structure=kernel, border_value=0)
    distances = np.concatenate([distance_transform_edt(~ga)[pa], distance_transform_edt(~pa)[ga]])
    return float(np.percentile(distances, 95))


class EMClicker:
    def __init__(self, gt, min_area=5, policy='distance'):
        self.gt = gt
        self.clicked = np.zeros_like(gt, bool)
        self.min_area = min_area
        self.policy = policy

    def next(self, pred):
        if self.policy == 'reference':
            # Match upstream: separate positive/negative 8-connected regions,
            # random interior location, no explicit exclusion of previous clicks.
            components = []
            for cls in [1, 0]:
                error = ((self.gt == cls) & (pred != cls)).astype(np.uint8)
                count, regions, stats, _ = cv2.connectedComponentsWithStats(error, 8, cv2.CV_32S)
                for i in range(1, count):
                    indices = np.argwhere(regions == i)
                    y, x = indices[random.choice(range(len(indices)))]
                    components.append((int(stats[i, cv2.CC_STAT_AREA]), int(y), int(x), cls))
            if not components:
                return None
            components.sort(key=lambda item: item[0])
            _, y, x, cls = components[-1]
            return y, x, cls
        error = (pred != self.gt) & ~self.clicked
        regions, count = label(error)
        if not count:
            return None
        sizes = np.bincount(regions.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        if sizes[largest] < self.min_area:
            return None
        distance = cv2.distanceTransform((regions == largest).astype(np.uint8), cv2.DIST_L2, 5)
        distance[self.clicked] = 0
        if distance.max() < 1:
            return None
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        self.clicked[y, x] = True
        return int(y), int(x), int(self.gt[y, x])


def click_tensors(clicks, device):
    points = torch.tensor([[[y, x] for y, x, _ in clicks]], dtype=torch.float32, device=device)
    labels = torch.tensor([[[c] for _, _, c in clicks]], dtype=torch.float32, device=device)
    return points, labels


def predict_from_logits(logits):
    return logits.detach().softmax(1).argmax(1).unsqueeze(1)


def run_image(model, image, gt, device, max_clicks, targets, mode, criterion,
              optimizer, online_module, click_policy='distance', min_error_area=5,
              correction_clicks=10):
    """Metrics precede MI; MI's next-click target never reaches earlier metrics.

    GT is confined to the clicker and metrics. PI receives only final prediction.
    NoC is first passage on the fixed-budget trajectory, not a separate stopped run.
    """
    tensor = torch.from_numpy(normalize_image(image)[None, None]).to(device)
    clicker = EMClicker(gt, min_error_area, click_policy)
    model.train(mode != 'frozen')
    clicks, trace, states = [], [], {}
    infer_time = mi_time = pi_time = 0.0
    mi_updates = 0
    mi_forwards = 0
    synchronize(device)
    started = time.perf_counter()
    first = clicker.next(np.zeros_like(gt))
    first_click = None
    final = None
    if first is not None:
        clicks.append(first)
        first_click = tuple(t.clone() for t in click_tensors(clicks, device))
        for k in range(1, max_clicks + 1):
            pts, labels = click_tensors(clicks, device)
            synchronize(device)
            t = time.perf_counter()
            with torch.set_grad_enabled(mode == 'mi_pi' and k < max_clicks):
                logits = model(tensor, pts, labels, train_attention=True)
            final = predict_from_logits(logits)
            synchronize(device)
            infer_time += time.perf_counter() - t
            pred = final[0, 0].cpu().numpy().astype(np.uint8)
            values = metrics(pred, gt)
            trace.append({'clicks': k, **values})
            states[k] = {'pred': pred.copy(), 'clicks': list(clicks), **values}
            if k == max_clicks:
                if click_policy == 'reference':
                    clicker.next(pred)  # Upstream consumes RNG for an unused last correction.
                break
            next_click = clicker.next(pred)
            if next_click is None:
                break
            if mode == 'mi_pi':
                synchronize(device)
                t = time.perf_counter()
                next_pts, next_labels = click_tensors(clicks + [next_click], device)
                with torch.no_grad():
                    target = predict_from_logits(model(tensor, next_pts, next_labels, train_attention=True))
                new_pts, new_labels = click_tensors([next_click], device)
                loss = criterion(logits, target.float(), new_pts, new_labels, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite MI loss')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                synchronize(device)
                mi_time += time.perf_counter() - t
                mi_updates += 1
                mi_forwards += 1
            clicks.append(next_click)
    else:
        # Do not fabricate the upstream fallback (0, 0, background) click.
        synchronize(device)
        t = time.perf_counter()
        with torch.no_grad():
            final = predict_from_logits(model(tensor, None, None))
        synchronize(device)
        infer_time += time.perf_counter() - t
        pred = final[0, 0].cpu().numpy().astype(np.uint8)
        values = metrics(pred, gt)
        trace.append({'clicks': 0, **values})
        states[0] = {'pred': pred.copy(), 'clicks': [], **values}
    last = states[trace[-1]['clicks']]
    for k in range(1, max_clicks + 1):
        states.setdefault(k, last)
    noc = {}
    for target in targets:
        hits = [row['clicks'] for row in trace if row['iou'] >= target]
        noc[target] = {'clicks': min(hits) if hits else max_clicks, 'success': bool(hits)}
    if mode in ('mi_pi', 'pi') and first_click is not None:
        synchronize(device)
        t = time.perf_counter()
        online_module.post_interaction_adapt(model, optimizer, criterion, tensor,
                                             final.detach().clone(), first_click, device,
                                             num_correction_clicks=correction_clicks)
        synchronize(device)
        pi_time += time.perf_counter() - t
    synchronize(device)
    total = time.perf_counter() - started
    return {'states': states, 'trace': trace, 'noc': noc, 'num_clicks': len(clicks),
            'inference_ms': infer_time * 1000, 'mi_ms': mi_time * 1000,
            'pi_ms': pi_time * 1000, 'image_compute_wall_ms': total * 1000,
            'mi_updates': mi_updates, 'pi_updates': 2 if pi_time else 0,
            'mi_target_forwards': mi_forwards}


def write_csv(path, rows, fields=None):
    if not rows:
        raise ValueError(f'No rows for {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def save_visual(path, image, gt, state, title):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y, x, cls in state['clicks']:
        cv2.circle(image_rgb, (x, y), 3, (0, 255, 0) if cls else (0, 0, 255), -1)
    pred = state['pred'].astype(bool)
    target = gt.astype(bool)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
    for region, color in [(pred & target, (0, 255, 0)),
                          (pred & ~target, (0, 0, 255)), (~pred & target, (255, 0, 0))]:
        overlay[region] = 0.5 * overlay[region] + 0.5 * np.array(color)
    gt_rgb = cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR)
    panel = np.concatenate([image_rgb, gt_rgb, overlay.astype(np.uint8)], axis=1)
    panel = cv2.copyMakeBorder(panel, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), panel):
        raise IOError(f'Cannot write {path}')
