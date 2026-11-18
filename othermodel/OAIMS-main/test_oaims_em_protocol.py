"""CPU tests only. No user dataset or trained weights are loaded."""
import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from oaims_em_common import (EMClicker, PaddedModel, collect_pairs, load_pair,
                             load_reference, metrics, run_image, set_seed)

ROOT = None


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(-2.0))

    def forward(self, images, points, labels, train_attention=True):
        score = self.bias + torch.zeros_like(images)
        if points is not None:
            click_map = torch.zeros_like(images)
            for i in range(points.shape[1]):
                y, x = points[0, i].long()
                click_map[0, 0, y, x] = 8 if labels[0, i, 0] else -8
            score = score + click_map
        return torch.cat([torch.zeros_like(score), score], dim=1)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        set_seed(7)
        torch.set_num_threads(1)

    def test_shared_mask_metrics(self):
        pred = np.array([[1, 0], [1, 0]], np.uint8)
        gt = np.array([[1, 1], [0, 0]], np.uint8)
        values = metrics(pred, gt)
        self.assertAlmostEqual(values['dice'], 2 * values['iou'] / (1 + values['iou']))

    def test_no_gt_supervision_or_future_click_leakage(self):
        model = ToyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        image = np.zeros((32, 32), np.uint8)
        gt = np.zeros_like(image)
        gt[5:27, 5:27] = 1
        losses, pi_targets = [], []
        def criterion(logits, target, points, labels, device):
            self.assertFalse(np.array_equal(target[0, 0].numpy(), gt))
            losses.append(target.clone())
            return torch.nn.functional.cross_entropy(logits, target[:, 0].long())
        def post(model, optimizer, criterion, images, final, first, device, **kwargs):
            pi_targets.append(final.clone())
        result = run_image(model, image, gt, torch.device('cpu'), 3, [.85], 'mi_pi',
                           criterion, optimizer, SimpleNamespace(post_interaction_adapt=post),
                           min_error_area=1)
        self.assertEqual(len(losses), 2)
        self.assertEqual(result['mi_updates'], 2)
        self.assertEqual(len(pi_targets), 1)
        np.testing.assert_array_equal(pi_targets[0][0, 0].numpy(), result['states'][3]['pred'])
        self.assertEqual(len(result['states'][1]['clicks']), 1)
        self.assertEqual(int(result['states'][1]['pred'].sum()), 1)
        self.assertFalse(result['noc'][.85]['success'])
        self.assertEqual(result['noc'][.85]['clicks'], 3)

    def test_modes_and_state_isolation(self):
        image = np.zeros((32, 32), np.uint8)
        gt = np.ones_like(image)
        for mode, expected in [('frozen', 0), ('pi', 1)]:
            model = ToyModel()
            saved = copy.deepcopy(model.state_dict())
            calls = []
            def post(*args, **kwargs):
                calls.append(True)
                with torch.no_grad():
                    model.bias.add_(1)
            result = run_image(model, image, gt, torch.device('cpu'), 2, [.9], mode,
                               None, None, SimpleNamespace(post_interaction_adapt=post))
            self.assertEqual(result['mi_updates'], 0)
            self.assertEqual(len(calls), expected)
            other = ToyModel()
            other.load_state_dict(saved)
            self.assertEqual(float(other.bias), -2.0)

    def test_data_pairing_and_mask_encodings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ['input', 'target']:
                (root / 'd/test' / name).mkdir(parents=True)
            for name, value in [('x10', 255), ('x2', 1)]:
                cv2.imwrite(str(root / f'd/test/input/{name}.png'), np.zeros((32, 32), np.uint8))
                cv2.imwrite(str(root / f'd/test/target/{name}.png'), np.full((32, 32), value, np.uint8))
            pairs = collect_pairs(root, 'd', 'test')
            self.assertEqual([p.name for p, _ in pairs], ['x2.png', 'x10.png'])
            self.assertTrue(all(load_pair(p, m)[1].all() for p, m in pairs))
            (root / 'd/test/target/x10.png').unlink()
            with self.assertRaises(ValueError):
                collect_pairs(root, 'd', 'test')

    def test_reference_trajectory_and_updates(self):
        if not ROOT:
            self.skipTest('Use --oaims-root for reference equivalence checks')
        training, online, _ = load_reference(ROOT)
        image = np.random.default_rng(3).integers(0, 256, (32, 32), dtype=np.uint8)
        gt = np.zeros((32, 32), np.uint8)
        gt[7:25, 8:24] = 1
        a = PaddedModel(training.UNet(3, 2))
        b = copy.deepcopy(a)
        oa = torch.optim.Adam(a.parameters(), lr=5e-5)
        ob = torch.optim.Adam(b.parameters(), lr=5e-5)
        from oaims_em_common import normalize_image
        tensor = torch.from_numpy(normalize_image(image)[None, None])
        target = torch.from_numpy(gt[None, None])
        set_seed(42)
        pred, first, dice = online.run_interactive_inference(
            a, tensor, target, torch.device('cpu'), 3, oa, online.build_loss())
        online.post_interaction_adapt(a, oa, online.build_loss(), tensor, pred, first, torch.device('cpu'))
        set_seed(42)
        result = run_image(b, image, gt, torch.device('cpu'), 3, [.85], 'mi_pi', online.build_loss(),
                           ob, online, click_policy='reference')
        np.testing.assert_array_equal(pred[0, 0].numpy(), result['states'][3]['pred'])
        np.testing.assert_allclose(dice, [r['dice'] for r in result['trace']], atol=1e-7)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-7)

    def test_grayscale_masks_match_baseline_threshold(self):
        masks = [np.array([[0, 1, 4, 7, 23, 89],
                           [112, 127, 128, 158, 250, 255]], dtype=np.uint8),
                 np.array([[0, 1], [1, 0]], dtype=np.uint8),
                 np.array([[0, 255], [255, 0]], dtype=np.uint8),
                 np.zeros((2, 2), dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as temp:
            image_path, mask_path = Path(temp) / 'image.png', Path(temp) / 'mask.png'
            for raw in masks:
                with self.subTest(values=np.unique(raw).tolist()):
                    cv2.imwrite(str(image_path), np.full(raw.shape, 90, dtype=np.uint8))
                    cv2.imwrite(str(mask_path), raw)
                    image, mask = load_pair(image_path, mask_path)
                    threshold = 0 if raw.max() <= 1 else 127
                    np.testing.assert_array_equal(mask, (raw > threshold).astype(np.uint8))
                    np.testing.assert_array_equal(image, np.full(raw.shape, 90, dtype=np.uint8))
                    self.assertEqual(mask.dtype, np.dtype(np.uint8))

    def test_nonmultiple_shape(self):
        if not ROOT:
            self.skipTest('Use --oaims-root for shape checks')
        training, _, _ = load_reference(ROOT)
        model = PaddedModel(training.UNet(3, 2))
        with torch.no_grad():
            output = model(torch.randn(1, 1, 35, 47), None, None)
        self.assertEqual(tuple(output.shape), (1, 2, 35, 47))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--oaims-root')
    args, extra = p.parse_known_args()
    ROOT = args.oaims_root
    unittest.main(argv=['test_oaims_em_protocol.py'] + extra)
