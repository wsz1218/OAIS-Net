import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np
import torch
from torch import nn

from seg.dataio import discover_pairs, load_pair, PatchDataset
from seg.interaction import add_click, empty_points, evaluate_sequence, next_click
from seg.inference.online import OnlineAdaptor, OnlineSettings


def write_pair(root, name, image, mask):
    for folder, array in (("input", image), ("target", mask)):
        path = root / folder
        path.mkdir(exist_ok=True, parents=True)
        cv2.imencode(".png", array)[1].tofile(str(path / (name + ".png")))


class ClickModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1))

    def forward(self, image, points, training=False):
        count = int((points[0, :, 0] >= 0).sum())
        logits = torch.full_like(image, -10)
        logits[..., :min(32, count * 8)] = 10
        return {"instances": logits}


class SpyAdaptor:
    def __init__(self):
        self.targets, self.sta_calls = [], 0

    def clear_momentum(self):
        pass

    def adapt_MI(self, image, points, clicks):
        self.sta_calls += 1

    def adapt_ltm(self, image, points, final_prediction):
        self.targets.append(final_prediction.clone())


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Module()
        self.feature_extractor.unet = nn.Module()
        self.feature_extractor.unet.encoder = nn.Conv2d(1, 2, 1)
        self.feature_extractor.unet.decoder = nn.Conv2d(2, 2, 1)
        self.feature_extractor.head = nn.Conv2d(2, 1, 1)
        self.maps_transform = nn.Conv2d(2, 2, 1)

    def forward(self, image, points, training=False):
        feature = self.feature_extractor
        return {"instances": feature.head(feature.unet.decoder(feature.unet.encoder(image)))}


class PublicWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_pairing_and_patch_on_small_image(self):
        image = np.full((20, 24), 120, np.uint8)
        write_pair(self.root, "image10", image, image)
        write_pair(self.root, "image2", image, image)
        pairs = discover_pairs(self.root)
        self.assertEqual(pairs[0][0].stem, "image2")
        x, y = PatchDataset(pairs, crop_size=32)[0]
        self.assertEqual(tuple(x.shape), (1, 32, 32))
        self.assertEqual(set(y.unique().tolist()), {1.0})
        (self.root / "target" / "image2.png").unlink()
        with self.assertRaises(ValueError):
            discover_pairs(self.root)

    def test_resize_preserves_binary_mask(self):
        mask = np.zeros((60, 90), np.uint8)
        mask[:, :45] = 255
        write_pair(self.root, "example", mask, mask)
        image, target = load_pair(discover_pairs(self.root)[0], 32)
        self.assertEqual(max(image.shape), 32)
        self.assertTrue(set(np.unique(target)).issubset({0, 1}))

    def test_points_allow_all_clicks_of_one_sign(self):
        points = empty_points(1, 15, "cpu")
        for i in range(15):
            add_click(points, (i, 0, 1))
        self.assertEqual(int((points[0, :15, 0] >= 0).sum()), 15)
        with self.assertRaises(ValueError):
            add_click(points, (16, 0, 1))

    def test_clicks_exclude_previous_locations(self):
        pred, target = np.zeros((32, 32), np.uint8), np.ones((32, 32), np.uint8)
        first = next_click(pred, target, [])
        second = next_click(pred, target, [first])
        self.assertNotEqual(first[:2], second[:2])

    def test_noc_uses_actual_click_count_and_prediction_target(self):
        for name in ("image1", "image2"):
            write_pair(self.root, name, np.zeros((32, 32), np.uint8), np.ones((32, 32), np.uint8))
        spy = SpyAdaptor()
        rows = evaluate_sequence(ClickModel(), discover_pairs(self.root), "cpu", 5,
                                 "oais", spy, target_iou=.7)
        self.assertEqual([r["noc"] for r in rows], [3, 3])
        self.assertEqual(spy.sta_calls, 6)
        self.assertEqual(len(spy.targets), 2)
        self.assertEqual(float(spy.targets[0].float().mean()), .75)
        self.assertTrue(all(r["target_attained"] for r in rows))

    def test_failure_is_capped_and_fixed_still_runs_budget(self):
        write_pair(self.root, "image", np.zeros((32, 32), np.uint8), np.ones((32, 32), np.uint8))
        pairs = discover_pairs(self.root)
        fail = evaluate_sequence(ClickModel(), pairs, "cpu", 2, target_iou=.9)[0]
        self.assertEqual(fail["noc"], 2)
        self.assertFalse(fail["target_attained"])
        fixed = evaluate_sequence(ClickModel(), pairs, "cpu", 3)[0]
        self.assertEqual(len(fixed["curve"]), 4)

    def test_parameter_freezing_and_supervision(self):
        torch.manual_seed(8)
        model = TinyModel()
        adaptor = OnlineAdaptor(model, OnlineSettings(sta_steps=1, ltm_steps=1))
        image, points = torch.rand(1, 1, 32, 32), empty_points(1, 2, "cpu")
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        adaptor.adapt_MI(image, points, [(16, 16, 1)])
        for n, p in model.named_parameters():
            if "encoder" in n or "maps_transform" in n:
                self.assertTrue(torch.equal(p, before[n]), n)
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters() if "head" in n))
        after_sta = {n: p.detach().clone() for n, p in model.named_parameters()}
        adaptor.adapt_ltm(image, points, torch.ones_like(image))
        for n, p in model.named_parameters():
            if "encoder" not in n:
                self.assertTrue(torch.equal(p, after_sta[n]), n)
        self.assertTrue(any(not torch.equal(p, after_sta[n]) for n, p in model.named_parameters() if "encoder" in n))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
