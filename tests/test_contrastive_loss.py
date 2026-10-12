"""InfoNCE по позитиву и намайненным негативам."""

import math
import unittest

import torch
from torch.nn import functional as F

from tgseqloc.training.trainer import contrastive_loss


class ContrastiveLossTests(unittest.TestCase):
    def test_equal_similarities_give_log_of_candidates(self) -> None:
        q = F.normalize(torch.ones(4, 8), dim=1)
        loss = contrastive_loss(q, q.clone(), q[:, None, :].repeat(1, 3, 1), 0.07)
        self.assertAlmostEqual(float(loss), math.log(4), places=5)

    def test_prefers_the_positive(self) -> None:
        g = torch.Generator().manual_seed(0)
        q = F.normalize(torch.randn(6, 16, generator=g), dim=1)
        negatives = F.normalize(torch.randn(6, 2, 16, generator=g), dim=1)
        near = contrastive_loss(q, q.clone(), negatives, 0.07)
        far = contrastive_loss(q, -q, negatives, 0.07)
        self.assertLess(float(near), float(far))
