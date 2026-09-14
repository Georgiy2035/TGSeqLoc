"""Нормализация дескриптора против схлопывания: по умолчанию ничего не меняет, включённая — разводит кадры."""

import unittest

import torch
from torch_geometric.data import Batch

from tgseqloc.models import GATGraphEncoder
from tests.test_text_set import graph, EDGE_DIM, TEXT_DIM, CLASSES, EDGE_CLASSES


def model(**extra) -> GATGraphEncoder:
    torch.manual_seed(0)
    return GATGraphEncoder(
        in_dim=4, hidden_dim=16, n_layers=2, proj_dim=8, num_node_classes=CLASSES,
        node_emb_dim=8, num_edge_classes=EDGE_CLASSES, edge_emb_dim=8,
        edge_cont_dim=EDGE_DIM, dropout=0.0, heads=2, use_text_nodes=False,
        text_emb_dim=TEXT_DIM, use_edge_geometry=True, **extra,
    )


class DescriptorNormTests(unittest.TestCase):
    def test_defaults_leave_the_model_as_it_was(self) -> None:
        m = model()
        self.assertNotIn("descriptor_norm", m.init_args)
        self.assertNotIn("proj_bias", m.init_args)
        self.assertIsNone(m.pooled_norm)
        self.assertIsNone(m.output_norm)

    def test_normalization_draws_no_random_numbers(self) -> None:
        plain = model().state_dict()
        for mode in ("pooled", "output"):
            normed = model(descriptor_norm=mode).state_dict()
            for key, value in plain.items():
                self.assertTrue(torch.equal(value, normed[key]), (mode, key))

    def test_batch_statistics_remove_the_shared_component(self) -> None:
        graphs = Batch.from_data_list([graph(seed=i) for i in range(16)])
        mean_cos = lambda d: float(((d @ d.T).sum() - len(d)) / (len(d) * (len(d) - 1)))
        with torch.no_grad():
            plain = model().train()(graphs)
            normed = model(descriptor_norm="output", proj_bias=False).train()(graphs)
        self.assertLess(mean_cos(normed), mean_cos(plain))
        self.assertLess(abs(mean_cos(normed)), 0.2)

    def test_eval_uses_running_statistics(self) -> None:
        m = model(descriptor_norm="pooled").eval()
        one = Batch.from_data_list([graph(seed=1)])
        two = Batch.from_data_list([graph(seed=1), graph(seed=2)])
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(one)[0], m(two)[0], atol=1e-6))

    def test_unknown_mode_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            model(descriptor_norm="layer")
