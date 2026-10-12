"""Граф без узлов получает свой дескриптор и не сдвигает соседние кадры."""

import unittest

import torch
from torch_geometric.data import Batch, Data

from tgseqloc.models import GATGraphEncoder
from tests.test_text_set import graph, EDGE_DIM, TEXT_DIM, CLASSES, EDGE_CLASSES


def empty() -> Data:
    return Data(x=torch.zeros(0, 4), node_class=torch.zeros(0, dtype=torch.long), is_text=torch.zeros(0, dtype=torch.bool),
                text_emb=torch.zeros(0, TEXT_DIM), edge_index=torch.zeros(2, 0, dtype=torch.long),
                edge_attr=torch.zeros(0, EDGE_DIM), edge_label=torch.zeros(0, dtype=torch.long),
                is_text_edge=torch.zeros(0, dtype=torch.bool))


def model(**extra) -> GATGraphEncoder:
    torch.manual_seed(0)
    return GATGraphEncoder(in_dim=4, hidden_dim=16, n_layers=1, proj_dim=8, num_node_classes=CLASSES, node_emb_dim=8,
                           num_edge_classes=EDGE_CLASSES, edge_emb_dim=8, edge_cont_dim=EDGE_DIM, dropout=0.0, heads=2,
                           text_emb_dim=TEXT_DIM, use_edge_geometry=True, **extra).eval()


class EmptyGraphTests(unittest.TestCase):
    def test_every_form_returns_one_descriptor_per_graph(self) -> None:
        for kwargs in ({"use_text_nodes": False}, {"use_text_nodes": True}, {"use_text_nodes": True, "text_fusion": "additive"},
                       {"use_text_nodes": True, "text_fusion": "set"}):
            m = model(**kwargs)
            with torch.no_grad():
                together = m(Batch.from_data_list([graph(seed=1), empty(), graph(seed=2)]))
                alone = m(Batch.from_data_list([graph(seed=2)]))
            self.assertEqual(together.shape[0], 3, kwargs)
            self.assertTrue(torch.isfinite(together).all(), kwargs)
            self.assertTrue(torch.allclose(together[2], alone[0], atol=1e-6), kwargs)
