from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from tgseqloc.data.schema import EDGE_FEATURE_DIM, graph_is_compatible, sanitize_graph
from tgseqloc.evaluation.retrieval import RecallAtK, Retriever, evaluate_retrieval
from tgseqloc.models.gat import GATGraphEncoder


def sample_graph(offset: float = 0.0) -> Data:
    return Data(
        x=torch.tensor(
            [[0.1 + offset, 0.2, 0.3, 0.4], [0.5, 0.6 + offset, 0.2, 0.2]]
        ),
        node_class=torch.tensor([1, 0]),
        is_text=torch.tensor([False, True]),
        text_emb=torch.tensor([[0.0] * 4, [1.0, 0.0, 0.0, 0.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.ones((2, EDGE_FEATURE_DIM)),
        edge_label=torch.tensor([1, 0]),
        edge_u_class=torch.tensor([1, 0]),
        edge_v_class=torch.tensor([0, 1]),
        is_text_edge=torch.tensor([False, True]),
        preprocess_fingerprint="fingerprint",
    )


class GraphSchemaTests(unittest.TestCase):
    def test_sanitize_removes_invalid_edges_and_repairs_nonfinite_fields(self) -> None:
        graph = sample_graph()
        graph.x[0, 0] = float("nan")
        graph.edge_index = torch.tensor([[0, 9, 1], [1, 0, -1]])
        graph.edge_attr = torch.full((1, 3), float("nan"))
        graph.edge_label = torch.tensor([1])
        clean = sanitize_graph(graph, text_embedding_dim=4)
        self.assertEqual(tuple(clean.edge_index.shape), (2, 1))
        self.assertEqual(tuple(clean.edge_attr.shape), (1, EDGE_FEATURE_DIM))
        self.assertTrue(torch.isfinite(clean.x).all())
        self.assertTrue(torch.isfinite(clean.edge_attr).all())
        self.assertEqual(clean.edge_label.tolist(), [1])

    def test_compatibility_checks_schema_fingerprint_and_dimensions(self) -> None:
        graph = sample_graph()
        self.assertTrue(graph_is_compatible(graph, "fingerprint", 4))
        self.assertFalse(graph_is_compatible(graph, "other", 4))
        self.assertFalse(graph_is_compatible(graph, "fingerprint", 3))
        graph.edge_index[0, 0] = 99
        self.assertFalse(graph_is_compatible(graph, "fingerprint", 4))


class GATTests(unittest.TestCase):
    def test_forward_batch_normalization_attention_and_backward(self) -> None:
        torch.manual_seed(3)
        model = GATGraphEncoder(
            in_dim=4,
            hidden_dim=8,
            n_layers=2,
            proj_dim=5,
            num_node_classes=2,
            node_emb_dim=4,
            num_edge_classes=2,
            edge_emb_dim=4,
            edge_cont_dim=EDGE_FEATURE_DIM,
            dropout=0.0,
            heads=1,
            use_text_nodes=True,
            text_emb_dim=4,
            use_edge_geometry=True,
        )
        batch = Batch.from_data_list([sample_graph(), sample_graph(0.1)])
        descriptors, attention = model(batch, return_attn=True)
        self.assertEqual(tuple(descriptors.shape), (2, 5))
        torch.testing.assert_close(
            descriptors.norm(dim=1), torch.ones(2), atol=1e-5, rtol=1e-5
        )
        self.assertIsNotNone(attention)
        loss = descriptors.square().sum() + descriptors[:, 0].sum()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(any(value is not None and torch.isfinite(value).all() for value in gradients))

    def test_rejects_missing_text_fields(self) -> None:
        model = GATGraphEncoder(
            4, hidden_dim=4, n_layers=1, proj_dim=2,
            num_node_classes=2, node_emb_dim=2, use_text_nodes=True, text_emb_dim=4,
            edge_cont_dim=EDGE_FEATURE_DIM, heads=1,
        )
        graph = sample_graph()
        del graph.text_emb
        with self.assertRaisesRegex(ValueError, "requires is_text and text_emb"):
            model(graph)


class RetrievalTests(unittest.TestCase):
    def test_cosine_retrieval_with_ids_and_validation(self) -> None:
        database = np.array([[2, 0], [0, 3], [-1, 0]], dtype=np.float32)
        retriever = Retriever(dimension=2).fit(database, ids=["east", "north", "west"])
        scores, ids = retriever.search([[1, 0], [0, 2]], 2)
        self.assertEqual(ids[:, 0].tolist(), ["east", "north"])
        np.testing.assert_allclose(scores[:, 0], [1.0, 1.0], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "descriptor dimension"):
            retriever.search([[1, 2, 3]], 1)
        with self.assertRaisesRegex(ValueError, "ids must align"):
            Retriever().fit(database, ids=[1])
        with self.assertRaisesRegex(RuntimeError, "fit must be called"):
            Retriever().search([[1, 0]], 1)

    def test_recall_at_k_and_unified_evaluation(self) -> None:
        nearest = np.array([[0, 1], [0, 1], [2, 1]])
        positives = {10: [0], "11": [1], 12: [1]}
        metric = RecallAtK([1, 2], percentage=False)
        self.assertEqual(
            metric(nearest, positives, [10, 11, 12]),
            {"R@1": 1 / 3, "R@2": 1.0},
        )
        recalls, result = evaluate_retrieval(
            np.eye(3, dtype=np.float32),
            np.eye(3, dtype=np.float32),
            {0: [0], 1: [1], 2: [2]},
            recall_values=[1],
        )
        self.assertEqual(recalls, {"R@1": 100.0})
        np.testing.assert_array_equal(result[:, 0], [0, 1, 2])
        self.assertEqual(RecallAtK(1)(np.empty((0, 1)), {}), {"R@1": 0.0})


if __name__ == "__main__":
    unittest.main()
