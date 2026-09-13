"""Текст как слагаемое объекта: кадр без текста обязан считаться как без текстового слоя."""

import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import Batch, Data

from tgseqloc.models import GATGraphEncoder
from tgseqloc.training.trainer import Trainer

EDGE_DIM, TEXT_DIM, CLASSES, EDGE_CLASSES = 10, 8, 5, 4


def graph(with_text: bool = False, seed: int = 0, text: torch.Tensor | None = None) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(3, 4, generator=g)
    node_class = torch.tensor([1, 2, 3])
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_attr = torch.rand(2, EDGE_DIM, generator=g)
    edge_label = torch.tensor([1, 2])
    is_text = torch.zeros(3, dtype=torch.bool)
    text_emb = torch.zeros(3, TEXT_DIM)
    is_text_edge = torch.zeros(2, dtype=torch.bool)
    if with_text:
        t = text if text is not None else torch.rand(TEXT_DIM, generator=g)
        x = torch.cat([x, torch.tensor([[0.5, 0.5, 0.1, 0.05]])])
        node_class = torch.cat([node_class, torch.tensor([0])])
        is_text = torch.cat([is_text, torch.tensor([True])])
        text_emb = torch.cat([text_emb, t[None]])
        edge_index = torch.cat([edge_index, torch.tensor([[3, 0], [0, 3]])], dim=1)
        edge_attr = torch.cat([edge_attr, torch.rand(2, EDGE_DIM, generator=g)])
        edge_label = torch.cat([edge_label, torch.tensor([0, 0])])
        is_text_edge = torch.cat([is_text_edge, torch.tensor([True, True])])
    return Data(x=x, node_class=node_class, is_text=is_text, text_emb=text_emb,
                edge_index=edge_index, edge_attr=edge_attr, edge_label=edge_label,
                is_text_edge=is_text_edge)


def model(fusion: str = "additive", use_text_nodes: bool = True, **extra) -> GATGraphEncoder:
    torch.manual_seed(0)
    return GATGraphEncoder(
        in_dim=4, hidden_dim=16, n_layers=2, proj_dim=8, num_node_classes=CLASSES,
        node_emb_dim=8, num_edge_classes=EDGE_CLASSES, edge_emb_dim=8,
        edge_cont_dim=EDGE_DIM, dropout=0.0, heads=2, use_text_nodes=use_text_nodes,
        text_emb_dim=TEXT_DIM, use_edge_geometry=True, text_fusion=fusion, **extra,
    ).eval()


def encode(m: GATGraphEncoder, *graphs: Data) -> torch.Tensor:
    with torch.no_grad():
        return m(Batch.from_data_list(list(graphs)))


class AdditiveTextTests(unittest.TestCase):
    def test_text_that_adds_nothing_changes_nothing(self) -> None:
        """Сам факт появления текста не должен сдвигать дескриптор."""

        m = model()
        with torch.no_grad():
            m.text_add.weight.zero_()
        self.assertTrue(torch.allclose(encode(m, graph(False)), encode(m, graph(True)), atol=1e-6))

    def test_the_node_form_is_not_neutral(self) -> None:
        """Причина новой формы: в узловой даже «пустой» текст меняет дескриптор."""

        m = model("node")
        with torch.no_grad():
            m.text_proj.weight.zero_(); m.text_proj.bias.zero_()
        self.assertFalse(torch.allclose(encode(m, graph(False)), encode(m, graph(True)), atol=1e-6))

    def test_text_reaches_only_the_frame_that_carries_it(self) -> None:
        m = model()
        together = encode(m, graph(False, seed=0), graph(True, seed=1))
        self.assertTrue(torch.allclose(together[0], encode(m, graph(False, seed=0))[0], atol=1e-6))
        self.assertFalse(torch.allclose(encode(m, graph(True, seed=1)), encode(m, graph(False, seed=1)), atol=1e-6))

    def test_scene_weights_start_where_the_text_free_model_starts(self) -> None:
        additive = model().state_dict()
        text_free = model(use_text_nodes=False).state_dict()
        self.assertEqual(set(additive) - set(text_free), {"text_add.weight"})
        for key, value in text_free.items():
            self.assertTrue(torch.equal(value, additive[key]), key)

    def test_the_node_form_keeps_its_checkpoint_arguments(self) -> None:
        m = model("node")
        self.assertNotIn("text_fusion", m.init_args)
        self.assertNotIn("text_dropout", m.init_args)
        self.assertIsNone(m.text_add)
        self.assertIn("text_proj.weight", m.state_dict())

    def test_text_dropout_acts_only_in_training(self) -> None:
        m = model(text_dropout=0.5)
        g = Batch.from_data_list([graph(True)])
        with torch.no_grad():
            self.assertTrue(torch.equal(m(g), m(g)))
            m.train()
            outputs = {tuple(m(g).flatten().tolist()) for _ in range(30)}
        self.assertGreater(len(outputs), 1)

    def test_unknown_fusion_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            model("concat")


class SplitGuardTests(unittest.TestCase):
    """Корень обслуживает несколько протоколов; чужое разбиение надо отвергать."""

    check = staticmethod(Trainer._check_split_matches_config)

    def test_single_split_accepts_its_own_file(self) -> None:
        self.check(SimpleNamespace(config={"dataset": {"split_folds_path": ""}}, split={}))

    def test_single_split_refuses_a_fold(self) -> None:
        with self.assertRaises(ValueError):
            self.check(SimpleNamespace(config={"dataset": {"split_folds_path": ""}},
                                       split={"split_fold": 4}))

    def test_fold_refuses_another_fold(self) -> None:
        with self.assertRaises(ValueError):
            self.check(SimpleNamespace(
                config={"dataset": {"split_folds_path": "f.json", "split_fold": 2}},
                split={"split_fold": 4}))

    def test_fold_zero_is_not_mistaken_for_unset(self) -> None:
        self.check(SimpleNamespace(
            config={"dataset": {"split_folds_path": "f.json", "split_fold": 0}},
            split={"split_fold": 0}))
