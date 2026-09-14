"""Текст как набор строк рядом со сценой: без текста — как без текстового слоя, строки не сливаются."""

import unittest

import torch
from torch_geometric.data import Batch, Data

from tgseqloc.models import GATGraphEncoder
from tgseqloc.training.trainer import text_center_from_graphs

EDGE_DIM, TEXT_DIM, CLASSES, EDGE_CLASSES = 10, 8, 5, 4


def graph(strings=(), seed: int = 0) -> Data:
    """Три объекта; каждая строка — текстовый узел на объекте 0 с одной и той же рамкой."""

    g = torch.Generator().manual_seed(seed)
    x = torch.rand(3, 4, generator=g)
    node_class = [1, 2, 3]
    edges = [[0, 1], [1, 2]]
    edge_attr = torch.rand(2, EDGE_DIM, generator=g)
    labels = [1, 2]
    is_text = [False] * 3
    text_emb = [torch.zeros(TEXT_DIM)] * 3
    text_edge = [False, False]
    for text in strings:
        index = len(node_class)
        x = torch.cat([x, torch.tensor([[0.5, 0.5, 0.1, 0.05]])])
        node_class.append(0); is_text.append(True); text_emb.append(text)
        edges[0] += [index, 0]; edges[1] += [0, index]
        edge_attr = torch.cat([edge_attr, torch.full((2, EDGE_DIM), 0.3)])
        labels += [0, 0]; text_edge += [True, True]
    return Data(x=x, node_class=torch.tensor(node_class), is_text=torch.tensor(is_text),
                text_emb=torch.stack(text_emb), edge_index=torch.tensor(edges),
                edge_attr=edge_attr, edge_label=torch.tensor(labels),
                is_text_edge=torch.tensor(text_edge))


def model(fusion: str = "set", use_text_nodes: bool = True, **extra) -> GATGraphEncoder:
    torch.manual_seed(0)
    return GATGraphEncoder(
        in_dim=4, hidden_dim=16, n_layers=2, proj_dim=8, num_node_classes=CLASSES,
        node_emb_dim=8, num_edge_classes=EDGE_CLASSES, edge_emb_dim=8,
        edge_cont_dim=EDGE_DIM, dropout=0.0, heads=2, use_text_nodes=use_text_nodes,
        text_emb_dim=TEXT_DIM, use_edge_geometry=True, text_fusion=fusion, **extra,
    ).eval()


def encode(m, *graphs):
    with torch.no_grad():
        return m(Batch.from_data_list(list(graphs)))


class SetTextTests(unittest.TestCase):
    def test_a_frame_without_text_gets_the_text_free_descriptor(self) -> None:
        """Не просто «стабильно», а в точности то, что дала бы модель без текста."""

        self.assertTrue(torch.allclose(encode(model(), graph()), encode(model(use_text_nodes=False), graph()), atol=1e-6))

    def test_scene_weights_start_where_the_text_free_model_starts(self) -> None:
        with_set = model().state_dict()
        text_free = model(use_text_nodes=False).state_dict()
        for key, value in text_free.items():
            self.assertTrue(torch.equal(value, with_set[key]), key)
        self.assertEqual({k.split(".")[0] for k in set(with_set) - set(text_free)},
                         {"text_mlp", "text_out", "text_center"})

    def test_strings_stay_a_set_rather_than_their_sum(self) -> None:
        """Слагаемое не отличает две вывески от одной с суммарным вектором; набор отличает."""

        g = torch.Generator().manual_seed(5)
        a, b = torch.rand(TEXT_DIM, generator=g), torch.rand(TEXT_DIM, generator=g)
        additive = model("additive")
        self.assertTrue(torch.allclose(encode(additive, graph([a, b])), encode(additive, graph([a + b])), atol=1e-6))
        with_set = model()
        self.assertFalse(torch.allclose(encode(with_set, graph([a, b])), encode(with_set, graph([a + b])), atol=1e-6))

    def test_text_reaches_only_the_frame_that_carries_it(self) -> None:
        m = model()
        together = encode(m, graph(seed=0), graph([torch.ones(TEXT_DIM)], seed=1))
        self.assertTrue(torch.allclose(together[0], encode(m, graph(seed=0))[0], atol=1e-6))
        self.assertFalse(torch.allclose(encode(m, graph([torch.ones(TEXT_DIM)], seed=1)), encode(m, graph(seed=1)), atol=1e-6))

    def test_centering_is_recorded_only_when_asked(self) -> None:
        self.assertNotIn("text_centering", model().init_args)
        self.assertTrue(model(text_centering=True).init_args["text_centering"])
        self.assertNotIn("text_centering", model("node").init_args)

    def test_center_is_the_mean_string(self) -> None:
        center = text_center_from_graphs([graph([torch.ones(TEXT_DIM)]), graph([torch.full((TEXT_DIM,), 3.0)]), graph()], TEXT_DIM)
        self.assertTrue(torch.allclose(center, torch.full((TEXT_DIM,), 2.0)))
        self.assertTrue(torch.equal(text_center_from_graphs([graph()], TEXT_DIM), torch.zeros(TEXT_DIM)))

    def test_zero_init_starts_from_the_text_free_descriptor(self) -> None:
        """С нулевым выходом текстовой ветки необученная модель в точности равна модели без текста."""

        strings = [torch.ones(TEXT_DIM), torch.full((TEXT_DIM,), 2.0)]
        text_free = encode(model(use_text_nodes=False), graph())
        for fusion in ("set", "additive"):
            m = model(fusion, text_zero_init=True)
            self.assertTrue(torch.allclose(encode(m, graph(strings)), text_free, atol=1e-6), fusion)
            self.assertTrue(m.init_args["text_zero_init"])

    def test_zero_init_leaves_every_other_weight_as_it_was(self) -> None:
        plain, zero = model().state_dict(), model(text_zero_init=True).state_dict()
        for key, value in plain.items():
            if key != "text_out.weight":
                self.assertTrue(torch.equal(value, zero[key]), key)
        self.assertEqual(int(zero["text_out.weight"].abs().sum()), 0)
        self.assertNotIn("text_zero_init", model().init_args)

    def test_zero_init_still_lets_text_in(self) -> None:
        """Градиент доходит до нулевого слоя, иначе текст никогда бы не включился."""

        m = model(text_zero_init=True).train()
        out = m(Batch.from_data_list([graph([torch.ones(TEXT_DIM)])]))
        out.sum().backward()
        self.assertGreater(float(m.text_out.weight.grad.abs().sum()), 0.0)

    def test_zero_init_is_refused_for_the_node_form(self) -> None:
        with self.assertRaises(ValueError):
            model("node", text_zero_init=True)
