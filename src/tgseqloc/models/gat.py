"""Graph-attention encoder used by TGSeqLoc."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch_geometric.utils import scatter


class GATGraphEncoder(nn.Module):
    """Encode a PyG graph or batch into L2-normalized place descriptors."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        proj_dim: int = 64,
        num_node_classes: int | None = None,
        node_emb_dim: int = 64,
        num_edge_classes: int | None = None,
        edge_emb_dim: int = 64,
        edge_cont_dim: int = 10,
        dropout: float = 0.1,
        heads: int = 4,
        use_text_nodes: bool = False,
        text_emb_dim: int = 384,
        use_edge_geometry: bool = False,
        text_fusion: str = "node",
        text_dropout: float = 0.0,
        text_centering: bool = False,
        text_zero_init: bool = False,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least one")
        if use_text_nodes and num_node_classes is None:
            raise ValueError("text nodes require num_node_classes")
        if text_zero_init and text_fusion == "node":
            raise ValueError("text_zero_init applies to the additive and set forms only")
        if text_fusion not in ("node", "additive", "set"):
            raise ValueError(
                f"text_fusion must be 'node', 'additive' or 'set', got {text_fusion!r}"
            )
        if not 0.0 <= float(text_dropout) < 1.0:
            raise ValueError("text_dropout must be in [0, 1)")
        self.use_node_class = num_node_classes is not None
        self.use_edge_label = num_edge_classes is not None
        self.use_text_nodes = bool(use_text_nodes)
        self.text_emb_dim = int(text_emb_dim)
        self.use_edge_geometry = bool(use_edge_geometry)
        self.edge_cont_dim = int(edge_cont_dim)
        self.text_fusion = str(text_fusion)
        self.text_dropout = float(text_dropout)
        self.text_centering = bool(text_centering)
        self.text_zero_init = bool(text_zero_init)
        self.init_args = {
            "in_dim": int(in_dim),
            "hidden_dim": int(hidden_dim),
            "n_layers": int(n_layers),
            "proj_dim": int(proj_dim),
            "num_node_classes": num_node_classes,
            "node_emb_dim": int(node_emb_dim),
            "num_edge_classes": num_edge_classes,
            "edge_emb_dim": int(edge_emb_dim),
            "edge_cont_dim": int(edge_cont_dim),
            "dropout": float(dropout),
            "heads": int(heads),
            "use_text_nodes": bool(use_text_nodes),
            "text_emb_dim": int(text_emb_dim),
            "use_edge_geometry": bool(use_edge_geometry),
        }
        if self.text_fusion != "node" or self.text_dropout:
            # Recorded only when set, so a checkpoint of the node form keeps the
            # arguments it was saved with and still loads.
            self.init_args["text_fusion"] = self.text_fusion
            self.init_args["text_dropout"] = self.text_dropout
        if self.text_centering:
            self.init_args["text_centering"] = True
        if self.text_zero_init:
            self.init_args["text_zero_init"] = True

        self.node_emb = (
            nn.Embedding(num_node_classes, node_emb_dim)
            if self.use_node_class
            else None
        )
        self.text_proj = (
            nn.Linear(text_emb_dim, node_emb_dim)
            if self.use_text_nodes and self.text_fusion == "node"
            else None
        )
        self.text_ln = (
            nn.LayerNorm(node_emb_dim)
            if self.use_text_nodes and self.text_fusion == "node"
            else None
        )
        self.edge_emb = (
            nn.Embedding(num_edge_classes, edge_emb_dim)
            if self.use_edge_label
            else None
        )
        if self.node_emb is not None:
            nn.init.xavier_uniform_(self.node_emb.weight)
        if self.edge_emb is not None:
            nn.init.xavier_uniform_(self.edge_emb.weight)

        effective_in = in_dim + (node_emb_dim if self.use_node_class else 0)
        self.input_mlp = nn.Sequential(
            nn.Linear(effective_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.edge_cont_mlp = nn.Sequential(
            nn.Linear(edge_cont_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_label_proj = (
            nn.Sequential(
                nn.Linear(edge_emb_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.use_edge_label
            else None
        )
        self.edge_cont_ln = nn.LayerNorm(hidden_dim)
        self.edge_lbl_ln = nn.LayerNorm(hidden_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                dropout=dropout,
                edge_dim=hidden_dim,
                add_self_loops=True,
                residual=True,
            )
            for _ in range(n_layers)
        )
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self._out_dim = int(proj_dim)
        # Created last and without a bias: every other weight is drawn exactly
        # as in a model without text, and a zero text vector adds nothing.
        self.text_add = (
            nn.Linear(text_emb_dim, node_emb_dim, bias=False)
            if self.use_text_nodes and self.text_fusion == "additive"
            else None
        )
        # The set form, created after everything else for the same reason as the
        # additive term: every scene weight is drawn as in a model without text.
        set_form = self.use_text_nodes and self.text_fusion == "set"
        self.text_mlp = (
            nn.Sequential(
                nn.Linear(text_emb_dim + in_dim + node_emb_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if set_form
            else None
        )
        # No bias: a frame without strings must add exactly nothing.
        self.text_out = nn.Linear(hidden_dim, proj_dim, bias=False) if set_form else None
        if set_form:
            self.register_buffer("text_center", torch.zeros(text_emb_dim))
        else:
            self.text_center = None
        # The last text layer starts at zero, so the untrained model is exactly
        # the text-free one and text enters only as far as training pulls it in.
        # The layer is still drawn first: the random stream, and with it every
        # other weight, stays as without this option.
        if self.text_zero_init:
            last = self.text_add if self.text_add is not None else self.text_out
            if last is not None:
                nn.init.zeros_(last.weight)

    @staticmethod
    def _indices(values: Tensor, size: int, name: str, device: torch.device) -> Tensor:
        values = values.to(device=device, dtype=torch.long)
        if values.numel() and (values.min() < 0 or values.max() >= size):
            raise ValueError(f"{name} contains an index outside its embedding table")
        return values

    def _node_features(self, batch, x: Tensor) -> Tensor:
        if not self.use_node_class:
            return x
        node_class = getattr(batch, "node_class", None)
        if node_class is None:
            raise ValueError("num_node_classes was configured but node_class is absent")
        assert self.node_emb is not None
        node_class = self._indices(
            node_class, self.node_emb.num_embeddings, "node_class", x.device
        )
        node_features = self.node_emb(node_class)
        if self.use_text_nodes and self.text_fusion == "node":
            is_text = getattr(batch, "is_text", None)
            text_emb = getattr(batch, "text_emb", None)
            if is_text is None or text_emb is None:
                raise ValueError("use_text_nodes=True requires is_text and text_emb")
            is_text = is_text.to(device=x.device, dtype=torch.bool).reshape(-1)
            text_emb = text_emb.to(device=x.device, dtype=x.dtype)
            if is_text.numel() != x.shape[0]:
                raise ValueError("is_text must contain one value per node")
            if tuple(text_emb.shape) != (x.shape[0], self.text_emb_dim):
                raise ValueError(
                    f"text_emb must have shape ({x.shape[0]}, {self.text_emb_dim})"
                )
            if is_text.any():
                assert self.text_proj is not None and self.text_ln is not None
                node_features = node_features.clone()
                node_features[is_text] = self.text_ln(self.text_proj(text_emb[is_text]))
        return torch.cat((x, node_features), dim=1)

    def _edge_features(self, batch, x: Tensor, edge_count: int) -> Tensor:
        edge_cont = None
        raw_cont = getattr(batch, "edge_attr", None)
        if raw_cont is not None:
            raw_cont = raw_cont.to(device=x.device, dtype=x.dtype)
            if tuple(raw_cont.shape) != (edge_count, self.edge_cont_dim):
                raise ValueError(
                    f"edge_attr must have shape ({edge_count}, {self.edge_cont_dim})"
                )
            edge_cont = self.edge_cont_mlp(raw_cont)

        edge_label = None
        raw_label = getattr(batch, "edge_label", None)
        if self.use_edge_label and raw_label is not None:
            assert self.edge_emb is not None and self.edge_label_proj is not None
            raw_label = self._indices(
                raw_label, self.edge_emb.num_embeddings, "edge_label", x.device
            )
            edge_label = self.edge_label_proj(self.edge_emb(raw_label))
            is_text_edge = getattr(batch, "is_text_edge", None)
            if is_text_edge is not None:
                is_text_edge = is_text_edge.to(x.device, torch.bool).reshape(-1)
                if is_text_edge.numel() != edge_count:
                    raise ValueError("is_text_edge must contain one value per edge")
                edge_label = edge_label.masked_fill(is_text_edge[:, None], 0)

        if edge_label is not None and edge_cont is not None and self.use_edge_geometry:
            continuous = self.edge_cont_ln(edge_cont)
            categorical = self.edge_lbl_ln(edge_label)
            gate = self.edge_gate(torch.cat((continuous, categorical), dim=1))
            return gate * continuous + (1 - gate) * categorical
        if edge_label is not None:
            return edge_label
        if edge_cont is not None and self.use_edge_geometry:
            return self.edge_cont_ln(edge_cont)
        return x.new_zeros((edge_count, self.init_args["hidden_dim"]))

    def forward(self, batch, return_attn: bool = False):
        x = batch.x.float()
        if x.ndim != 2 or x.shape[1] != self.init_args["in_dim"]:
            raise ValueError(
                f"x must have shape [nodes, {self.init_args['in_dim']}]"
            )
        if x.shape[0] == 0:
            raise ValueError("graphs with no nodes cannot be encoded")
        edge_index = batch.edge_index.to(device=x.device, dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if self.text_mlp is not None:
            return self._forward_set(batch, x, edge_index, return_attn)
        if self.text_add is not None:
            return self._forward_additive(batch, x, edge_index, return_attn)
        h = self.input_mlp(self._node_features(batch, x))
        edge_attr = self._edge_features(batch, x, edge_index.shape[1])
        attention = None
        for index, conv in enumerate(self.convs):
            if return_attn and index == len(self.convs) - 1:
                h, attention = conv(
                    h, edge_index, edge_attr, return_attention_weights=True
                )
            else:
                h = conv(h, edge_index, edge_attr)
            h = self.dropout(self.activation(h))

        graph_index = getattr(batch, "batch", None)
        if graph_index is None:
            graph_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            graph_index = graph_index.to(x.device)
        pooled = torch.cat(
            (global_mean_pool(h, graph_index), global_max_pool(h, graph_index)), dim=1
        )
        descriptor = F.normalize(self.proj(pooled), p=2, dim=1)
        return (descriptor, attention) if return_attn else descriptor

    def _forward_additive(self, batch, x: Tensor, edge_index: Tensor, return_attn: bool):
        """Text as a term added to the object it is written on.

        A text node adds nothing to the graph that is encoded: its projected
        embedding is summed into the feature of the object it is attached to,
        and then the text nodes and their edges are dropped before message
        passing and pooling. For a frame without text the sum is empty, so the
        frame is encoded by exactly the computation of a model without the text
        layer -- the same nodes, the same edges, the same attention
        normalization, the same pooling set.

        This is what the node form cannot give. There an extra node shifts the
        mean over nodes, can only raise the maximum, takes a share of its
        object's attention and alters the mean edge attribute that fills that
        object's self-loop, so that whether a frame carries text changes its
        descriptor regardless of what the text says. On Oxford RobotCar that
        pulled queries with text towards any frame with text.
        """

        is_text = getattr(batch, "is_text", None)
        text_emb = getattr(batch, "text_emb", None)
        if is_text is None or text_emb is None:
            raise ValueError("text_fusion='additive' requires is_text and text_emb")
        is_text = is_text.to(device=x.device, dtype=torch.bool).reshape(-1)
        text_emb = text_emb.to(device=x.device, dtype=x.dtype)
        if is_text.numel() != x.shape[0]:
            raise ValueError("is_text must contain one value per node")
        if tuple(text_emb.shape) != (x.shape[0], self.text_emb_dim):
            raise ValueError(
                f"text_emb must have shape ({x.shape[0]}, {self.text_emb_dim})"
            )
        edge_count = edge_index.shape[1]
        text_edge = getattr(batch, "is_text_edge", None)
        if text_edge is None:
            text_edge = is_text[edge_index[0]] | is_text[edge_index[1]]
        else:
            text_edge = text_edge.to(device=x.device, dtype=torch.bool).reshape(-1)
            if text_edge.numel() != edge_count:
                raise ValueError("is_text_edge must contain one value per edge")

        features = self._node_features(batch, x)
        in_dim = x.shape[1]
        link = text_edge & is_text[edge_index[0]] & ~is_text[edge_index[1]]
        if bool(link.any()):
            assert self.text_add is not None
            term = self.text_add(text_emb[edge_index[0, link]])
            if self.training and self.text_dropout > 0:
                keep = torch.rand(term.shape[0], 1, device=x.device) >= self.text_dropout
                term = term * keep.to(term.dtype)
            added = scatter(
                term, edge_index[1, link], dim=0, dim_size=x.shape[0], reduce="sum"
            )
            features = torch.cat(
                (features[:, :in_dim], features[:, in_dim:] + added), dim=1
            )

        scene = ~is_text
        edge_attr = self._edge_features(batch, x, edge_count)
        scene_edge = scene[edge_index[0]] & scene[edge_index[1]] & ~text_edge
        remap = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        remap[scene] = torch.arange(int(scene.sum()), device=x.device)
        scene_index = remap[edge_index[:, scene_edge]]
        scene_attr = edge_attr[scene_edge]

        h = self.input_mlp(features[scene])
        attention = None
        for index, conv in enumerate(self.convs):
            if return_attn and index == len(self.convs) - 1:
                h, attention = conv(
                    h, scene_index, scene_attr, return_attention_weights=True
                )
            else:
                h = conv(h, scene_index, scene_attr)
            h = self.dropout(self.activation(h))

        graph_index = getattr(batch, "batch", None)
        if graph_index is None:
            graph_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            graphs = 1
        else:
            graph_index = graph_index.to(x.device)
            graphs = int(getattr(batch, "num_graphs", int(graph_index.max()) + 1))
        graph_index = graph_index[scene]
        pooled = torch.cat(
            (
                global_mean_pool(h, graph_index, size=graphs),
                global_max_pool(h, graph_index, size=graphs),
            ),
            dim=1,
        )
        descriptor = F.normalize(self.proj(pooled), p=2, dim=1)
        return (descriptor, attention) if return_attn else descriptor

    def _forward_set(self, batch, x: Tensor, edge_index: Tensor, return_attn: bool):
        """Scene graph over objects, and the strings of the frame as a set.

        ``descriptor = normalize(P_scene(pool(objects)) + P_text(sum_i phi(text_i)))``

        Each string is encoded on its own from its embedding, its box and the
        class of the object it is written on, and the strings of a frame are
        summed only after that nonlinear encoding. Several signs on one facade
        therefore stay a set: in the additive form a linear term summed them
        into one vector, and two strings could not be told from their sum.

        The scene part is computed over objects alone, exactly as in a model
        without text, and a frame without strings adds an empty sum, so such a
        frame gets precisely the text-free descriptor. The strings' embeddings
        are centred on the mean string of the training data, so that an
        unremarkable string starts close to adding nothing.
        """

        is_text = getattr(batch, "is_text", None)
        text_emb = getattr(batch, "text_emb", None)
        if is_text is None or text_emb is None:
            raise ValueError("text_fusion='set' requires is_text and text_emb")
        is_text = is_text.to(device=x.device, dtype=torch.bool).reshape(-1)
        text_emb = text_emb.to(device=x.device, dtype=x.dtype)
        if is_text.numel() != x.shape[0]:
            raise ValueError("is_text must contain one value per node")
        if tuple(text_emb.shape) != (x.shape[0], self.text_emb_dim):
            raise ValueError(
                f"text_emb must have shape ({x.shape[0]}, {self.text_emb_dim})"
            )
        edge_count = edge_index.shape[1]
        text_edge = getattr(batch, "is_text_edge", None)
        if text_edge is None:
            text_edge = is_text[edge_index[0]] | is_text[edge_index[1]]
        else:
            text_edge = text_edge.to(device=x.device, dtype=torch.bool).reshape(-1)
            if text_edge.numel() != edge_count:
                raise ValueError("is_text_edge must contain one value per edge")

        features = self._node_features(batch, x)
        in_dim = x.shape[1]
        scene = ~is_text
        edge_attr = self._edge_features(batch, x, edge_count)
        scene_edge = scene[edge_index[0]] & scene[edge_index[1]] & ~text_edge
        remap = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        remap[scene] = torch.arange(int(scene.sum()), device=x.device)
        scene_index = remap[edge_index[:, scene_edge]]
        scene_attr = edge_attr[scene_edge]

        h = self.input_mlp(features[scene])
        attention = None
        for index, conv in enumerate(self.convs):
            if return_attn and index == len(self.convs) - 1:
                h, attention = conv(
                    h, scene_index, scene_attr, return_attention_weights=True
                )
            else:
                h = conv(h, scene_index, scene_attr)
            h = self.dropout(self.activation(h))

        graph_index = getattr(batch, "batch", None)
        if graph_index is None:
            graph_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            graphs = 1
        else:
            graph_index = graph_index.to(x.device)
            graphs = int(getattr(batch, "num_graphs", int(graph_index.max()) + 1))
        scene_graph_index = graph_index[scene]
        pooled = torch.cat(
            (
                global_mean_pool(h, scene_graph_index, size=graphs),
                global_max_pool(h, scene_graph_index, size=graphs),
            ),
            dim=1,
        )
        descriptor = self.proj(pooled)

        link = text_edge & is_text[edge_index[0]] & ~is_text[edge_index[1]]
        if bool(link.any()):
            assert self.text_mlp is not None and self.text_out is not None
            strings = edge_index[0, link]
            objects = edge_index[1, link]
            element = torch.cat(
                (
                    text_emb[strings] - self.text_center,
                    x[strings],
                    features[objects, in_dim:],
                ),
                dim=1,
            )
            encoded = self.text_mlp(element)
            if self.training and self.text_dropout > 0:
                keep = torch.rand(encoded.shape[0], 1, device=x.device) >= self.text_dropout
                encoded = encoded * keep.to(encoded.dtype)
            per_frame = scatter(
                encoded, graph_index[strings], dim=0, dim_size=graphs, reduce="sum"
            )
            descriptor = descriptor + self.text_out(per_frame)
        descriptor = F.normalize(descriptor, p=2, dim=1)
        return (descriptor, attention) if return_attn else descriptor

    @property
    def out_dim(self) -> int:
        return self._out_dim
