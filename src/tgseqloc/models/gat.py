"""Graph-attention encoder used by TGSeqLoc."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


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
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be at least one")
        if use_text_nodes and num_node_classes is None:
            raise ValueError("text nodes require num_node_classes")
        self.use_node_class = num_node_classes is not None
        self.use_edge_label = num_edge_classes is not None
        self.use_text_nodes = bool(use_text_nodes)
        self.text_emb_dim = int(text_emb_dim)
        self.use_edge_geometry = bool(use_edge_geometry)
        self.edge_cont_dim = int(edge_cont_dim)
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

        self.node_emb = (
            nn.Embedding(num_node_classes, node_emb_dim)
            if self.use_node_class
            else None
        )
        self.text_proj = (
            nn.Linear(text_emb_dim, node_emb_dim) if self.use_text_nodes else None
        )
        self.text_ln = nn.LayerNorm(node_emb_dim) if self.use_text_nodes else None
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
        if self.use_text_nodes:
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

    @property
    def out_dim(self) -> int:
        return self._out_dim
