"""Prepared features for TriGlue training and prediction."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import dgl
from torch.utils.data import Dataset


def _pad(arrays: Sequence[np.ndarray]) -> np.ndarray:
    normalized = []
    for array in arrays:
        array = np.asarray(array)
        if array.ndim == 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1:
            array = array[None, :]
        normalized.append(array)
    trailing = {tuple(array.shape[1:]) for array in normalized}
    if len(trailing) != 1:
        raise ValueError(f"inconsistent feature dimensions within batch: {sorted(trailing)}")
    maximum = max(array.shape[0] for array in normalized)
    return np.stack(
        [
            np.pad(array, [(0, maximum - array.shape[0])] + [(0, 0)] * (array.ndim - 1))
            for array in normalized
        ]
    )


class PreparedFeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_root: Path, cache_size: int = 64):
        self.frame = frame.reset_index(drop=True)
        self.root = feature_root
        self._load_array = lru_cache(maxsize=cache_size)(self._load_array_uncached)
        self._load_graph = lru_cache(maxsize=cache_size)(self._load_graph_uncached)

    def __len__(self):
        return len(self.frame)

    @staticmethod
    def _load_array_uncached(path: str) -> np.ndarray:
        return np.load(path, allow_pickle=False)

    @staticmethod
    def _load_graph_uncached(path: str):
        graph = dgl.load_graphs(path)[0][0]
        if "feats" not in graph.ndata and "feat" in graph.ndata:
            graph.ndata["feats"] = graph.ndata["feat"]
        if "feats" not in graph.ndata:
            raise ValueError(f"graph has no node feature key 'feats': {path}")
        return graph

    def _protein(self, protein_id: str):
        return (
            self._load_array(str(self.root / "protein_embedding" / "one_hot" / f"{protein_id}.npy")),
            self._load_graph(str(self.root / "protein_graph" / f"{protein_id}.bin")).clone(),
            self._load_array(str(self.root / "protein_3d_embedding" / f"{protein_id}.npy")),
        )

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        compound_id = str(row["compound_id"])
        p1 = str(row["protein1_id"])
        p2 = str(row["protein2_id"])
        p1_one_hot, p1_graph, p1_esm2 = self._protein(p1)
        p2_one_hot, p2_graph, p2_esm2 = self._protein(p2)
        return {
            "smiles": str(row["smiles"]),
            "compound_graph": self._load_graph(
                str(self.root / "compound_graph" / f"{compound_id}.bin")
            ).clone(),
            "compound_embedding": self._load_array(
                str(self.root / "compound_3d_embedding" / f"{compound_id}.npy")
            ),
            "p1_one_hot": p1_one_hot,
            "p1_graph": p1_graph,
            "p1_esm2": p1_esm2,
            "p2_one_hot": p2_one_hot,
            "p2_graph": p2_graph,
            "p2_esm2": p2_esm2,
        }

    @staticmethod
    def collate(batch):
        def graph_batch(key: str):
            graphs = [sample[key] for sample in batch]
            for graph_index, graph in enumerate(graphs):
                graph.ndata["graph_id"] = torch.full(
                    (graph.num_nodes(),), graph_index, dtype=torch.long
                )
            return dgl.batch(graphs)

        return (
            [sample["smiles"] for sample in batch],
            graph_batch("compound_graph"),
            torch.tensor(
                _pad([sample["compound_embedding"] for sample in batch]), dtype=torch.float32
            ),
            torch.tensor(np.stack([sample["p1_one_hot"] for sample in batch]), dtype=torch.float32),
            graph_batch("p1_graph"),
            torch.tensor(_pad([sample["p1_esm2"] for sample in batch]), dtype=torch.float32),
            torch.tensor(np.stack([sample["p2_one_hot"] for sample in batch]), dtype=torch.float32),
            graph_batch("p2_graph"),
            torch.tensor(_pad([sample["p2_esm2"] for sample in batch]), dtype=torch.float32),
        )


class LabelledFeatureDataset(PreparedFeatureDataset):
    def __init__(self, frame, feature_root, *, training=False):
        super().__init__(frame, feature_root)
        self.training = training

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['label'] = float(self.frame.iloc[index]['label'])
        if self.training:
            for key in ('compound_graph', 'p1_graph', 'p2_graph'):
                graph = sample[key]
                keep = torch.rand(graph.num_edges()) >= 0.10
                sample[key] = dgl.edge_subgraph(graph, graph.edges(form='eid')[keep],
                                              relabel_nodes=False)
            # These arrays are encoder embeddings, not atomic coordinates.
            for key in ('compound_embedding', 'p1_esm2', 'p2_esm2'):
                value = np.asarray(sample[key], dtype=np.float32)
                sample[key] = value + np.random.normal(0, 0.02, value.shape).astype(np.float32)
        return sample

    @staticmethod
    def collate(batch):
        return (PreparedFeatureDataset.collate(batch),
                torch.tensor([sample['label'] for sample in batch], dtype=torch.float32))
