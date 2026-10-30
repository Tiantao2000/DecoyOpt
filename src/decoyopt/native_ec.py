"""Native CLEAN triplet sampling and EC-center updates."""
from __future__ import annotations
import random
import numpy as np
import torch
from .model import ECModel
LayerNormNet = ECModel

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_memberships(records: list[tuple[str, frozenset[str]]]):
    vocabulary = sorted({ec for _, labels in records for ec in labels})
    ec_index = {ec: i for i, ec in enumerate(vocabulary)}
    members: list[list[int]] = [[] for _ in vocabulary]
    label_sets = [labels for _, labels in records]
    for sample_index, labels in enumerate(label_sets):
        for ec in labels:
            members[ec_index[ec]].append(sample_index)
    return vocabulary, members, label_sets


@torch.no_grad()
def encode(model: LayerNormNet, data: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    return torch.cat([model(data[i:i + batch_size]) for i in range(0, len(data), batch_size)])


@torch.no_grad()
def centers_from_embeddings(emb: torch.Tensor, members: list[list[int]]) -> torch.Tensor:
    return torch.stack([emb[index].mean(0) for index in members])


@torch.no_grad()
def neighbor_pool(centers: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    if len(centers) < 2:
        raise ValueError("At least two training EC classes are needed")
    dist = torch.cdist(centers, centers).cpu().numpy()
    np.fill_diagonal(dist, np.inf)
    count = min(k, len(centers) - 1)
    closest = np.argpartition(dist, count - 1, axis=1)[:, :count]
    order = np.argsort(np.take_along_axis(dist, closest, axis=1), axis=1)
    closest = np.take_along_axis(closest, order, axis=1)
    d = np.take_along_axis(dist, closest, axis=1)
    inv = 1.0 / np.maximum(d, 1e-8)
    weights = inv / inv.sum(axis=1, keepdims=True)
    return closest, weights


def sample_triplets(
    members: list[list[int]], labels: list[frozenset[str]],
    vocabulary: list[str], neighbors: np.ndarray, weights: np.ndarray, rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sampled anchor per EC, as in the original CLEAN dataset."""
    classes = list(range(len(members)))
    ec_index = {name: index for index, name in enumerate(vocabulary)}
    rng.shuffle(classes)
    anchors, positives, negatives = [], [], []
    all_indices = range(len(labels))
    for ec in classes:
        anchor = rng.choice(members[ec])
        anchor_labels = labels[anchor]
        # Positive shares this sampled EC; singleton uses two dropout views.
        positive = rng.choice(members[ec])
        if len(members[ec]) > 1:
            while positive == anchor:
                positive = rng.choice(members[ec])
        negative = None
        # Use inverse-distance weights over the nearest 30 EC classes, checking
        # the entire EC label set of the chosen protein to avoid false negatives.
        negative_query_ec = ec_index[rng.choice(tuple(sorted(anchor_labels)))]
        for _ in range(60):
            choice = int(np_rng.choice(len(neighbors[negative_query_ec]), p=weights[negative_query_ec]))
            candidate = rng.choice(members[int(neighbors[negative_query_ec, choice])])
            if anchor_labels.isdisjoint(labels[candidate]):
                negative = candidate
                break
        if negative is None:
            for _ in range(1000):
                candidate = rng.choice(all_indices)
                if anchor_labels.isdisjoint(labels[candidate]):
                    negative = candidate
                    break
        if negative is None:
            raise ValueError("Could not find a training protein with disjoint EC labels")
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
    return (np.asarray(anchors), np.asarray(positives), np.asarray(negatives))
