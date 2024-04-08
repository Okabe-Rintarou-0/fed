import copy
from typing import Any, Dict, List

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F


def aggregate_weights(
    w: List[Dict[str, Any]],
    agg_weight: List[float] | torch.Tensor,
    aggregatable_weights: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Returns the average of the weights.
    """
    if len(w) == 0:
        return
    w_avg = copy.deepcopy(w[0])
    if not isinstance(agg_weight, torch.Tensor):
        weight = torch.tensor(agg_weight)
    else:
        weight = agg_weight
    agg_w = weight / (weight.sum(dim=0))
    for key in w_avg.keys():
        if aggregatable_weights is not None and key not in aggregatable_weights:
            continue
        w_avg[key] = torch.zeros_like(w_avg[key], dtype=float)
        for i in range(len(w)):
            w_avg[key] += agg_w[i] * w[i][key]

    return w_avg


def aggregate_personalized_model(
    client_idxs: List[int],
    weights_map: Dict[int, Dict[str, Any]],
    adjacency_matrix: torch.Tensor,
    aggregatable_weights: List[str] | None = None,
) -> Dict[int, Dict[str, Any]]:
    tmp_client_weights_map = {}
    for client_idx in client_idxs:
        model_i = weights_map[client_idx]
        tmp_client_weights_map[client_idx] = {}
        agg_vector = adjacency_matrix[client_idx]
        for key in model_i:
            if aggregatable_weights is not None and key not in aggregatable_weights:
                continue
            tmp_client_weights_map[client_idx][key] = torch.zeros_like(
                model_i[key], dtype=float
            )
            for neighbor_idx in client_idxs:
                neighbor_model = weights_map[neighbor_idx]
                tmp_client_weights_map[client_idx][key] += (
                    neighbor_model[key] * agg_vector[neighbor_idx]
                )

    return tmp_client_weights_map


def aggregate_protos(local_protos_list, local_label_sizes_list):
    agg_protos_label = {}
    agg_sizes_label = {}
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        local_sizes = local_label_sizes_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
                agg_sizes_label[label].append(local_sizes[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
                agg_sizes_label[label] = [local_sizes[label]]

    for [label, proto_list] in agg_protos_label.items():
        sizes_list = agg_sizes_label[label]
        proto = 0 * proto_list[0]
        for i in range(len(proto_list)):
            proto += sizes_list[i] * proto_list[i]
        agg_protos_label[label] = proto / sum(sizes_list)

    return agg_protos_label


def write_client_datasets(
    idx: int,
    writer: SummaryWriter,
    dataloader: DataLoader,
    train: bool,
    get_index: bool,
):
    tag = f'client_{idx}_{"train" if train else "test"}_dataset'
    data_iter = iter(dataloader)
    if get_index:
        imgs, _, _ = next(data_iter)
    else:
        imgs, _ = next(data_iter)
    writer.add_images(tag, imgs)


def calc_label_distribution(dataloader: DataLoader, num_classes: int, get_index: bool):
    distribution = [0] * num_classes
    num_iter = len(dataloader)
    data_iter = iter(dataloader)
    for _ in range(num_iter):
        if get_index:
            _, labels, _ = next(data_iter)
        else:
            _, labels = next(data_iter)

        for label in labels:
            distribution[label] += 1
    return distribution


def write_client_label_distribution(
    idx: int,
    writer: SummaryWriter,
    train_loader: DataLoader,
    num_classes: int,
    get_index: bool,
):
    # print(f"calculating client {idx}'s label distribution...", end='')
    train_distribution = calc_label_distribution(
        train_loader, num_classes, get_index=get_index
    )
    labels = list(range(num_classes))
    plt.clf()
    plt.bar(labels, train_distribution)
    plt.xticks(labels)
    writer.add_figure(f"client_{idx}_label_distribution", plt.gcf())
    # print('done')


def get_protos(protos):
    """
    Returns the average of the feature embeddings of samples from per-class.
    """
    protos_mean = {}
    for [label, proto_list] in protos.items():
        proto = 0 * proto_list[0]
        for i in proto_list:
            proto += i
        protos_mean[label] = proto / len(proto_list)

    return protos_mean


def weight_flatten(model: Dict[str, Any]):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def weight_flatten_cls(model: Dict[str, Any]):
    params = []
    for k in model:
        if "cls" in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def draw_label_dist(dists: List[Dict[int, int]], num_classes: int):
    plt.clf()
    label_tot_cnts = [
        sum([dists[idx][label] for label in range(num_classes)])
        for idx in range(len(dists))
    ]
    for idx in range(len(dists)):
        this_cnts = np.array([dists[idx][label]
                             for label in range(num_classes)])
        for label in range(num_classes):
            label_radius = 80 * this_cnts[label] / label_tot_cnts[label]
            plt.scatter([idx], [label], s=label_radius, color="red")
    plt.xticks(range(len(dists)))
    plt.yticks(range(num_classes))
    plt.tick_params(axis="y", labelsize=16)
    plt.tick_params(axis="x", labelsize=14)
    plt.xlabel("Client", fontsize=16)
    plt.ylabel("Label", fontsize=16)
    plt.savefig("dist.png", dpi=600, bbox_inches="tight")


def cal_cosine_difference_vector(
    client_idxs: List[int],
    initial_global_parameters: Dict[str, Any],
    weights_map: Dict[int, Dict[str, Any]],
):
    num_clients = len(client_idxs)
    difference_vector = torch.zeros((num_clients))
    flatten_weights_map = {}
    dw = {}
    for idx in client_idxs:
        model_i = weights_map[idx]
        dw[idx] = {}
        model_i = weights_map[idx]
        for key in model_i:
            if "cls" not in key:
                continue
            dw[idx][key] = model_i[key] - initial_global_parameters[key]
        flatten_weights_map[idx] = weight_flatten_cls(dw[idx]).unsqueeze(0)

    for i in range(num_clients):
        idx_i = client_idxs[i]
        flatten_weight_i = flatten_weights_map[idx_i]
        for j in range(i + 1, num_clients):
            idx_j = client_idxs[j]
            flatten_weight_j = flatten_weights_map[idx_j]
            diff = -torch.nn.functional.cosine_similarity(
                flatten_weight_i, flatten_weight_j
            ).unsqueeze(0)
            difference_vector[i] += diff.item()
        difference_vector[i] /= num_clients
    return difference_vector


def cal_cosine_difference_matrix(
    client_idxs: List[int],
    initial_global_parameters: Dict[str, Any],
    weights_map: Dict[int, Dict[str, Any]],
    aggregatable_weights: List[str],
):
    num_clients = len(client_idxs)
    difference_matrix = torch.zeros((num_clients, num_clients))
    flatten_weights_map = {}
    dw = {}
    for idx in client_idxs:
        dw[idx] = {}
        model_i = weights_map[idx]
        for key in model_i:
            if key in aggregatable_weights:
                dw[idx][key] = model_i[key] - initial_global_parameters[key]
        flatten_weights_map[idx] = weight_flatten_cls(dw[idx]).unsqueeze(0)

    for i in range(num_clients):
        idx_i = client_idxs[i]
        flatten_weight_i = flatten_weights_map[idx_i]
        for j in range(i, num_clients):
            idx_j = client_idxs[j]
            flatten_weight_j = flatten_weights_map[idx_j]
            diff = -torch.nn.functional.cosine_similarity(
                flatten_weight_i, flatten_weight_j
            ).unsqueeze(0)
            difference_matrix[i, j] = diff
            difference_matrix[j, i] = diff
    return difference_matrix


def optimize_collaborate_vector(
    difference_vector: torch.Tensor,
    alpha: float,
    agg_weights: List[float],
):
    n = difference_vector.shape[0]
    p = np.array(agg_weights)
    P = np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = -np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)

    d = difference_vector.numpy()
    q = alpha * d - 2 * p
    x = cp.Variable(n)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(x, P) + q.T @ x), [G @ x <= h, A @ x == b]
    )
    prob.solve()
    return torch.Tensor(x.value)


def optimize_adjacency_matrix(
    adjacency_matrix: torch.Tensor,
    client_idxs: List[int],
    difference_matrix: torch.Tensor,
    alpha: float,
    agg_weights: List[float],
):
    n = difference_matrix.shape[0]
    p = np.array(agg_weights)
    P = np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = -np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(n):
        model_difference_vector = difference_matrix[i]
        d = model_difference_vector.numpy()
        q = alpha * d - 2 * p
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, P) + q.T @ x), [G @ x <= h, A @ x == b]
        )
        prob.solve()
        adjacency_matrix[client_idxs[i], client_idxs] = torch.Tensor(x.value)
    return adjacency_matrix


def update_adjacency_matrix(
    adjacency_matrix: torch.Tensor,
    client_idxs: List[int],
    initial_global_parameters: Dict[str, Any],
    weights_map: Dict[int, Dict[str, Any]],
    agg_weights: List[float],
    alpha: float,
    aggregatable_weights: List[str],
):
    difference_matrix = cal_cosine_difference_matrix(
        client_idxs, initial_global_parameters, weights_map, aggregatable_weights
    )
    adjacency_matrix = optimize_adjacency_matrix(
        adjacency_matrix, client_idxs, difference_matrix, alpha, agg_weights
    )
    return adjacency_matrix
