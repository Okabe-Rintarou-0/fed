import copy
from typing import Any, Dict, List

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import cvxpy as cp
from models import CifarCNN


def aggregate_weights(w: List[Dict[str, Any]], agg_weight: List[float], aggregatable_weights: List[str] | None = None) -> Dict[str, Any]:
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = torch.tensor(agg_weight)
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        if aggregatable_weights is not None and key not in aggregatable_weights:
            continue
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += agg_w[i]*w[i][key]

    return w_avg


def aggregate_personalized_model(client_idxs: List[int], weights_map: Dict[int, Dict[str, Any]], adjacency_matrix: torch.tensor) -> Dict[int, Dict[str, Any]]:
    tmp_client_weights_map = {}
    for client_idx in client_idxs:
        model_i = weights_map[client_idx]
        tmp_client_weights_map[client_idx] = {}
        agg_vector = adjacency_matrix[client_idx]
        for key in model_i:
            tmp_client_weights_map[client_idx][key] = torch.zeros_like(
                model_i[key])
            for neighbor_idx in client_idxs:
                neighbor_model = weights_map[neighbor_idx]
                tmp_client_weights_map[client_idx][key] += neighbor_model[key] * \
                    agg_vector[neighbor_idx]
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


def write_client_datasets(idx: int, writer: SummaryWriter, dataloader: DataLoader, train: bool):
    tag = f'client_{idx}_{"train" if train else "test"}_dataset'
    data_iter = iter(dataloader)
    imgs, _ = next(data_iter)
    writer.add_images(tag, imgs)


def calc_label_distribution(dataloader: DataLoader, num_classes: int):
    distribution = [0] * num_classes
    for _, labels in dataloader:
        for label in labels:
            distribution[label] += 1
    return distribution


def write_client_label_distribution(idx: int, writer: SummaryWriter, train_loader: DataLoader, num_classes: int):
    train_distribution = calc_label_distribution(train_loader, num_classes)
    labels = list(range(num_classes))
    plt.clf()
    plt.bar(labels, train_distribution)
    plt.xticks(labels)
    writer.add_figure(f'client_{idx}_label_distribution', plt.gcf())


def show_img_batch(imgs):
    imgs = torchvision.utils.make_grid(imgs).numpy()
    plt.imshow(np.transpose(imgs, (1, 2, 0)))
    plt.show()


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


def cal_cosine_difference_matrix(client_idxs: List[int], initial_global_parameters: Dict[str, Any], weights_map: Dict[int, Dict[str, Any]]):
    num_clients = len(client_idxs)
    difference_matrix = torch.zeros((num_clients, num_clients))
    flatten_weights_map = {}
    for idx in client_idxs:
        model_i = weights_map[idx]
        for key in model_i:
            model_i[key] -= initial_global_parameters[key]
        flatten_weights_map[idx] = weight_flatten(model_i).unsqueeze(0)

    for i in range(num_clients):
        idx_i = client_idxs[i]
        flatten_weight_i = flatten_weights_map[idx_i]
        for j in range(i, num_clients):
            idx_j = client_idxs[j]
            flatten_weight_j = flatten_weights_map[idx_j]
            diff = - torch.nn.functional.cosine_similarity(
                flatten_weight_i, flatten_weight_j).unsqueeze(0)
            if diff < - 0.9:
                diff = - 1.0
            difference_matrix[i, j] = diff
            difference_matrix[j, i] = diff
    return difference_matrix


def optimize_adjacency_matrix(adjacency_matrix: torch.tensor, client_idxs: List[int], difference_matrix: torch.tensor, alpha: float, agg_weights: List[float]):
    n = difference_matrix.shape[0]
    p = np.array(agg_weights)
    P = alpha * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(n):
        model_difference_vector = difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * alpha * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                          [G @ x <= h,
                           A @ x == b]
                          )
        prob.solve()
        adjacency_matrix[client_idxs[i], client_idxs] = torch.Tensor(x.value)
    return adjacency_matrix


def update_adjacency_matrix(adjacency_matrix: torch.tensor, client_idxs: List[int], initial_global_parameters: Dict[str, Any], weights_map: Dict[int, Dict[str, Any]], agg_weights: List[float], alpha: float):
    difference_matrix = cal_cosine_difference_matrix(
        client_idxs, initial_global_parameters, weights_map)
    adjacency_matrix = optimize_adjacency_matrix(
        adjacency_matrix, client_idxs, difference_matrix, alpha, agg_weights)
    return adjacency_matrix


if __name__ == '__main__':
    num_clients = 10
    adjacency_matrix = torch.ones(num_clients, num_clients) / (num_clients)
    client_idxs = list(range(int(num_clients / 2)))
    model = CifarCNN(num_classes=10)
    model_weights = model.state_dict()
    client_weights_map = {idx: copy.deepcopy(
        model_weights) for idx in client_idxs}
    for idx in client_weights_map:
        client_weight = client_weights_map[idx]
        for key in client_weight:
            client_weight[key] *= np.random.random()
    agg_weights = np.ones((int(num_clients / 2), )) / num_clients
    adjacency_matrix = update_adjacency_matrix(
        adjacency_matrix, client_idxs, model_weights, client_weights_map, agg_weights, 0.5)
    print(adjacency_matrix)
