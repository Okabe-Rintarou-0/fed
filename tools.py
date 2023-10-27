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
from models import CifarCNN


def pairwise(data):
    """Simple generator of the pairs (x, y) in a tuple such that index x < index y.
    Args:
    data Indexable (including ability to query length) containing the elements
    Returns:
    Generator over the pairs of the elements of 'data'
    """
    n = len(data)
    for i in range(n):
        for j in range(i, n):
            yield (data[i], data[j])


def aggregate_weights(
    w: List[Dict[str, Any]],
    agg_weight: List[float] | torch.Tensor,
    aggregatable_weights: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Returns the average of the weights.
    """
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


def agg_classifier_weighted_p(w, avg_weight, keys, idx):
    """
    Returns the average of the weights.
    """
    w_0 = copy.deepcopy(w[idx])
    for key in keys:
        w_0[key] = torch.zeros_like(w_0[key])
    wc = 0
    for i in range(len(w)):
        wi = avg_weight[i]
        wc += wi
        for key in keys:
            w_0[key] += wi * w[i][key]
    for key in keys:
        w_0[key] = torch.div(w_0[key], wc)
    return w_0


def get_head_agg_weight(num_users, Vars, Hs):
    device = Hs[0][0].device
    num_cls = Hs[0].shape[0]  # number of classes
    d = Hs[0].shape[1]  # dimension of feature representation
    avg_weight = []
    for i in range(num_users):
        # ---------------------------------------------------------------------------
        # variance ter
        v = torch.tensor(Vars, device=device)
        # ---------------------------------------------------------------------------
        # bias term
        h_ref = Hs[i]
        dist = torch.zeros((num_users, num_users), device=device)
        for j1, j2 in pairwise(tuple(range(num_users))):
            h_j1 = Hs[j1]
            h_j2 = Hs[j2]
            h = torch.zeros((d, d), device=device)
            for k in range(num_cls):
                h += torch.mm(
                    (h_ref[k] - h_j1[k]).reshape(d, 1),
                    (h_ref[k] - h_j2[k]).reshape(1, d),
                )
            dj12 = torch.trace(h)
            dist[j1][j2] = dj12
            dist[j2][j1] = dj12

        # QP solver
        p_matrix = torch.diag(v) + dist
        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem
        evals, evecs = torch.linalg.eig(torch.tensor(p_matrix))
        evals = evals.float()
        evecs = evecs.float()

        # for numerical stablity
        p_matrix_new = 0
        p_matrix_new = 0
        for ii in range(num_users):
            if evals[ii] >= 0.01:
                p_matrix_new += evals[ii] * torch.mm(
                    evecs[:, ii].reshape(num_users, 1),
                    evecs[:, ii].reshape(1, num_users),
                )
        p_matrix = (
            p_matrix_new.numpy()
            if not np.all(np.linalg.eigvals(p_matrix) >= 0.0)
            else p_matrix
        )

        # solve QP
        alpha = 0
        eps = 1e-3
        if np.all(np.linalg.eigvals(p_matrix) >= 0):
            alphav = cp.Variable(num_users)
            obj = cp.Minimize(cp.quad_form(alphav, p_matrix))
            prob = cp.Problem(obj, [cp.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(i) * (i > eps) for i in alpha]  # zero-out small weights (<eps)
            if i == 0:
                print("({}) Agg Weights of Classifier Head".format(i + 1))
                print(alpha, "\n")

        else:
            alpha = None  # if no solution for the optimization problem, use local classifier only

        avg_weight.append(alpha)

    return avg_weight


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


def aggregate_dist(
    mus: List[torch.Tensor],
    sigmas: List[torch.Tensor],
    avg_confs: List[torch.Tensor],
    agg_weights: List[float],
    num_classes: int,
):
    agg_weights = torch.tensor(agg_weights)
    agg_weights /= torch.sum(agg_weights)

    agg_mu = torch.zeros_like(mus[0])
    agg_sigma = torch.zeros_like(sigmas[0])

    avg_confs = torch.stack(avg_confs)
    # print(avg_confs.shape)
    for label in range(num_classes):
        avg_conf_weight = F.softmax(avg_confs[:, label], dim=0)
        for i in range(len(mus)):
            agg_mu[label] += avg_conf_weight[i] * agg_weights[i] * mus[i][label]
            agg_sigma[label] += avg_conf_weight[i] * agg_weights[i] * sigmas[i][label]
    return agg_mu, agg_sigma


def weight_flatten(model: Dict[str, Any]):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


def weight_flatten_fc(model: Dict[str, Any]):
    params = []
    for k in model:
        if "fc" in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params


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
        flatten_weights_map[idx] = weight_flatten_fc(dw[idx]).unsqueeze(0)

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


def cal_dist_avg_difference_vector(
    client_idxs: List[int],
    weights_map: Dict[int, Dict[str, Any]],
):
    mus = []
    sigmas = []
    num_clients = len(client_idxs)
    for client_idx in client_idxs:
        this_weights = weights_map[client_idx]
        this_mu = this_weights.get("r.mu")
        this_sigma = this_weights.get("r.sigma")
        mus.append(this_mu)
        sigmas.append(this_sigma)

    dist_avg_difference_vector = []
    for i in range(num_clients):
        mu1 = mus[i]
        sigma1 = sigmas[i]
        kl_div = 0
        for j in range(num_clients):
            if i == j:
                continue
            mu2 = mus[j]
            sigma2 = sigmas[j]
            this_kl_div = (
                torch.log(sigma2 / sigma1)
                + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
                - 0.5
            )
            kl_div += this_kl_div.mean()
        dist_avg_difference_vector.append(kl_div)
    dist_avg_difference_vector = torch.tensor(
        dist_avg_difference_vector, dtype=torch.float
    )
    dist_avg_difference_vector /= torch.max(dist_avg_difference_vector) + 1e-10
    return F.softmax(dist_avg_difference_vector, dim=0)


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


def update_collaborate_vector(
    client_idxs: List[int],
    weights_map: Dict[int, Dict[str, Any]],
    agg_weights: List[float],
    alpha: float,
):
    print("alpha", alpha)
    print("old agg", agg_weights)
    difference_vector = cal_dist_avg_difference_vector(client_idxs, weights_map)
    collabrate_vector = optimize_collaborate_vector(
        difference_vector, alpha, agg_weights
    )
    print("new agg", collabrate_vector)
    return collabrate_vector


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


if __name__ == "__main__":
    num_clients = 10
    adjacency_matrix = torch.ones(num_clients, num_clients) / (num_clients)
    client_idxs = list(range(int(num_clients / 2)))
    model = CifarCNN(num_classes=10)
    model_weights = model.state_dict()
    client_weights_map = {idx: copy.deepcopy(model_weights) for idx in client_idxs}
    for idx in client_weights_map:
        client_weight = client_weights_map[idx]
        for key in client_weight:
            client_weight[key] *= np.random.random()
    agg_weights = np.ones((int(num_clients / 2),)) / num_clients
    adjacency_matrix = update_adjacency_matrix(
        adjacency_matrix,
        client_idxs,
        model_weights,
        client_weights_map,
        agg_weights,
        0.5,
    )
    print(adjacency_matrix)


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    # mat_b = mat_b.to(mat_a.device)
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)
