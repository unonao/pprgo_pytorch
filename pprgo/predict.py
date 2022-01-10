import time
import numpy as np
import torch

from .pytorch_utils import matrix_to_torch


def get_local_logits(model, attr_matrix, batch_size=10000):
    device = next(model.parameters()).device

    nnodes = attr_matrix.shape[0]
    logits = []
    with torch.set_grad_enabled(False):
        for i in range(0, nnodes, batch_size):
            batch_attr = matrix_to_torch(attr_matrix[i:i + batch_size]).to(device)
            logits.append(model(batch_attr).to('cpu').numpy())
    logits = np.row_stack(logits)
    return logits


def predict(model, adj_matrix, attr_matrix, alpha,
            nprop=2, inf_fraction=1.0, ppr_normalization='sym', batch_size_logits=10000, use_argmax=True):

    model.eval()

    start = time.time()
    if inf_fraction < 1.0:
        idx_sub = sorted(np.random.choice(adj_matrix.shape[0], int(inf_fraction * adj_matrix.shape[0]), replace=False))
        attr_sub = attr_matrix[idx_sub]
        logits_sub = get_local_logits(model.mlp, attr_sub, batch_size_logits)
        local_logits = np.zeros([adj_matrix.shape[0], logits_sub.shape[1]], dtype=np.float32)
        local_logits[idx_sub] = logits_sub
    else:
        local_logits = get_local_logits(model.mlp, attr_matrix, batch_size_logits)
    time_logits = time.time() - start

    start = time.time()
    row, col = adj_matrix.nonzero()
    logits = local_logits.copy()

    if ppr_normalization == 'sym':
        # Assume undirected (symmetric) adjacency matrix
        deg = adj_matrix.sum(1).A1
        deg_sqrt_inv = 1. / np.sqrt(np.maximum(deg, 1e-12))
        for _ in range(nprop):
            logits = (1 - alpha) * deg_sqrt_inv[:, None] * (adj_matrix @ (deg_sqrt_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'col':
        deg_col = adj_matrix.sum(0).A1
        deg_col_inv = 1. / np.maximum(deg_col, 1e-12)
        for _ in range(nprop):
            logits = (1 - alpha) * (adj_matrix @ (deg_col_inv[:, None] * logits)) + alpha * local_logits
    elif ppr_normalization == 'row':
        deg_row = adj_matrix.sum(1).A1
        deg_row_inv_alpha = (1 - alpha) / np.maximum(deg_row, 1e-12)
        for _ in range(nprop):
            logits = deg_row_inv_alpha[:, None] * (adj_matrix @ logits) + alpha * local_logits
    else:
        raise ValueError(f"Unknown PPR normalization: {ppr_normalization}")
    if use_argmax:
        predictions = logits.argmax(1)
    else:
        predictions = logits
    time_propagation = time.time() - start
    return predictions, time_logits, time_propagation


def apply_linear(model, hidden_features, batch_size=10000):
    device = next(model.parameters()).device

    nnodes = hidden_features.shape[0]
    logits = []
    with torch.set_grad_enabled(False):
        for i in range(0, nnodes, batch_size):
            batch_hidden_features = matrix_to_torch(hidden_features[i:i + batch_size]).to(device)
            logits.append(model(batch_hidden_features).squeeze().to('cpu').numpy())
    logits = np.row_stack(logits)
    return logits


def predict_sappr(model, adj_matrix, attr_matrix, alphas,
                  nprop=2, inf_fraction=1.0, ppr_normalization='sym', batch_size_logits=10000):
    ''' それぞれのalphaごとにpredictionsを出力し、SAPPRGo の最終出力層
    '''

    model.eval()
    logits_list = []
    time_logits_list = []
    time_propagation_list = []
    for alpha in alphas:
        logits, time_logits, time_propagation = predict(model, adj_matrix, attr_matrix, alpha,
                                                             nprop=nprop, inf_fraction=inf_fraction,
                                                             ppr_normalization=ppr_normalization,
                                                             batch_size_logits=batch_size_logits,
                                                             use_argmax=False)
        logits_list.append(torch.tensor(logits).float())
        time_logits_list.append(time_logits)
        time_propagation_list.append(time_propagation)
    logits_torch = torch.stack(logits_list, dim=2)
    logits = apply_linear(model.linear_before_squeeze, logits_torch)
    predictions = logits.argmax(1)
    return predictions, np.sum(time_logits_list), np.sum(time_propagation_list)
