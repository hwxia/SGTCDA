import torch
import numpy as np
from tqdm import tqdm
from numpy.random import default_rng

def drug_edgeshaper(model, x, edge_idx, x_drug, edge_idx_drug, x_cir, edge_idx_cir, drug_dis_matrix, M=100, P=None, seed=42):

    rng = default_rng(seed=seed)
    model.eval()
    phi_edges = []
    drug_i = []
    drug_j = []

    edge_idx_drug_numpy = edge_idx_drug.cpu().numpy()
    num_nodes = x_drug.shape[0]
    num_edges = edge_idx_drug.shape[1]

    if P is None:
        max_num_edges = num_nodes * (num_nodes - 1)
        graph_density = num_edges / max_num_edges
        P = graph_density


    for j in tqdm(range(num_edges)):
        drug_i.append(edge_idx_drug_numpy[0][j])
        drug_j.append(edge_idx_drug_numpy[1][j])
        marginal_contrib = 0
        for i in range(M):
            E_z_mask = rng.binomial(1, P, num_edges)
            E_mask = torch.ones(num_edges)
            pi = torch.randperm(num_edges)

            E_j_plus_index = torch.ones(num_edges, dtype=torch.int)
            E_j_minus_index = torch.ones(num_edges, dtype=torch.int)
            selected_edge_index = pi[j].item()
            for k in range(num_edges):
                if k <= selected_edge_index:
                    E_j_plus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_plus_index[pi[k]] = E_z_mask[pi[k]]

            for k in range(num_edges):
                if k < selected_edge_index:
                    E_j_minus_index[pi[k]] = E_mask[pi[k]]
                else:
                    E_j_minus_index[pi[k]] = E_z_mask[pi[k]]

            retained_indices_plus = torch.nonzero(E_j_plus_index).squeeze()
            retained_indices_minus = torch.nonzero(E_j_minus_index).squeeze()

            E_j_plus = edge_idx_drug[:, retained_indices_plus].to('cuda')
            E_j_minus = edge_idx_drug[:, retained_indices_minus].to('cuda')

            out_plus = model(x, edge_idx, x_drug, E_j_plus, x_cir, edge_idx_cir).cpu().reshape(218, 271).detach().numpy()
            zero_indices = np.where(drug_dis_matrix == 0)
            out_plus[zero_indices] = 0

            out_minus = model(x, edge_idx, x_drug, E_j_minus, x_cir, edge_idx_cir).cpu().reshape(218, 271).detach().numpy()
            zero_indices = np.where(drug_dis_matrix == 0)
            out_minus[zero_indices] = 0

            V_j_plus = np.sum(out_plus)
            V_j_minus = np.sum(out_minus)

            marginal_contrib += (V_j_plus - V_j_minus)

            del E_z_mask, E_mask, pi, E_j_plus_index, E_j_minus_index, selected_edge_index, retained_indices_plus, \
                retained_indices_minus, E_j_plus, E_j_minus, out_plus, out_minus, V_j_plus, V_j_minus
            torch.cuda.empty_cache()


        phi_edges.append((marginal_contrib / M))


    return phi_edges, drug_i, drug_j