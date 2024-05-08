from calculate_shap import *



if __name__ == '__main__':
    set_seed(666)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparam_dict = {
        'kfolds':5,
        'heads': 2,
        'num_embedding_features':256,
        'num_epoch': 300,
        'knn_nums':1,
        'knn_nums_25':25,
        'lr': 1e-3,
        'weight_decay': 5e-3,
        'dropout':0.1
    }

    drug_sim, cir_sim, edge_idx_dict, drug_dis_matrix = load_data()

    diag = np.diag(cir_sim)
    if np.sum(diag) != 0:
        cir_sim = cir_sim - np.diag(diag)
    diag = np.diag(drug_sim)
    if np.sum(diag) != 0:
        drug_sim = drug_sim - np.diag(diag)


    # calculate shap value
    pred_mat_ = calculate_shap(edge_idx_dict, drug_dis_matrix, drug_sim.shape[0], cir_sim.shape[0], drug_sim, cir_sim, hyperparam_dict, device)