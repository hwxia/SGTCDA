from sklearn.model_selection import KFold
from test import *

def main(edge_idx_dict, n_drug, n_cir, drug_sim, cir_sim, args_config, device):

    lr = args_config['lr']
    weight_decay = args_config['weight_decay']
    kfolds = args_config['kfolds']
    num_epoch = args_config['num_epoch']
    knn_nums = args_config['knn_nums']
    heads = args_config['heads']
    num_embedding_features = args_config['num_embedding_features']
    dropout = args_config['dropout']

    # define the dataset
    pos_edges = edge_idx_dict['pos_edges']
    neg_edges = edge_idx_dict['neg_edges']
    train_pos_edges = pos_edges[:, :int(pos_edges.shape[1] * 0.8)]
    train_neg_edges = neg_edges[:, :int(neg_edges.shape[1] * 0.8)]
    test_pos_edges = pos_edges[:, int(pos_edges.shape[1] * 0.8):]
    test_neg_edges = neg_edges[:, int(neg_edges.shape[1] * 0.8):]

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=256)
    
    model = SGTCDA(
        n_drug + n_cir, num_embedding_features, heads, n_drug, n_cir, dropout).to(device)
    num_u, num_v = n_drug, n_cir
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_lr = 5e-5
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr=lr, step_size_up=200,
                                            step_size_down=200, mode='exp_range', gamma=0.99, scale_fn=None,
                                            cycle_momentum=False, last_epoch=-1)

    # training set for data processing
    train_dataset = construct_train_dataset(train_pos_edges, train_neg_edges)

    i = 1
    metrics_list1 = []
    metrics_list2 = []
    metrics_list3 = []
    metrics_list4 = []
    metrics_list5 = []
    total_result = []

    for train_index, val_index in kf.split(train_dataset):
        train_data = train_dataset[train_index]
        val_data = train_dataset[val_index]

        train_pos_data = train_data.T[:2, :]
        train_neg_data = train_data.T[3:5, :]

        val_pos_data = val_data.T[:2, :]
        val_neg_data = val_data.T[3:5, :]

        temp_drug_cir = np.zeros((n_drug, n_cir))
        # !!!!!
        # Note: Put only positive samples from the training set into the association matrix
        temp_drug_cir[train_pos_data[0], train_pos_data[1]] = 1

        drug_integrate_sim, cir_integrate_sim = get_syn_sim(temp_drug_cir, drug_sim, cir_sim, 1)
        drug_adj = k_matrix(drug_integrate_sim, knn_nums)
        cir_adj = k_matrix(cir_integrate_sim, knn_nums)

        edge_idx_drug, edge_idx_cir = np.array(tuple(np.where(drug_adj != 0))), np.array(tuple(np.where(cir_adj != 0)))
        edge_idx_drug = torch.tensor(edge_idx_drug, dtype=torch.long,
                                     device=device)
        edge_idx_cir = torch.tensor(edge_idx_cir, dtype=torch.long,
                                    device=device)

        het_mat = construct_het_mat(temp_drug_cir, cir_integrate_sim, drug_integrate_sim)
        adj_mat = construct_adj_mat(temp_drug_cir)

        drug_integrate_sim = torch.tensor(drug_integrate_sim, dtype=torch.float, device=device)
        cir_integrate_sim = torch.tensor(cir_integrate_sim, dtype=torch.float, device=device)

        edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
        het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)


        # SDNE algorithm
        drug_adjacency_matrix = np.where(drug_adj != 0, 1, drug_adj)
        # np.fill_diagonal(drug_adjacency_matrix, 0)
        cir_adjacency_matrix = np.where(cir_adj != 0, 1, cir_adj)
        # np.fill_diagonal(cir_adjacency_matrix, 0)

        G_drug = nx.Graph(drug_adjacency_matrix)
        G_cir = nx.Graph(cir_adjacency_matrix)

        model_drug = SDNE(G_drug, hidden_size=[256, 128],)
        model_cir = SDNE(G_cir, hidden_size=[256, 128],)

        model_drug.train(batch_size=500, epochs=300, verbose=0)
        model_cir.train(batch_size=500, epochs=300, verbose=0)

        embeddings_drug = model_drug.get_embeddings()
        embeddings_cir = model_cir.get_embeddings()

        embeddings_drug = np.vstack(list(embeddings_drug.values()))
        embeddings_cir = np.vstack(list(embeddings_cir.values()))
        embeddings_drug = torch.tensor(embeddings_drug, dtype=torch.float, device=device)
        embeddings_cir = torch.tensor(embeddings_cir, dtype=torch.float, device=device)


        print('第', i, '折')
        for epoch in range(num_epoch):
            model.train()
            pred_mat = model(het_mat_device, edge_idx_device, embeddings_drug, edge_idx_drug, embeddings_cir, edge_idx_cir).cpu().reshape(
                num_u, num_v)
            loss = calculate_loss(pred_mat, train_pos_data, train_neg_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # validate
            model.eval()
            with torch.no_grad():
                pred_mat = model(het_mat_device, edge_idx_device, embeddings_drug, edge_idx_drug, embeddings_cir,
                                 edge_idx_cir).cpu().reshape(num_u, num_v)  # pred_mat的大小为(218,271)
                metrics = calculate_evaluation_metrics(pred_mat.detach(), val_pos_data, val_neg_data)
                auc = metrics[0]
                aupr = metrics[1]
                f1_score = metrics[2]
                accuracy = metrics[3]
                recall = metrics[4]
                specificity = metrics[5]
                precision = metrics[6]

                print('Auc = ', auc, '\t  Aupr = ', aupr, '\t  f1_score = ', f1_score,
                          '\t  Accuracy = ', accuracy, '\t  Recall = ', recall,
                      '\t  Specificity = ', specificity, '\t  Precision = ', precision)
                if (i == 1):
                    metrics_list1.append(metrics)
                elif (i == 2):
                    metrics_list2.append(metrics)
                elif (i == 3):
                    metrics_list3.append(metrics)
                elif (i == 4):
                    metrics_list4.append(metrics)
                elif (i == 5):
                    metrics_list5.append(metrics)

        i += 1

    for num in range(num_epoch):
        array = np.array([ metrics_list1[num], metrics_list2[num], metrics_list3[num]
                             , metrics_list4[num], metrics_list5[num]])
        epoch_folds_mean = np.mean(array, axis=0).tolist()
        print(epoch_folds_mean)
        total_result.append(epoch_folds_mean)


    max_auc = 0
    max_index = -1
    for index in range(num_epoch):
        if total_result[index][0] > max_auc:
            max_auc = total_result[index][0]
            max_index = index
    print('5folds求得最终的轮数为', max_index + 1)
    print(total_result[max_index])




if __name__ == '__main__':
    set_seed(666)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hyperparam_dict = {
        'kfolds':5,
        'heads': 2,
        'num_embedding_features':256,
        'num_epoch': 300,
        'knn_nums':25,
        'lr': 1e-3,
        'weight_decay': 5e-3,
        'dropout':0.1
    }

    # load data
    drug_sim, cir_sim, edge_idx_dict, drug_dis_matrix = load_data()

    diag = np.diag(cir_sim)
    if np.sum(diag) != 0:
        cir_sim = cir_sim - np.diag(diag)
    diag = np.diag(drug_sim)
    if np.sum(diag) != 0:
        drug_sim = drug_sim - np.diag(diag)

    # 5-fold cross validation
    pred_mat = main(edge_idx_dict, drug_sim.shape[0], cir_sim.shape[0], drug_sim, cir_sim, hyperparam_dict, device)

    # Independent Test Set Testing
    # pred_mat_ = test(edge_idx_dict, drug_sim.shape[0], cir_sim.shape[0], drug_sim, cir_sim, hyperparam_dict, device)