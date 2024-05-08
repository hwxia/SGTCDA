import torch
import torch.optim as optim
from utils import *
from model import SGTCDA
from ge import SDNE
import networkx as nx

def test(edge_idx_dict, n_drug, n_cir, drug_sim, cir_sim, args_config, device):
    lr = args_config['lr']
    weight_decay = args_config['weight_decay']
    num_epoch = 100
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

    temp_drug_cir = np.zeros((n_drug, n_cir))

    # !!!!!
    # Note: Put only positive samples from the training set into the association matrix
    temp_drug_cir[train_pos_edges[0], train_pos_edges[1]] = 1

    drug_sim, cir_sim = get_syn_sim(temp_drug_cir, drug_sim, cir_sim, 1)
    drug_adj = k_matrix(drug_sim, knn_nums)
    cir_adj = k_matrix(cir_sim, knn_nums)
    edge_idx_drug, edge_idx_cir = np.array(tuple(np.where(drug_adj != 0))), np.array(tuple(np.where(cir_adj != 0)))
    edge_idx_drug = torch.tensor(edge_idx_drug, dtype=torch.long,
                                 device=device)
    edge_idx_cir = torch.tensor(edge_idx_cir, dtype=torch.long,
                                device=device)

    model = SGTCDA(
        n_drug + n_cir, num_embedding_features, heads, n_drug, n_cir, dropout).to(device)
    num_u, num_v = n_drug, n_cir
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    base_lr = 5e-5
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr=lr, step_size_up=200,
                                            step_size_down=200, mode='exp_range', gamma=0.99, scale_fn=None,
                                            cycle_momentum=False, last_epoch=-1)


    het_mat = construct_het_mat(temp_drug_cir, cir_sim, drug_sim)
    adj_mat = construct_adj_mat(temp_drug_cir)

    drug_sim = torch.tensor(drug_sim, dtype=torch.float, device=device)
    cir_sim = torch.tensor(cir_sim, dtype=torch.float, device=device)

    edge_idx_device = torch.tensor(np.where(adj_mat == 1), dtype=torch.long, device=device)
    het_mat_device = torch.tensor(het_mat, dtype=torch.float32, device=device)

    # SDNE algorithm
    drug_adjacency_matrix = np.where(drug_adj != 0, 1, drug_adj)
    # np.fill_diagonal(drug_adjacency_matrix, 0)
    cir_adjacency_matrix = np.where(cir_adj != 0, 1, cir_adj)
    # np.fill_diagonal(cir_adjacency_matrix, 0)

    G_drug = nx.Graph(drug_adjacency_matrix)
    G_cir = nx.Graph(cir_adjacency_matrix)

    model_drug = SDNE(G_drug, hidden_size=[256, 128], )
    model_cir = SDNE(G_cir, hidden_size=[256, 128], )

    model_drug.train(batch_size=500, epochs=300, verbose=0)
    model_cir.train(batch_size=500, epochs=300, verbose=0)

    embeddings_drug = model_drug.get_embeddings()
    embeddings_cir = model_cir.get_embeddings()

    embeddings_drug = np.vstack(list(embeddings_drug.values()))
    embeddings_cir = np.vstack(list(embeddings_cir.values()))
    embeddings_drug = torch.tensor(embeddings_drug, dtype=torch.float, device=device)
    embeddings_cir = torch.tensor(embeddings_cir, dtype=torch.float, device=device)

    # train
    for epoch in range(num_epoch):
        model.train()
        pred_mat = model(het_mat_device, edge_idx_device, embeddings_drug, edge_idx_drug, embeddings_cir, edge_idx_cir).cpu().reshape(
            num_u, num_v)

        loss = calculate_loss(pred_mat, train_pos_edges, train_neg_edges)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print('------EOPCH {} of {}------'.format(epoch + 1, num_epoch))
            print('Loss: {}'.format(loss))


    # test
    model.eval()
    with torch.no_grad():
        pred_mat = model(het_mat_device, edge_idx_device, embeddings_drug, edge_idx_drug, embeddings_cir,
                         edge_idx_cir).cpu().reshape(num_u, num_v)
        metrics = calculate_evaluation_metrics(pred_mat.detach(), test_pos_edges, test_neg_edges)
        print('Independent Test Set：Auc = ', metrics[0], '\t  Aupr = ', metrics[1], '\t  f1_score = ', metrics[2],
              '\t  Accuracy = ', metrics[3], '\t  Recall = ', metrics[4], '\t  Specificity = ', metrics[5],
              '\t  Precision = ', metrics[6])
        return pred_mat