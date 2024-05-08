from torch_geometric.nn import GCNConv
from layers import *



class SGTCDA(nn.Module):
    def __init__(self, n_in_features: int, hid_features: int, heads: int,
                 n_drug: int, n_cir: int, dropout: float):
        super(SGTCDA, self).__init__()

        self.n_in_features = n_in_features
        self.hid_features = hid_features
        self.heads = heads
        self.n_drug = n_drug
        self.n_cir = n_cir
        self.dropout = nn.Dropout(dropout)


        self.reconstructions = InnerProductDecoder(
            name='gan_decoder',
            input_dim=hid_features, num_d=self.n_drug, act=torch.sigmoid)

        self.CNN_hetero = nn.Conv2d(in_channels=2,
                                    out_channels=hid_features,
                                    kernel_size=(hid_features, 1),
                                    stride=1,
                                    bias=True)
        self.CNN_drug = nn.Conv2d(in_channels=2,
                                  out_channels=hid_features,
                                  kernel_size=(hid_features, 1),
                                  stride=1,
                                  bias=True)
        self.CNN_dis = nn.Conv2d(in_channels=2,
                                 out_channels=hid_features,
                                 kernel_size=(hid_features, 1),
                                 stride=1,
                                 bias=True)

        self.gcn_hetro1 = GCNConv(in_channels=self.n_in_features-1, out_channels=512)
        self.gcn_drug1 = GCNConv(in_channels=128, out_channels=512)
        self.gcn_cir1 = GCNConv(in_channels=128, out_channels=512)

        self.hetero_transformerEncoder = Encoder(d_model=488, n_heads=2, n_layers=3)
        self.drug_transformerEncoder = Encoder(d_model=128, n_heads=2, n_layers=3)
        self.circ_transformerEncoder = Encoder(d_model=128, n_heads=2, n_layers=3)


    def forward(self, x, edge_idx, x_drug, edge_idx_drug, x_cir, edge_idx_cir):
        # encoder

        embd_tmp = x
        embd_tmp = embd_tmp[:, :-1]
        embd_hetro = self.hetero_transformerEncoder(embd_tmp)
        hetro_gcn1 = torch.relu(self.gcn_hetro1(embd_hetro, edge_idx))
        cnn_embd_hetro = hetro_gcn1.t().view(1, 2, self.hid_features,
                                                 self.n_drug + self.n_cir)
        cnn_embd_hetro = self.CNN_hetero(cnn_embd_hetro)
        cnn_embd_hetro = cnn_embd_hetro.view(self.hid_features, self.n_drug + self.n_cir).t()



        embd_tmp_drug = x_drug
        embd_drug = self.drug_transformerEncoder(embd_tmp_drug)
        drug_gcn1 = torch.relu(self.gcn_drug1(embd_drug, edge_idx_drug))
        cnn_embd_drug = drug_gcn1.t().view(1, 2, self.hid_features, self.n_drug)
        cnn_embd_drug = self.CNN_drug(cnn_embd_drug)
        cnn_embd_drug = cnn_embd_drug.view(self.hid_features, self.n_drug).t()


        embd_tmp_cir = x_cir
        embd_cir = self.circ_transformerEncoder(embd_tmp_cir)
        cir_gcn1 = torch.relu(self.gcn_cir1(embd_cir, edge_idx_cir))
        cnn_embd_cir = cir_gcn1.t().view(1, 2, self.hid_features, self.n_cir)
        cnn_embd_cir = self.CNN_dis(cnn_embd_cir)
        cnn_embd_cir = cnn_embd_cir.view(self.hid_features, self.n_cir).t()


        embd_heter = cnn_embd_hetro
        embd_drug = cnn_embd_drug
        embd_cir = cnn_embd_cir


        final_embd = self.dropout(embd_heter)
        final_embd_drug = self.dropout(embd_drug)
        final_embd_cir= self.dropout(embd_cir)
     
        ret=self.reconstructions(final_embd,final_embd_cir,final_embd_drug)
        return ret