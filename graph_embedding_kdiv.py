import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import args_kdiverse
import data_generator

np.random.seed(1234567890)
torch.manual_seed(1234567890)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_graph(adj, add_eye=True):
    for i in range(len(adj)):
        adj[i][i] = 0.0

    if (add_eye == True):
        adj_ = adj + np.eye(adj.shape[0])
    else:
        adj_ = adj
    rowsum = np.array(adj_.sum(1))
    deg_mat = np.diag(rowsum)
    degree_mat_inv_sqrt = np.sqrt(np.linalg.inv(deg_mat))
    adj_normalized = np.matmul(np.matmul(degree_mat_inv_sqrt, adj_), degree_mat_inv_sqrt)
    return adj_normalized


def param_init(input_dim, output_dim):
    initial = torch.rand(input_dim, output_dim)
    return nn.Parameter(initial)


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = param_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.matmul(Z, Z.t())
    return A_pred


class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args_kdiverse.ae_input_dim, args_kdiverse.ae_hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(args_kdiverse.ae_hidden1_dim, args_kdiverse.ae_hidden2_dim, adj,
                                        activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return Z, A_pred


def train_GAE_rmsLoss(adj_matrix, add_eye=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    adj_train = adj_matrix
    adj = adj_train

    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    features = np.eye(num_nodes, dtype=np.float)

    if add_eye == True:
        adj_label = adj_train + np.eye(num_nodes)
    else:
        adj_label = adj_train

    adj_norm = torch.FloatTensor(adj_norm).to(device)
    adj_label = torch.FloatTensor(adj_label).to(device)
    features = torch.FloatTensor(features).to(device)

    model = GAE(adj_norm).to(device)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=args_kdiverse.ae_learning_rate)

    def get_rms_loss(adj_rec, adj_label):
        rms_loss = torch.sqrt(torch.sum((adj_label - adj_rec) ** 2))
        return rms_loss.item()

    for epoch in range(args_kdiverse.ae_num_epoch):
        t = time.time()
        optimizer.zero_grad()

        _, A_pred = model(features)
        loss = torch.sum((A_pred - adj_label) ** 2)
        loss.backward()
        optimizer.step()

        train_rms_loss = get_rms_loss(A_pred, adj_label)

        if epoch % 20000 == 0 or epoch == args_kdiverse.ae_num_epoch-1:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()),
                  "train_rms_loss = ", "{:.5f}".format(train_rms_loss),
                  "time=", "{:.5f}".format(time.time() - t))

    print("\n\n")

    Z_final, A_pred_final = model(features)

    return np.array(Z_final.cpu().detach().numpy())


def train_GAE_BCELoss(adj_matrix):
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    adj_train = adj_matrix
    adj = adj_train

    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]
    features = np.eye(num_nodes, dtype=np.float)
    adj_label = adj_train

    adj_norm = torch.FloatTensor(adj_norm).to(device)
    adj_label = torch.FloatTensor(adj_label).to(device)
    features = torch.FloatTensor(features).to(device)

    model = GAE(adj_norm).to(device)


    optimizer = torch.optim.Adadelta(model.parameters(), lr=args_kdiverse.ae_learning_rate)

    for epoch in range(args_kdiverse.ae_num_epoch):
        t = time.time()
        optimizer.zero_grad()

        _, A_pred = model(features)
        loss = F.binary_cross_entropy_with_logits(A_pred.view(-1), adj_label.view(-1))
        loss.backward()
        optimizer.step()

        if epoch % 5000 == 0 or epoch == args_kdiverse.ae_num_epoch - 1:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()))

    print("\n\n")
    Z_final, A_pred_final = model(features)

    # A_pred_final = torch.sigmoid(A_pred_final)
    # A_pred_final = np.array(A_pred_final.detach().numpy())
    # A_pred_final = np.array((A_pred_final >= 0.5), dtype=np.int)
    # A_pred_final = np.reshape(A_pred_final, (-1))
    # adj_orig = np.array(adj_orig, dtype=np.int) + np.eye(adj_orig.shape[0])
    # adj_orig = np.reshape(adj_orig, (-1))
    # conf = metrics.confusion_matrix(adj_orig, A_pred_final)
    # print(conf)

    return np.array(Z_final.cpu().detach().numpy())


def get_POI_embeddings(load_from_file=False):
    if (load_from_file == False):
        args_kdiverse.ae_hidden1_dim = 12
        args_kdiverse.ae_hidden2_dim = 4
        args_kdiverse.ae_num_epoch = 15000
        args_kdiverse.ae_learning_rate = 0.1

        Z_final_cat = train_GAE_BCELoss(data_generator.poi_poi_categories_train_gae)
        Z_final_norm_cat = np.linalg.norm(Z_final_cat, axis=1, keepdims=True)
        Z_final_cat = Z_final_cat / Z_final_norm_cat

        args_kdiverse.ae_hidden1_dim = 20
        args_kdiverse.ae_hidden2_dim = 12
        args_kdiverse.ae_num_epoch = 100000
        args_kdiverse.ae_learning_rate = 0.1

        Z_final_dist = train_GAE_rmsLoss(data_generator.poi_poi_distance_matrix_train_gae)
        Z_final_concat = np.concatenate([Z_final_dist, Z_final_cat], axis=1)

        np.save("model_files/POI_embedding_" + data_generator.dat_suffix[data_generator.dat_ix] + ".npy",
                Z_final_concat)

    Zb = np.load("model_files/POI_embedding_" + data_generator.dat_suffix[data_generator.dat_ix] + ".npy")
    return Zb

#get_POI_embeddings(load_from_file=False)