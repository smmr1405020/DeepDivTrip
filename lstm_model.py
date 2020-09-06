import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import args_kdiverse
import data_generator
import graph_embedding_kdiv
import numpy as np


torch.manual_seed(1234567890)
np.random.seed(1234567890)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TrajPredictor(nn.Module):
    def __init__(self, pretrained_node_embeddings, hidden_size):
        super(TrajPredictor, self).__init__()

        self.embedding = nn.Embedding(len(data_generator.vocab_to_int) - 3,
                                      embedding_dim=pretrained_node_embeddings.shape[1]).from_pretrained(
            pretrained_node_embeddings,
            freeze=False)
        self.embedding_dim = pretrained_node_embeddings.shape[1]
        self.vocab_size = pretrained_node_embeddings.shape[0]
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=self.embedding_dim,
                               hidden_size=self.hidden_size, num_layers=1)

        self.linear_inp_size = self.hidden_size + self.embedding_dim
        self.fc1 = nn.Linear(self.linear_inp_size, 30)
        self.fc2 = nn.Linear(30, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, seq, seq_lengths):
        embeds = self.embedding(seq)
        lstm_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths, batch_first=False)

        poi_emb_unsqueezed = torch.unsqueeze(torch.unsqueeze(self.embedding.weight, 0), 0)
        poi_emb_repeated = poi_emb_unsqueezed.repeat(seq.shape[0], seq.shape[1], 1, 1)

        output, (hidden, _) = self.encoder(lstm_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        output_unsqueezed = torch.unsqueeze(output, 2)
        output_repeated = output_unsqueezed.repeat(1, 1, poi_emb_repeated.shape[2], 1)
        poi_out_concat = torch.cat([poi_emb_repeated, output_repeated], dim=3)
        out_1 = torch.relu(self.fc1(poi_out_concat))
        out_2 = self.fc2(out_1)
        out_sq = torch.squeeze(out_2)

        return out_sq


def loss_fn(model, pred, target):
    return torch.nn.CrossEntropyLoss()(torch.transpose(pred, 1, 2), target)


def print_all(model, backward_model):
    if backward_model == False:
        tr_s, _ = data_generator.get_trajectory_dataset()
    else:
        _,  tr_s = data_generator.get_trajectory_dataset()

    dataset_trajectory = tr_s

    for i in range(dataset_trajectory.no_training_batches(args_kdiverse.lstm_model_batch_size)):
        inputs, seq_lengths, targets = dataset_trajectory(i, args_kdiverse.lstm_model_batch_size)
        inputs = torch.LongTensor(inputs)
        seq_lengths = torch.LongTensor(seq_lengths)

        targets = torch.LongTensor(targets)
        output = torch.softmax(model(inputs, seq_lengths), dim=2)

        op = torch.transpose(output, 0, 1)
        op = op.detach().numpy()
        op = np.argmax(op, axis=2)

        tgt = torch.transpose(targets, 0, 1)
        tgt = tgt.numpy()

        print("PRED:")
        print(op)
        print("GT:")
        print(tgt)
        print("\n")



def train(model, optimizer, loss_fn, epochs=100, backward_model=False):
    train_loss_min = 100000.0
    for epoch in range(epochs):
        training_loss = 0.0

        if backward_model == False:
            tr_s, _ = data_generator.get_trajectory_dataset()
        else:
             _, tr_s = data_generator.get_trajectory_dataset()

        model.train()

        for i in range(tr_s.no_training_batches(args_kdiverse.lstm_model_batch_size)):
            inputs, seq_lengths, targets = tr_s(i, args_kdiverse.lstm_model_batch_size)
            inputs = torch.LongTensor(inputs).to(device)
            seq_lengths = torch.LongTensor(seq_lengths).to(device)
            targets = torch.LongTensor(targets).to(device)
            optimizer.zero_grad()
            output = model(inputs, seq_lengths)
            loss = loss_fn(model, output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            training_loss += loss.data.item()

        training_loss /= tr_s.no_training_batches(args_kdiverse.lstm_model_batch_size)

        if training_loss < train_loss_min:
            train_loss_min = training_loss
            if not backward_model:
                torch.save(model.state_dict(),
                           "model_files/LSTM_net_1_f_" + data_generator.dat_suffix[data_generator.dat_ix])
            else:
                torch.save(model.state_dict(),
                           "model_files/LSTM_net_1_b_" + data_generator.dat_suffix[data_generator.dat_ix])

        if epoch % 100 == 0 or epoch == epochs -1:
            print('Epoch: {}, Loss: {:.3f}'.format(epoch, training_loss))

def get_forward_lstm_model(load_from_file=True):
    pretrained_embeddings = graph_embedding_kdiv.get_POI_embeddings(load_from_file=True)
    pretrained_embeddings = torch.FloatTensor(pretrained_embeddings).to(device)
    if (load_from_file == False):
        trajpredictor_forward = TrajPredictor(pretrained_embeddings, args_kdiverse.lstm_model_hidden_size).to(device)
        optimizer_forward = optim.Adam(trajpredictor_forward.parameters(), lr=0.001)
        print("\nForward")
        train(trajpredictor_forward, optimizer_forward, loss_fn, epochs=500)
        print("\n")
    forward_lstm_model = TrajPredictor(pretrained_embeddings, args_kdiverse.lstm_model_hidden_size).to(device)
    fwd_model_state_dict = torch.load("model_files/LSTM_net_1_f_" + data_generator.dat_suffix[data_generator.dat_ix])
    forward_lstm_model.load_state_dict(fwd_model_state_dict)

    return forward_lstm_model


def get_backward_lstm_model(load_from_file=True):
    pretrained_embeddings = graph_embedding_kdiv.get_POI_embeddings(load_from_file=True)
    pretrained_embeddings = torch.FloatTensor(pretrained_embeddings).to(device)
    if (load_from_file == False):
        trajpredictor_backward = TrajPredictor(pretrained_embeddings, args_kdiverse.lstm_model_hidden_size).to(device)
        optimizer_backward = optim.Adam(trajpredictor_backward.parameters(), lr=0.001)
        print("\nBackward")
        train(trajpredictor_backward, optimizer_backward, loss_fn, epochs=500, backward_model=True)
        print("\n")
    backward_lstm_model = TrajPredictor(pretrained_embeddings, args_kdiverse.lstm_model_hidden_size).to(device)
    bwd_model_state_dict = torch.load("model_files/LSTM_net_1_b_" + data_generator.dat_suffix[data_generator.dat_ix])
    backward_lstm_model.load_state_dict(bwd_model_state_dict)

    return backward_lstm_model


# get_forward_lstm_model(load_from_file=False)
# get_backward_lstm_model(load_from_file=False)