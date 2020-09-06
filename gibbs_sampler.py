import numpy as np
import lstm_model
import torch
import copy
import random

np.random.seed(1234567890)
torch.manual_seed(1234567890)
random.seed(1234567890)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model_probs(input_seq):
    input_seq_f = list(input_seq)
    input_seq_b = list(reversed(list(input_seq)))

    input_seq_length = [len(input_seq)]
    input_seq_length = torch.LongTensor(input_seq_length).to(device)

    input_seq_batch_f = [input_seq_f]
    input_seq_batch_f = np.array(input_seq_batch_f)
    input_seq_batch_f = np.transpose(input_seq_batch_f)
    input_seq_batch_f = torch.LongTensor(input_seq_batch_f).to(device)

    model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
    output_prb_f = torch.softmax(model_fwd(input_seq_batch_f, input_seq_length), dim=-1)
    output_prb_f = output_prb_f.cpu().detach().numpy()

    input_seq_batch_b = [input_seq_b]
    input_seq_batch_b = np.array(input_seq_batch_b)
    input_seq_batch_b = np.transpose(input_seq_batch_b)
    input_seq_batch_b = torch.LongTensor(input_seq_batch_b).to(device)

    model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
    output_prb_b = torch.softmax(model_bwd(input_seq_batch_b, input_seq_length), dim=-1)
    output_prb_b = output_prb_b.cpu().detach().numpy()

    return output_prb_f, output_prb_b


def get_prob_in_idx(input_seq, I):
    """
    :param seq: list sequence (forward trajectory)
    :param I: index at which probability is going to be generated , F:(s->I-1) , B(e->I+1)
    :return: (vocab_size) sized numpy array
    """

    S = len(list(input_seq))
    output_prb_f, output_prb_b = get_model_probs(input_seq)

    final_op = np.zeros((output_prb_f.shape[1]))

    for i in range(len(final_op)):
        final_op[i] = (I * output_prb_f[I - 1][i] + (S - I - 1) * output_prb_b[S - I - 2][i]) / (S - 1)

    candidates_replacement = (np.argsort(-final_op))

    return final_op, candidates_replacement


def get_traj_perplexity(input_seq):
    """
    :param input_seq:
    :return: raw_prob --> HIGHER BETTER , perplexity --> LOWER IS BETTER
    """

    output_prb_f, output_prb_b = get_model_probs(input_seq)
    fwd_perplexity = 0

    raw_prob_fwd = 1
    raw_prob_bwd = 1

    for i in range(1, len(input_seq)):
        prob_idx = output_prb_f[i - 1][input_seq[i]]
        raw_prob_fwd *= prob_idx
        fwd_perplexity += np.log(prob_idx)

    fwd_perplexity = (-1.0) * fwd_perplexity

    bwd_perplexity = 0
    input_seq_b = list(reversed(list(input_seq)))
    for i in range(1, len(input_seq_b)):
        prob_idx = output_prb_b[i - 1][input_seq_b[i]]
        raw_prob_bwd *= prob_idx
        if prob_idx != 0:
            bwd_perplexity += np.log(prob_idx)
        else:
            bwd_perplexity += -100

    bwd_perplexity = (-1.0) * bwd_perplexity

    perplexity = 0.5 * (fwd_perplexity + bwd_perplexity)
    raw_prob = 0.5 * (raw_prob_fwd + raw_prob_bwd)

    return perplexity, raw_prob


def normalize(x):
    tem = copy.copy(x)
    tem = np.array(tem)
    if np.max(tem) == 0:
        return tem
    return list(tem / np.sum(tem))


def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))


def choose_action(c):
    r = np.random.random()
    c = np.array(c)
    for i in range(1, len(c)):
        c[i] = c[i] + c[i - 1]
    for i in range(len(c)):
        if c[i] >= r:
            return i


def just_acc():
    r = np.random.random()
    if r < 0.2:
        return 0
    else:
        return 1


def gibbs_sampling(barebone_seq, no_samples=150):
    samples_done = 0
    final_candidate_trajectories = []

    curr_seq = list(barebone_seq).copy()
    curr_sta_vec = [1] * len(barebone_seq)

    curr_seq_perp, curr_seq_rawprob = get_traj_perplexity(barebone_seq)
    final_candidate_trajectories.append((barebone_seq, curr_seq_perp))

    def insertion(seq, sta_vec, idx):

        old_seq = seq.copy()
        seq = list(seq)
        seq.insert(idx + 1, 0)
        sta_vec = list(sta_vec)
        sta_vec.insert(idx + 1, 0)

        final_prob, _ = get_prob_in_idx(seq, idx + 1)
        # for i in range(len(old_seq)):
        #     final_prob[old_seq[i]] = 0
        sampled_idx = sample_from_candidate(final_prob)
        seq[idx + 1] = sampled_idx

        return seq, sta_vec

    def replacement(seq, sta_vec, idx):
        old_seq = seq.copy()

        final_prob, _ = get_prob_in_idx(seq, idx)

        for i in range(len(old_seq)):
            final_prob[old_seq[i]] = 0

        sampled_idx = sample_from_candidate(final_prob)
        seq[idx] = sampled_idx

        return seq, sta_vec

    def deletion(seq, sta_vec, idx):

        seq = list(seq)
        seq.pop(idx)
        sta_vec = list(sta_vec)
        sta_vec.pop(idx)

        return seq, sta_vec

    while samples_done < no_samples:

        new_seq = curr_seq.copy()
        new_sta_vec = curr_sta_vec.copy()
        curr_idx = 0
        max_seq_length = 13

        while curr_idx < len(new_seq) - 1:
            if new_sta_vec[curr_idx] == 1:
                if len(new_seq) >= max_seq_length:
                    curr_idx += 1
                elif len(new_seq) == 3:
                    action = choose_action([0.5, 0.5])
                    if action == 0:
                        curr_idx += 1
                    else:
                        new_seq, new_sta_vec = insertion(new_seq, new_sta_vec, curr_idx)
                        curr_idx += 2
                else:
                    action = choose_action([0.7, 0.3])
                    if action == 0:
                        curr_idx += 1
                    else:
                        new_seq, new_sta_vec = insertion(new_seq, new_sta_vec, curr_idx)
                        curr_idx += 2
            else:
                action = 0

                if len(new_seq) >= max_seq_length:
                    action_prb = [0.3, 0.0, 0.6, 0.1]
                    action = choose_action(action_prb)
                    if action == 1:
                        curr_idx += 1
                        continue
                else:
                    action_prb = [0.3, 0.3, 0.3, 0.1]
                    action = choose_action(action_prb)

                if action == 0:
                    new_seq, new_sta_vec = replacement(new_seq, new_sta_vec, curr_idx)
                    curr_idx += 1
                elif action == 1:
                    new_seq, new_sta_vec = insertion(new_seq, new_sta_vec, curr_idx)
                    curr_idx += 2
                elif action == 2:
                    new_seq, new_sta_vec = deletion(new_seq, new_sta_vec, curr_idx)
                else:
                    curr_idx += 1

        new_seq_perp, _ = get_traj_perplexity(new_seq)
        final_candidate_trajectories.append((new_seq, new_seq_perp))
        if new_seq_perp < curr_seq_perp:
            curr_seq = new_seq
            curr_sta_vec = new_sta_vec
            curr_seq_perp = new_seq_perp
        else:
            action = just_acc()
            if action == 0:
                curr_seq = new_seq
                curr_sta_vec = new_sta_vec
                curr_seq_perp = new_seq_perp

        samples_done += 1

    final_candidate_trajectories = sorted(final_candidate_trajectories, key=lambda tuple: tuple[1])
    return final_candidate_trajectories[0][0]
