import pprint
import random
import numpy as np
import csv
import kfold_dataset_generator

random.seed(1234567890)
np.random.seed(1234567890)

dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb', 'caliAdv', 'disHolly', 'disland', 'epcot', 'MagicK']
dat_ix = kfold_dataset_generator.dat_ix
embedding_name = dat_suffix[dat_ix]
poi_name = "poi-" + dat_suffix[dat_ix] + ".csv"  # Edin
tra_name = "traj-" + dat_suffix[dat_ix] + ".csv"

pp = pprint.PrettyPrinter(indent=4, width=180)

# =============================== data load ====================================== #
op_tdata = open('origin_data/' + poi_name, 'r')
ot_tdata = open('origin_data/' + tra_name, 'r')

print('To Train', dat_suffix[dat_ix])

POIs = []
Trajectory = []

for line in op_tdata.readlines():
    lineArr = line.split(',')
    temp_line = list()
    for item in lineArr:
        temp_line.append(item.strip('\n'))
    POIs.append(temp_line)
POIs = POIs[1:]

ALL_POI_IDS = []

for i in range(len(POIs)):
    ALL_POI_IDS.append(int(POIs[i][0]))


def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin(np.sqrt(
        (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
    return dist


#####################   Generate Raw Data   ##################################################


query_dict_trajectory_train = dict()
query_dict_users_train = dict()
query_dict_traj_ids_train = dict()
query_dict_traj_time_train = dict()
query_dict_freq_train = dict()


def isSame(traj_a, traj_b):
    if (len(traj_a) != len(traj_b)):
        return False
    for i in range(len(traj_a)):
        if (traj_a[i] != traj_b[i]):
            return False

    return True


with open("processed_data/" + embedding_name + '_train_set.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    line_no = 0

    row0 = []
    row1 = []
    row2 = []

    for row in csv_reader:

        if (line_no == 0):
            row0 = row.copy()
        elif (line_no == 1):
            row1 = row.copy()
        elif (line_no == 2):
            row2 = row.copy()

            curr_traj = [int(poi) for poi in row0]
            st_poi = curr_traj[0]
            ed_poi = curr_traj[-1]
            qu = str(st_poi) + "-" + str(ed_poi)

            curr_traj_time = [int(poi_time) for poi_time in row1]
            curr_user = row2[0]
            curr_traj_id = int(row2[1])

            gotBefore = False
            all_traj_pos = -1

            if (qu in query_dict_trajectory_train.keys()):
                all_traj = query_dict_trajectory_train[qu]
                for prev_traj_itr in range(len(all_traj)):
                    if (isSame(all_traj[prev_traj_itr], curr_traj)):
                        gotBefore = True
                        all_traj_pos = prev_traj_itr
                        break

                if (gotBefore == False):

                    all_traj.append(curr_traj)
                    query_dict_trajectory_train[qu] = all_traj

                    all_u = query_dict_users_train[qu]
                    all_u.append([curr_user])
                    query_dict_users_train[qu] = all_u

                    all_traj_id = query_dict_traj_ids_train[qu]
                    all_traj_id.append([curr_traj_id])
                    query_dict_traj_ids_train[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_train[qu]
                    all_traj_time.append([curr_traj_time])
                    query_dict_traj_time_train[qu] = all_traj_time

                    all_freq = query_dict_freq_train[qu]
                    all_freq.append(1)
                    query_dict_freq_train[qu] = all_freq

                else:

                    all_u = query_dict_users_train[qu]
                    all_u[all_traj_pos].append(curr_user)
                    query_dict_users_train[qu] = all_u

                    all_traj_id = query_dict_traj_ids_train[qu]
                    all_traj_id[all_traj_pos].append(curr_traj_id)
                    query_dict_traj_ids_train[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_train[qu]
                    all_traj_time[all_traj_pos].append(curr_traj_time)
                    query_dict_traj_time_train[qu] = all_traj_time

                    all_freq = query_dict_freq_train[qu]
                    all_freq[all_traj_pos] += 1
                    query_dict_freq_train[qu] = all_freq

            else:

                query_dict_trajectory_train.setdefault(qu, []).append(curr_traj)
                query_dict_users_train.setdefault(qu, []).append([curr_user])
                query_dict_traj_ids_train.setdefault(qu, []).append([curr_traj_id])
                query_dict_traj_time_train.setdefault(qu, []).append([curr_traj_time])
                query_dict_freq_train.setdefault(qu, []).append(1)

        line_no = (line_no + 1) % 3

query_dict_trajectory_test = dict()
query_dict_users_test = dict()
query_dict_traj_ids_test = dict()
query_dict_traj_time_test = dict()
query_dict_freq_test = dict()

with open("processed_data/" + embedding_name + '_test_set.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    line_no = 0

    row0 = []
    row1 = []
    row2 = []

    for row in csv_reader:

        if (line_no == 0):
            row0 = row.copy()
        elif (line_no == 1):
            row1 = row.copy()
        elif (line_no == 2):
            row2 = row.copy()

            curr_traj = [int(poi) for poi in row0]
            st_poi = curr_traj[0]
            ed_poi = curr_traj[-1]
            qu = str(st_poi) + "-" + str(ed_poi)

            curr_traj_time = [int(poi_time) for poi_time in row1]
            curr_user = row2[0]
            curr_traj_id = int(row2[1])

            gotBefore = False
            all_traj_pos = -1

            if (qu in query_dict_trajectory_test.keys()):
                all_traj = query_dict_trajectory_test[qu]
                for prev_traj_itr in range(len(all_traj)):
                    if (isSame(all_traj[prev_traj_itr], curr_traj)):
                        gotBefore = True
                        all_traj_pos = prev_traj_itr
                        break

                if (gotBefore == False):

                    all_traj.append(curr_traj)
                    query_dict_trajectory_test[qu] = all_traj

                    all_u = query_dict_users_test[qu]
                    all_u.append([curr_user])
                    query_dict_users_test[qu] = all_u

                    all_traj_id = query_dict_traj_ids_test[qu]
                    all_traj_id.append([curr_traj_id])
                    query_dict_traj_ids_test[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_test[qu]
                    all_traj_time.append([curr_traj_time])
                    query_dict_traj_time_test[qu] = all_traj_time

                    all_freq = query_dict_freq_test[qu]
                    all_freq.append(1)
                    query_dict_freq_test[qu] = all_freq

                else:

                    all_u = query_dict_users_test[qu]
                    all_u[all_traj_pos].append(curr_user)
                    query_dict_users_test[qu] = all_u

                    all_traj_id = query_dict_traj_ids_test[qu]
                    all_traj_id[all_traj_pos].append(curr_traj_id)
                    query_dict_traj_ids_test[qu] = all_traj_id

                    all_traj_time = query_dict_traj_time_test[qu]
                    all_traj_time[all_traj_pos].append(curr_traj_time)
                    query_dict_traj_time_test[qu] = all_traj_time

                    all_freq = query_dict_freq_test[qu]
                    all_freq[all_traj_pos] += 1
                    query_dict_freq_test[qu] = all_freq

            else:

                query_dict_trajectory_test.setdefault(qu, []).append(curr_traj)
                query_dict_users_test.setdefault(qu, []).append([curr_user])
                query_dict_traj_ids_test.setdefault(qu, []).append([curr_traj_id])
                query_dict_traj_time_test.setdefault(qu, []).append([curr_traj_time])
                query_dict_freq_test.setdefault(qu, []).append(1)

        line_no = (line_no + 1) % 3


def get_training_raw_dict():
    q_tr = dict()
    for k, v in query_dict_trajectory_train.items():
        GT_set = v

        GT_freq = query_dict_freq_train[k]

        v_new = []
        for i in range(len(v)):
            for j in range(GT_freq[i]):
                v_new.append(v[i])

        q_tr[k] = v_new

    q_u = dict()
    for k, v in query_dict_users_train.items():
        v_new = []
        for i in range(len(v)):
            for j in range(len(v[i])):
                v_new.append(v[i][j])
        q_u[k] = v_new

    training_data_dicts = q_tr, q_u

    return training_data_dicts


training_data_dicts = get_training_raw_dict()

POI_endpoints_dict_t = training_data_dicts[0]


def get_test_raw_dict():
    q_u = dict()
    for k, v in query_dict_users_test.items():
        v_new = []
        for i in range(len(v)):
            v_new.append(v[i][0])
        q_u[k] = v_new

    test_data_dicts = query_dict_trajectory_test, q_u

    return test_data_dicts


test_data_dicts = get_test_raw_dict()

ALL_POI_IDS_FREQ = dict()
for k, v in POI_endpoints_dict_t.items():
    for i in range(len(v)):
        for j in range(len(v[i])):
            ALL_POI_IDS_FREQ.setdefault(v[i][j], []).append(1)

for i in range(len(ALL_POI_IDS)):
    ALL_POI_IDS_FREQ.setdefault(ALL_POI_IDS[i], []).append(1)

for k, v in ALL_POI_IDS_FREQ.items():
    ALL_POI_IDS_FREQ[k] = len(v)

ALL_POI_IDS_FREQ_SORTED = sorted(ALL_POI_IDS_FREQ.items(), key=lambda item: item[1], reverse=True)
vocab_to_int = dict()
for i in range(len(ALL_POI_IDS_FREQ_SORTED)):
    vocab_to_int[ALL_POI_IDS_FREQ_SORTED[i][0]] = i
vocab_to_int['GO'] = len(ALL_POI_IDS_FREQ_SORTED)
vocab_to_int['PAD'] = len(ALL_POI_IDS_FREQ_SORTED) + 1
vocab_to_int['END'] = len(ALL_POI_IDS_FREQ_SORTED) + 2
int_to_vocab = dict()
for k, v in vocab_to_int.items():
    int_to_vocab[v] = k


def convert_vocab_to_int(traj_dict):
    traj_dict_new = dict()

    for k, v in traj_dict.items():
        st_p = v[0][0]
        en_p = v[0][-1]
        new_k = str(vocab_to_int[st_p]) + "-" + str(vocab_to_int[en_p])
        new_v = []
        for i in range(len(v)):
            new_v_i = [vocab_to_int[poi] for poi in v[i]]
            new_v.append(new_v_i)
        traj_dict_new[new_k] = new_v
    return traj_dict_new


training_data_dicts_vi = (convert_vocab_to_int(training_data_dicts[0]), training_data_dicts[1])
test_data_dicts_vi = (convert_vocab_to_int(test_data_dicts[0]), test_data_dicts[1])


def get_training_rawdata(batch_size=8):
    all_training_rawdata = []
    for k, v in training_data_dicts_vi[0].items():
        for i in range(len(v)):
            all_training_rawdata.append(v[i])

    np.random.shuffle(all_training_rawdata)

    train_size = len(all_training_rawdata) % batch_size
    all_training_rawdata = all_training_rawdata + [all_training_rawdata[-1]] * (batch_size - train_size)

    return [all_training_rawdata]


training_rawdata = get_training_rawdata()


def get_training_data_for_KDiverse_Raw():
    trainT = training_rawdata[0]
    # print(trainT)

    POI_endpoints_dict = {}

    for i in range(len(trainT)):
        for path_length in range(3, len(trainT[i]) + 1):
            for j in range(0, len(trainT[i]) - path_length + 1):
                path = trainT[i][j:j + path_length]
                POI_endpoints_dict.setdefault(str(path[0]) + '-' + str(path[-1]), []).append(path)

    def isSame(traj_a, traj_b):
        if (len(traj_a) != len(traj_b)):
            return False
        for i in range(len(traj_a)):
            if (traj_a[i] != traj_b[i]):
                return False

        return True

    # print(len(trainT))

    for k, v in POI_endpoints_dict.items():
        v_new = list(v)

        str_k = str(k).split("-")
        poi_start = int(str_k[0])
        poi_end = int(str_k[1])
        for i in range(len(trainT)):
            for j in range(1, len(trainT[i]) - 1):
                if (trainT[i][j] != poi_start):
                    continue
                for l in range(j + 1, len(trainT[i]) - 1):
                    if (trainT[i][l] == poi_end):
                        new_traj = trainT[i][j:l + 1]
                        if (len(new_traj) >= 3):
                            # print(str(poi_start) + "-" + str(poi_end)+" NEW")
                            # print(trainT[i])
                            # print(new_traj)
                            v_new.append(new_traj)

        POI_endpoints_dict[k] = v_new

    for k, v in POI_endpoints_dict.items():
        v_new = []
        for i in range(len(v)):
            gotBefore = False
            for j in range(i):
                if (isSame(v[i], v[j])):
                    gotBefore = True
                    break
            if (not gotBefore):
                v_new.append(v[i])

        POI_endpoints_dict[k] = v_new

    total_traj = 0
    for k, v in POI_endpoints_dict.items():
        total_traj += len(v)

    # print(total_traj)

    POI_endpoints_dict = {k: v for k, v in
                          sorted(POI_endpoints_dict.items(), key=lambda items: len(items[1]), reverse=True)}

    POI_endpoints_dict_lengths = {k: len(v) for k, v in
                                  sorted(POI_endpoints_dict.items(), key=lambda items: len(items[1]), reverse=True)}

    # print(POI_endpoints_dict_lengths)

    POI_endopints_dict_seq_lengths = {}

    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(POI_endpoints_dict)

    # print([ popularity_metric(traj,POI_PAIRS_FREQ) for traj in list(POI_endpoints_dict['15-7'])])

    for k, v in POI_endpoints_dict.items():
        v_new = []
        for i in range(len(v)):
            v_new.append(len(v[i]))

        POI_endopints_dict_seq_lengths.setdefault(k, v_new)

    # print(POI_endopints_dict_seq_lengths)

    return POI_endpoints_dict, POI_endopints_dict_seq_lengths


POI_endpoints_dict, POI_endpoints_dict_seq_lengths = get_training_data_for_KDiverse_Raw()

poi_line_dict = dict()

for i in range(len(POIs)):
    c_poi = vocab_to_int[int(POIs[i][0])]
    poi_line_dict[c_poi] = i

poi_categories_dict = dict()
for i in range(len(vocab_to_int) - 3):
    if (i in poi_line_dict.keys()):
        poi_categories_dict[i] = POIs[poi_line_dict[i]][1]

poi_poi_distance_matrix = np.zeros((len(vocab_to_int) - 3, len(vocab_to_int) - 3))
for i in range(len(vocab_to_int) - 3):
    poi_1_lon = float(POIs[poi_line_dict[i]][2])
    poi_1_lat = float(POIs[poi_line_dict[i]][3])
    for j in range(len(vocab_to_int) - 3):
        poi_2_lon = float(POIs[poi_line_dict[j]][2])
        poi_2_lat = float(POIs[poi_line_dict[j]][3])

        dist = calc_dist_vec(poi_1_lon, poi_1_lat, poi_2_lon, poi_2_lat)
        poi_poi_distance_matrix[i][j] = np.round(dist, 2)


#############################################################################################

def POI_transition_matrix(poi_endp_dict):
    transition_mat = np.zeros((len(vocab_to_int) - 3, len(vocab_to_int) - 3))

    for k, v in poi_endp_dict.items():

        for i in range(len(v)):
            for j in range(len(v[i]) - 1):
                first_poi = v[i][j]
                second_poi = v[i][j + 1]

                transition_mat[first_poi][second_poi] += 1

    return transition_mat


poi_poi_transition_matrix_train = POI_transition_matrix(training_data_dicts_vi[0])

expo_trans1 = np.log10(max(1, np.min(poi_poi_transition_matrix_train)))
expo_trans2 = np.log10(np.max(poi_poi_transition_matrix_train))

max_dist = np.max(poi_poi_distance_matrix)
poi_poi_distance_matrix_train_gae = np.exp((max_dist - poi_poi_distance_matrix))
n_min = np.min(poi_poi_distance_matrix_train_gae)
n_max = np.max(poi_poi_distance_matrix_train_gae)
if n_max - n_min != 0:
    poi_poi_distance_matrix_train_gae = (poi_poi_distance_matrix_train_gae - n_min) / (n_max - n_min)


def normalise_transmat(transmat_cnt):
    transmat = np.array(transmat_cnt.copy())
    t_min = np.min(transmat) - 1
    t_max = np.max(transmat)
    if (t_max - t_min != 0):
        transmat = (transmat - t_min) / (t_max - t_min)
    else:
        transmat = (transmat - (t_min - 1))
    return transmat


def gen_poi_poi_category_matrix():
    transmat = np.zeros((len(vocab_to_int) - 3, len(vocab_to_int) - 3))
    for i in range(len(transmat)):
        for j in range(len(transmat[i])):
            if poi_categories_dict[i] == poi_categories_dict[j]:
                transmat[i, j] = 1

    return transmat


poi_poi_categories_train_gae = gen_poi_poi_category_matrix()

########################################################################################################

class dataset_trajectory(object):
    def __init__(self, input_seqs, input_seq_lengths, backward=False):
        self.input_seqs = input_seqs
        if backward == True:
            self.input_seqs = [list(reversed(traj)) for traj in self.input_seqs]
        self.input_seq_lengths = input_seq_lengths

        self.ds_length = len(input_seqs)
        self.max_seq_length = np.max(input_seq_lengths)
        self.pad_id = 0

    def process_batch(self, inp_batch, inp_seq_batch):

        inp_seq_batch_final = [ln - 1 for ln in inp_seq_batch]
        max_length_batch = np.max(inp_seq_batch)
        inp_batch_final = [ls[:-1] + [self.pad_id] * (max_length_batch - len(ls)) for ls in inp_batch]
        tgt_batch_final = [ls[1:] + [self.pad_id] * (max_length_batch - len(ls)) for ls in inp_batch]

        zp = zip(inp_batch_final, inp_seq_batch_final, tgt_batch_final)
        zp_l = list(zp)
        zp_l = sorted(zp_l, key=lambda tuple: tuple[1], reverse=True)

        ib, sb, tb = zip(*zp_l)
        ib = list(ib)
        sb = list(sb)
        tb = list(tb)

        ib = np.array(ib)
        ib = np.transpose(ib)
        sb = np.array(sb)
        tb = np.array(tb)
        tb = np.transpose(tb)

        return ib, sb, tb

    def no_training_batches(self, batch_size):

        no_seqs = self.ds_length
        no_batches = int(np.ceil(no_seqs / batch_size))

        return no_batches

    def __call__(self, step, batch_size):

        no_seqs = self.ds_length
        no_batches = int(np.ceil(no_seqs / batch_size))

        step_no = step % no_batches

        if (step_no + 1) * batch_size > no_seqs:
            ts_pad = [self.input_seqs[itr % no_seqs] for itr in range((step_no + 1) * batch_size - no_seqs)]
            tsl_pad = [self.input_seq_lengths[itr % no_seqs] for itr in range((step_no + 1) * batch_size - no_seqs)]

            inp_batch = self.input_seqs[step_no * batch_size:] + ts_pad
            inp_seq_batch = self.input_seq_lengths[step_no * batch_size:] + tsl_pad

        else:
            inp_batch = self.input_seqs[step_no * batch_size:(step_no + 1) * batch_size]
            inp_seq_batch = self.input_seq_lengths[step_no * batch_size:(step_no + 1) * batch_size]

        return self.process_batch(inp_batch, inp_seq_batch)


def get_trajectory_dataset():
    all_traj_data_train = []
    all_traj_data_train_seq = []

    train_keys = list(training_data_dicts_vi[0].keys())
    random.shuffle(train_keys)

    for i in range(len(train_keys)):
        v = training_data_dicts_vi[0][train_keys[i]]
        for j in range(len(v)):
            all_traj_data_train.append(v[j])
            all_traj_data_train_seq.append(len(v[j]))

    return dataset_trajectory(all_traj_data_train, all_traj_data_train_seq), \
           dataset_trajectory(all_traj_data_train, all_traj_data_train_seq, backward=True)
