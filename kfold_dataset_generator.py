import time
import numpy as np
import random
import csv
import os
import glob
import args_kdiverse

random.seed(1234567890)
np.random.seed(1234567890)

dat_suffix = ['Osak', 'Glas', 'Edin', 'Toro', 'Melb','caliAdv', 'disHolly', 'disland', 'epcot', 'MagicK']
dat_ix = args_kdiverse.dataset_index
embedding_name = dat_suffix[dat_ix]

poi_name = "poi-" + dat_suffix[dat_ix] + ".csv"  # Edin
tra_name = "traj-" + dat_suffix[dat_ix] + ".csv"

print('To Train', dat_suffix[dat_ix])
# =============================== data load ====================================== #
op_tdata = open('origin_data/' + poi_name, 'r')
ot_tdata = open('origin_data/' + tra_name, 'r')
Trajectory = []

for line in ot_tdata.readlines():
    lineArr = line.split(',')
    temp_line = list()
    if lineArr[0] == 'userID':
        continue
    for i in range(len(lineArr)):
        temp_line.append(lineArr[i].strip('\n'))
    Trajectory.append(temp_line)

TRAIN_TRA = []
TRAIN_USER = []
TRAIN_TIME = []
TRAIN_DIST = []
DATA = {}  # temp_data

for index in range(len(Trajectory)):
    if (int(Trajectory[index][-2]) >= 3):  # the length of the trajectory must over than 3
        DATA.setdefault(Trajectory[index][0] + '-' + Trajectory[index][1], []).append(
            [Trajectory[index][2], Trajectory[index][3], Trajectory[index][4]])  # userID+trajID

for k, v in DATA.items():
    v_new = sorted(v, key=lambda item: item[1])
    DATA[k] = v_new

    for i in range(len(v_new)):
        v_new[i][0] = int(v_new[i][0])
        v_new[i][1] = int(time.strftime("%H:%M:%S", time.localtime(int(v_new[i][1]))).split(":")[0])
        v_new[i][2] = int(time.strftime("%H:%M:%S", time.localtime(int(v_new[i][2]))).split(":")[0])

ALL_TRAJ = []
ALL_TRAJID = []
ALL_USER = []
ALL_TIME = []

for k, v in DATA.items():
    traj = []
    traj_time = []

    for i in range(len(v)):
        traj.append(v[i][0])
        traj_time.append(v[i][1])

    ALL_TRAJ.append(traj)
    ALL_TIME.append(traj_time)

    str_k = k.split("-")

    ALL_TRAJID.append(int(str_k[1]))
    ALL_USER.append(str_k[0])

query_set_dict_traj = {}
query_set_dict_user = {}
query_set_dict_time = {}
query_set_dict_tid = {}

for i in range(len(ALL_TRAJ)):
    q_str = str(ALL_TRAJ[i][0]) + "-" + str(ALL_TRAJ[i][-1])
    query_set_dict_traj.setdefault(q_str, []).append(ALL_TRAJ[i])
    query_set_dict_user.setdefault(q_str, []).append(ALL_USER[i])
    query_set_dict_time.setdefault(q_str, []).append(ALL_TIME[i])
    query_set_dict_tid.setdefault(q_str, []).append(ALL_TRAJID[i])

qkeys = list(query_set_dict_traj.keys())
np.random.shuffle(qkeys)


def generate_ds_kfold_parts(KFOLD):
    files = glob.glob('processed_data/*')
    for f in files:
        os.remove(f)

    keys_per_fold = len(qkeys) // KFOLD

    kfold_qkeys = {}
    for i in range(KFOLD - 1):
        kfold_qkeys[i + 1] = qkeys[i * keys_per_fold: (i + 1) * keys_per_fold]
    kfold_qkeys[KFOLD] = qkeys[(KFOLD - 1) * keys_per_fold:]

    for k, v in kfold_qkeys.items():
        to_traj_csv_train = []

        for i in range(len(v)):
            trajectories = query_set_dict_traj[v[i]]
            users = query_set_dict_user[v[i]]
            traj_ids = query_set_dict_tid[v[i]]
            traj_times = query_set_dict_time[v[i]]

            for j in range(len(trajectories)):
                to_traj_csv_train.append(trajectories[j])
                to_traj_csv_train.append(traj_times[j])
                to_traj_csv_train.append([users[j], traj_ids[j]])

        with open("processed_data/" + embedding_name + '_set_part_' + str(k) + '.csv', mode='w',
                  newline="") as csv_file:
            csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in to_traj_csv_train:
                csv_file_writer.writerow(row)


def generate_train_test_data(test_index, KFOLD):
    train_lines = []
    for i in range(1, KFOLD + 1):
        if i == test_index:
            continue
        with open("processed_data/" + embedding_name + '_set_part_' + str(i) + '.csv', mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                train_lines.append(row)

    with open("processed_data/" + embedding_name + '_train_set.csv', mode='w', newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in train_lines:
            csv_file_writer.writerow(row)

    test_lines = []
    with open("processed_data/" + embedding_name + '_set_part_' + str(test_index) + '.csv', mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            test_lines.append(row)

    with open("processed_data/" + embedding_name + '_test_set.csv', mode='w', newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in test_lines:
            csv_file_writer.writerow(row)
