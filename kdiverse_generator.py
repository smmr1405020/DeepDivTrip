import metric
import numpy as np
import data_generator
import graph_embedding_kdiv
import lstm_model
import gibbs_sampler
import time

np.random.seed(1234567890)


def generate_result(load_from_file, K):
    graph_embedding_kdiv.get_POI_embeddings(load_from_file=load_from_file)
    lstm_model.get_forward_lstm_model(load_from_file=load_from_file)
    lstm_model.get_backward_lstm_model(load_from_file=load_from_file)

    total_score_curr_f1 = 0
    total_score_curr_pf1 = 0
    total_score_curr_edt = 0

    all_iou_scores = []

    total_traj_curr = 0
    count = 1

    # st = time.time()
    for k, v in data_generator.test_data_dicts_vi[0].items():

        str_k = str(k).split("-")
        poi_start = int(str_k[0])
        poi_end = int(str_k[1])

        _, lstm_order = gibbs_sampler.get_prob_in_idx([poi_start, 0, poi_end], 1)
        lstm_rank = np.argsort(lstm_order)

        def get_next_poi(use_freq, rank):
            proposed_poi = 0
            for i in range(len(rank)):
                if i == poi_start or i == poi_end:
                    continue
                if (proposed_poi == poi_start or proposed_poi == poi_end) and (i != poi_start and i != poi_end):
                    proposed_poi = i
                    continue

                if use_freq[i] < use_freq[proposed_poi]:
                    proposed_poi = i
                elif use_freq[i] == use_freq[proposed_poi] and rank[i] < rank[proposed_poi]:
                    proposed_poi = i

            use_freq[proposed_poi] += 1
            return proposed_poi

        use_freq = np.zeros([len(lstm_rank)])
        all_traj = []
        for i in range(K):
            next_poi = get_next_poi(use_freq, lstm_rank)
            new_traj = gibbs_sampler.gibbs_sampling([poi_start, next_poi, poi_end])
            for i in range(len(new_traj)):
                use_freq[new_traj[i]] += 1
            all_traj.append(new_traj)

        print("{}/{}".format(count, len(data_generator.test_data_dicts_vi[0])))
        count += 1
        print([poi_start, poi_end])
        print(all_traj)
        total_score_curr_f1 += metric.tot_f1_evaluation(v, data_generator.test_data_dicts_vi[2][k], all_traj)
        total_score_curr_pf1 += metric.tot_pf1_evaluation(v, data_generator.test_data_dicts_vi[2][k], all_traj)
        total_traj_curr += np.sum(data_generator.test_data_dicts_vi[2][k]) * len(all_traj)

        all_iou_scores.append(metric.coverage_iou(v, all_traj))

        avg_f1 = total_score_curr_f1 / total_traj_curr
        avg_pf1 = total_score_curr_pf1 / total_traj_curr
        avg_iou = np.average(np.array(all_iou_scores))

        print("Avg. upto now: F1: " + str(avg_f1) + " PF1: " + str(avg_pf1) + " IOU: " + str(avg_iou))

    print("\n")
    print("Final Score - With K = {}".format(K))
    avg_f1 = total_score_curr_f1 / total_traj_curr
    avg_pf1 = total_score_curr_pf1 / total_traj_curr
    avg_iou = np.average(np.array(all_iou_scores))

    print("F1: " + str(avg_f1) + " PF1: " + str(avg_pf1) + " IOU: " + str(avg_iou))

    # en = time.time()
    # print(en-st)
    # print((en-st)/count)
    return
