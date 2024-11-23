import numpy as np
import more_itertools as mit
from task_utils.evaluation.spot import SPOT
from scipy.stats import rankdata
from sklearn.metrics import f1_score, confusion_matrix


def find_epsilon(errors, reg_level=1):
    """
    Threshold method proposed by Hundman et. al. (https://arxiv.org/abs/1802.04431)
    Code from TelemAnom (https://github.com/khundman/telemanom)
    """
    e_s = errors
    best_epsilon = None
    max_score = -10000000
    mean_e_s = np.mean(e_s)
    sd_e_s = np.std(e_s)

    for z in np.arange(2.5, 12, 0.5):
        epsilon = mean_e_s + sd_e_s * z
        pruned_e_s = e_s[e_s < epsilon]

        i_anom = np.argwhere(e_s >= epsilon).reshape(-1,)
        buffer = np.arange(1, 50)
        i_anom = np.sort(
            np.concatenate(
                (
                    i_anom,
                    np.array([i + buffer for i in i_anom]).flatten(),
                    np.array([i - buffer for i in i_anom]).flatten(),
                )
            )
        )
        i_anom = i_anom[(i_anom < len(e_s)) & (i_anom >= 0)]
        i_anom = np.sort(np.unique(i_anom))

        if len(i_anom) > 0:
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            # E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

            mean_perc_decrease = (mean_e_s - np.mean(pruned_e_s)) / mean_e_s
            sd_perc_decrease = (sd_e_s - np.std(pruned_e_s)) / sd_e_s
            if reg_level == 0:
                denom = 1
            elif reg_level == 1:
                denom = len(i_anom)
            elif reg_level == 2:
                denom = len(i_anom) ** 2

            score = (mean_perc_decrease + sd_perc_decrease) / denom

            if score >= max_score and len(i_anom) < (len(e_s) * 0.5):
                max_score = score
                best_epsilon = epsilon

    if best_epsilon is None:
        best_epsilon = np.max(e_s)
    return best_epsilon


def epsilon_eval(train_scores, test_scores, test_labels, reg_level=1):
    best_epsilon = find_epsilon(train_scores, reg_level)
    pred, p_latency = adjust_predicts(test_scores, test_labels, best_epsilon, calc_latency=True)
    if test_labels is not None:
        p_t = calc_point2point(pred, test_labels)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": best_epsilon,
            "latency": p_latency,
            "reg_level": reg_level,
        }
    else:
        return {"threshold": best_epsilon, "reg_level": reg_level}


def adjust_predicts(score, label, threshold, pred=None, calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                    A point is labeled as "anomaly" if its score is lower than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
            calc_latency (bool):
    Returns:
            np.ndarray: predict labels

    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    if label is None:
        predict = score > threshold
        return predict, None

    if pred is None:
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        predict = score > threshold
    else:
        predict = pred

    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def adjust_predicts_(predict, label, calc_latency=False):
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    latency = 0

    for i in range(len(predict)):
        if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_point2point(predict, actual, calc_confusion_matrix=False):
    """
    calculate f1 score by predict and actual.
    Args:
            predict (np.ndarray): the predict label
            actual (np.ndarray): np.ndarray
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = f1_score(predict, actual)#2 * precision * recall / (precision + recall + 0.00001)
    #print(confusion_matrix())
    if calc_confusion_matrix:
        print(confusion_matrix(predict, actual))
    return f1, precision, recall, TP, TN, FP, FN


def eval_scores_multi(scores_list, labels_list, th_steps, return_thresold=False, adjust_predict=True):
    fmeas = [None] * th_steps
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    thresholds = [None] * th_steps

    for i in range(th_steps):
        pred_label = []
        true_label = []
        for scores, labels in zip(scores_list, labels_list):
            scores_sorted = rankdata(scores, method='ordinal')
            # th_steps = 500
            preds = scores_sorted > th_vals[i] * len(scores)

            if adjust_predict:
                preds = adjust_predicts_(preds, labels, calc_latency=False)

            pred_label.append(preds)
            true_label.append(labels)
            score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
            thresholds[i] = scores[score_index]
            #thresholds[i] = th_vals[i]
        fmeas[i] = calc_point2point(np.concatenate(pred_label), np.concatenate(true_label))

    if return_thresold:
        return fmeas, thresholds
    else:
        return fmeas


def eval_scores(scores, labels, th_steps=400, metric_fn=f1_score, return_threshold=False, adjust_predict=True):

    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    meas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        preds = scores_sorted > th_vals[i] * len(scores)

        if adjust_predict:
            preds = adjust_predicts_(preds, labels, calc_latency=False)
        meas[i] = metric_fn(labels, preds)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]

    if return_threshold:
        return meas, thresholds
    return meas

# # calculate F1 scores
# def eval_scores(scores, labels, th_steps, return_thresold=False, adjust_predict=True):
#     # padding_list = [0]*(len(true_scores) - len(scores))
#     # # print(padding_list)
#     #
#     # if len(padding_list) > 0:
#     #     scores = np.concatenate([padding_list, scores])
#
#     scores_sorted = rankdata(scores, method='ordinal')
#     th_steps = th_steps
#     # th_steps = 500
#     th_vals = np.array(range(th_steps)) * 1.0 / th_steps
#     fmeas = [None] * th_steps
#     thresholds = [None] * th_steps
#     for i in range(th_steps):
#         preds = scores_sorted > th_vals[i] * len(scores)
#
#         if adjust_predict:
#             preds = adjust_predicts_(preds, labels, calc_latency=False)#f1_score(true_scores, cur_pred)
#         fmeas[i] = f1_score(labels, preds)
#
#         score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
#         thresholds[i] = scores[score_index]
#
#     if return_thresold:
#         return fmeas, thresholds
#     return fmeas


def bf_rank_search_multi(scores, gt_labels, adjust_predict=True):

        total_topk_err_scores = scores
        final_topk_fmeas ,thresolds = eval_scores_multi(total_topk_err_scores, gt_labels, 400, return_thresold=True, adjust_predict=adjust_predict)

        th_i = final_topk_fmeas.index(max(final_topk_fmeas))
        threshold = thresolds[th_i]

        pred_labels = np.zeros(len(total_topk_err_scores))
        pred_labels[total_topk_err_scores > threshold] = 1

        for i in range(len(pred_labels)):
            pred_labels[i] = int(pred_labels[i])
            gt_labels[i] = int(gt_labels[i])

        if adjust_predict:
            pred_labels, latency = adjust_predicts_(pred_labels, gt_labels)

        return pred_labels, latency


def bf_rank_search(score, gt_labels, adjust_predict=True):

    total_topk_err_scores = score
    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True, adjust_predict=adjust_predict)

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))
    threshold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > threshold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    if adjust_predict:
        pred_labels, latency = adjust_predicts_(pred_labels, gt_labels, calc_latency=True)
    else:
        latency = 0

    p_t = calc_point2point(pred_labels, gt_labels)
    #latency = 0
    return {
        "f1": p_t[0],#max(final_topk_fmeas),#p_t[0],
        "precision": p_t[1],
        "recall": p_t[2],
        "TP": p_t[3],
        "TN": p_t[4],
        "FP": p_t[5],
        "FN": p_t[6],
        "threshold": threshold,
        "latency": latency,
    }


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True, calc_confusion_table=False):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Finding best f1-score by searching for threshold..")
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1.0, -1.0, -1.0)
    m_t = 0.0
    m_l = 0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, adjust_predict=False)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            #m_l = latency
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)

    return {
        "f1": m[0],
        "precision": m[1],
        "recall": m[2],
        "TP": m[3],
        "TN": m[4],
        "FP": m[5],
        "FN": m[6],
        "threshold": m_t,
        #"latency": m_l,
    }


def pot_eval(init_score, score, label, q=1e-3, level=0.99, dynamic=False):
    """
    Run POT method on given score.
    :param init_score (np.ndarray): The data to get init threshold.
                    For `OmniAnomaly`, it should be the anomaly score of train set.
    :param: score (np.ndarray): The data to run POT method.
                    For `OmniAnomaly`, it should be the anomaly score of test set.
    :param label (np.ndarray): boolean list of true anomalies in score
    :param q (float): Detection level (risk)
    :param level (float): Probability associated with the initial threshold t
    :return dict: pot result dict
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """

    print(f"Running POT with q={q}, level={level}..")
    s = SPOT(q)  # SPOT object
    s.fit(init_score, score)
    s.initialize(level=level, min_extrema=False)  # Calibration step
    ret = s.run(dynamic=dynamic, with_alarm=False)

    print(len(ret["alarms"]))
    print(len(ret["thresholds"]))

    pot_th = np.mean(ret["thresholds"])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    if label is not None:
        p_t = calc_point2point(pred, label)
        return {
            "f1": p_t[0],
            "precision": p_t[1],
            "recall": p_t[2],
            "TP": p_t[3],
            "TN": p_t[4],
            "FP": p_t[5],
            "FN": p_t[6],
            "threshold": pot_th,
            "latency": p_latency,
        }
    else:
        return {
            "threshold": pot_th,
        }


def calc_seq(score, label, threshold, calc_confusion_matrix=False, adjust_predict=True):
    if adjust_predict:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=True)
    else:
        predict = score > threshold
    if calc_confusion_matrix:
        result = calc_point2point(predict, label, calc_confusion_matrix=calc_confusion_matrix)
    else:
        result = calc_point2point(predict, label)

    if adjust_predict:
        return result, latency
    else:
        return result



