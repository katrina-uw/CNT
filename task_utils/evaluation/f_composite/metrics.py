import numpy as np
from task_utils.evaluation.f_composite.utils import get_events, get_composite_fscore, get_pointadjusted_fscore
from scipy.stats import rankdata
from multiprocessing import Pool, cpu_count
from sklearn.metrics import f1_score


def get_best_composite_fscore(scores, labels, th_steps=400, return_threshold=False):
    true_events = get_events(labels)
    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    meas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        preds = scores_sorted > th_vals[i] * len(scores)

        meas[i] = get_composite_fscore(preds, labels, true_events, return_prec_rec=False)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]

    th_i = meas.index(max(meas))
    threshold = thresholds[th_i]
    predict = (scores > threshold).astype(int)
    precision, recall, f1 = get_composite_fscore(predict, labels, true_events, return_prec_rec=True)

    if return_threshold:
        return f1, precision, recall, threshold
    return f1, precision, recall


def get_best_adjusted_fscore(scores, labels, th_steps=400, return_threshold=False):
    anomaly_events = get_events(labels)
    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    meas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        preds = scores_sorted > th_vals[i] * len(scores)

        meas[i] = get_pointadjusted_fscore(preds, labels, anomaly_events, return_prec_rec=False)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores)+1))
        thresholds[i] = scores[score_index]

    th_i = meas.index(max(meas))
    threshold = thresholds[th_i]
    predict = (scores > threshold).astype(int)
    precision, recall, f1 = get_pointadjusted_fscore(predict, labels, anomaly_events, return_prec_rec=True)

    if return_threshold:
        return f1, precision, recall, threshold
    return f1, precision, recall


def process_threshold_f1_adjust(args):
    scores_sorted, scores, labels, anomaly_events, th_val = args

    preds = scores_sorted > th_val * len(scores)
    meas_val = get_pointadjusted_fscore(preds, labels, anomaly_events, return_prec_rec=False, adjust=True)
    score_index = scores_sorted.tolist().index(int(th_val * len(scores) + 1))
    threshold_val = scores[score_index]

    return meas_val, threshold_val
    # import logging
    # try:
    #     scores_sorted, scores, labels, anomaly_events, th_val = args
    #     val = int(th_val * len(scores) + 1)
    #
    #     #logging.debug(f"Searching for value {val} in list {scores_sorted.tolist()}")
    #
    #     score_index = scores_sorted.tolist().index(val)
    #     threshold_val = scores[score_index]
    #
    #     preds = scores_sorted > th_val * len(scores)
    #     meas_val = get_pointadjusted_fscore(preds, labels, anomaly_events, return_prec_rec=False, adjust=True)
    #
    #     return meas_val, threshold_val
    # except ValueError as e:
    #     logging.error(f"ValueError occurred: {e}")
    #     logging.error(f"Value {val} not found in the list {scores_sorted.tolist()}")
    #     # handle error or return a default value
    # except Exception as e:
    #     logging.error(f"An unexpected error occurred: {e}")


def process_threshold_f1(args):
    scores_sorted, scores, labels, anomaly_events, th_val = args

    preds = scores_sorted > th_val * len(scores)
    meas_val = f1_score(labels, preds)
    score_index = scores_sorted.tolist().index(int(th_val * len(scores) + 1))
    threshold_val = scores[score_index]

    return meas_val, threshold_val


def process_threshold_f1_composite(args):
    scores_sorted, scores, labels, anomaly_events, th_val = args

    preds = scores_sorted > th_val * len(scores)
    meas_val = get_composite_fscore(preds, labels, anomaly_events, return_prec_rec=False)
    score_index = scores_sorted.tolist().index(int(th_val * len(scores) + 1))
    threshold_val = scores[score_index]

    return meas_val, threshold_val


def get_percentile_fscore(test_scores, train_scores, labels, return_threshold=True, eval_fn_type="f1_adjust", percentile=99):

    threshold = np.percentile(train_scores, percentile)

    anomaly_events = get_events(labels)
    #scores_sorted = rankdata(test_scores, method='ordinal')
    predict = (test_scores > threshold).astype(int)

    if eval_fn_type == "f1":
        precision, recall, f1 = get_pointadjusted_fscore(predict, labels, anomaly_events, return_prec_rec=True, adjust=False)
    elif eval_fn_type == "f1_adjust":
        precision, recall, f1 = get_pointadjusted_fscore(predict, labels, anomaly_events, return_prec_rec=True)
    elif eval_fn_type == "f1_composite":
        precision, recall, f1 = get_composite_fscore(predict, labels, anomaly_events, return_prec_rec=True)

    if return_threshold:
        return f1, precision, recall, threshold
    return f1, precision, recall


def get_zscore_fscore(test_scores, train_scores, labels, return_threshold=True, eval_fn_type="f1_adjust", threshold_factor=3):

    mean = np.mean(train_scores)
    std = np.std(train_scores)
    threshold = mean + threshold_factor * std

    anomaly_events = get_events(labels)
    #scores_sorted = rankdata(test_scores, method='ordinal')
    predict = (test_scores > threshold).astype(int)

    if eval_fn_type == "f1":
        precision, recall, f1 = get_pointadjusted_fscore(predict, labels, anomaly_events, return_prec_rec=True, adjust=False)
    elif eval_fn_type == "f1_adjust":
        precision, recall, f1 = get_pointadjusted_fscore(predict, labels, anomaly_events, return_prec_rec=True)
    elif eval_fn_type == "f1_composite":
        precision, recall, f1 = get_composite_fscore(predict, labels, anomaly_events, return_prec_rec=True)

    if return_threshold:
        return f1, precision, recall, threshold
    return f1, precision, recall


def get_best_fscore_faster(scores, labels, th_steps=400, return_threshold=False, eval_fn_type = "f1"):
    anomaly_events = get_events(labels)
    scores_sorted = rankdata(scores, method='ordinal')
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps

    #print(scores_sorted)
    # Prepare arguments for multiprocessing
    args = [(scores_sorted, scores, labels, anomaly_events, th_val) for th_val in th_vals]

    if eval_fn_type == "f1":
        process_threshold = process_threshold_f1
    elif eval_fn_type == "f1_adjust":
        process_threshold = process_threshold_f1_adjust
    elif eval_fn_type == "f1_composite":
        process_threshold = process_threshold_f1_composite
    # Use multiprocessing to process the thresholds in parallel
    with Pool(cpu_count()) as pool:
        results = pool.map(process_threshold, args)

    # Unpack the results
    meas, thresholds = zip(*results)

    # Find the best threshold
    th_i = meas.index(max(meas))
    threshold = thresholds[th_i]
    predict = (scores > threshold).astype(int)
    if eval_fn_type == "f1":
        precision, recall, f1 = get_pointadjusted_fscore(predict, labels, anomaly_events, return_prec_rec=True, adjust=False)
    elif eval_fn_type == "f1_adjust":
        precision, recall, f1 = get_pointadjusted_fscore(predict, labels, anomaly_events, return_prec_rec=True)
    elif eval_fn_type == "f1_composite":
        precision, recall, f1 = get_composite_fscore(predict, labels, anomaly_events, return_prec_rec=True)

    if return_threshold:
        return f1, precision, recall, threshold
    return f1, precision, recall