import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score
from multiprocessing import Pool, cpu_count


def get_events(y_test, outlier=1, normal=0, breaks=[]):
    events = dict()
    label_prev = normal
    event = 0  # corresponds to no event
    event_start = 0
    for tim, label in enumerate(y_test):
        if label == outlier:
            if label_prev == normal:
                event += 1
                event_start = tim
            elif tim in breaks:
                # A break point was hit, end current event and start new one
                event_end = tim - 1
                events[event] = (event_start, event_end)
                event += 1
                event_start = tim

        else:
            # event_by_time_true[tim] = 0
            if label_prev == outlier:
                event_end = tim - 1
                events[event] = (event_start, event_end)
        label_prev = label

    if label_prev == outlier:
        event_end = tim - 1
        events[event] = (event_start, event_end)
    return events


def get_composite_fscore(pred_labels, y_test, true_events, return_prec_rec=False):
    #true_events = get_events(y_test)
    tp = np.sum([pred_labels[start:end + 1].any() for start, end in true_events.values()])
    fn = len(true_events) - tp
    rec_e = tp/(tp + fn)
    prec_t = precision_score(y_test, pred_labels)
    fscore_c = 2 * rec_e * prec_t / (rec_e + prec_t)
    if prec_t == 0 and rec_e == 0:
        fscore_c = 0
    if return_prec_rec:
        return prec_t, rec_e, fscore_c
    return fscore_c


def get_pointadjusted_fscore(pred_labels, y_test, anomaly_events, return_prec_rec=False, adjust=True):
    #tp = np.sum([pred_labels[start:end + 1].any()*(end-start+1) for start, end in anomaly_events.values()])
    #fn = len(anomaly_events) - tp
    #fn = np.sum([pred_labels[start:end + 1].any()*(end-start+1) for start, end in anomaly_events.values() if pred_labels[start:end + 1].any() == 0])

    #fp = ((y_test == 0) & (pred_labels == 1)).sum()
    #tn = len(normal_events) - fp

    #rec_adjust = tp/(tp + fn)
    #prec_adjust = tp/(tp + fp)

    if adjust:
        adjust_pred_labels = adjust_predicts_(pred_labels, anomaly_events)
    else:
        adjust_pred_labels = pred_labels
    fscore_adjust = f1_score(y_test, adjust_pred_labels)
    prec_adjust = precision_score(y_test, adjust_pred_labels)
    rec_adjust = recall_score(y_test, adjust_pred_labels)

    #fscore_adjust = 2 * rec_adjust * prec_adjust / (rec_adjust + prec_adjust)
    if return_prec_rec:
        return prec_adjust, rec_adjust, fscore_adjust
    return fscore_adjust


def adjust_predicts_(predict, anomaly_events):

    for _, (start, end) in anomaly_events.items():
        if predict[start:end + 1].any():
            predict[start:end + 1] = True
    return predict

# def adjust_predicts_(predict, label, calc_latency=False):
#     actual = label > 0.1
#     anomaly_state = False
#     anomaly_count = 0
#     latency = 0
#
#     for i in range(len(predict)):
#         if any(actual[max(i, 0) : i + 1]) and predict[i] and not anomaly_state:
#             anomaly_state = True
#             anomaly_count += 1
#             for j in range(i, 0, -1):
#                 if not actual[j]:
#                     break
#                 else:
#                     if not predict[j]:
#                         predict[j] = True
#                         latency += 1
#         elif not actual[i]:
#             anomaly_state = False
#         if anomaly_state:
#             predict[i] = True
#     if calc_latency:
#         return predict, latency / (anomaly_count + 1e-4)
#     else:
#         return predict
