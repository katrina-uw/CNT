import os.path

from task_utils.evaluation.evaluator_utils import adjust_predicts, adjust_predicts_
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from data_utils.dataset import get_events
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from task_utils.evaluation.vus.metrics import get_range_vus_roc
from task_utils.evaluation.f_composite.metrics import get_best_fscore_faster, get_percentile_fscore, get_zscore_fscore
from task_utils.evaluation.evaluator_utils import pot_eval
import pickle
from numpy.lib.stride_tricks import sliding_window_view


class Evaluator(object):

    def __init__(self, ds_object, batch_size, window_size=50, reg_level=0, \
    level=0.9, q=0.005, dynamic_pot=False, scale_scores=False, MCDO=False, use_mov_av=False, anomaly_ratio=None, anomaly_percentile=None, anomaly_zscore_factor=None):
        self.use_mov_av = use_mov_av
        self.batch_size = batch_size
        self.window_size = window_size
        self.scale_scores = scale_scores
        self.reg_level = reg_level
        self.level = level
        self.dynamic_pot = dynamic_pot
        self.q = q
        self.MCDO = MCDO
        self.dataset = ds_object.name
        self.ds_object = ds_object
        self.anomaly_ratio = anomaly_ratio
        self.anomaly_percentile = anomaly_percentile
        self.anomaly_zscore_factor = anomaly_zscore_factor

    def evaluate_only(self, test_prediction, labels, metric_types=["f1_pa", "f1", "f1_cp", "vus_pr", "vus_roc"]):

        test_pred_df = self.convert_to_df(test_prediction, ref_prediction_dic=test_prediction)
        test_anomaly_scores = test_pred_df["score_global"].values

        test_anomaly_scores = self.adjust_anomaly_scores(test_anomaly_scores, self.dataset, False)

        test_pred_df['score_global'] = test_anomaly_scores

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        # if bf_adjust == True:
        #     bf_eval = bf_rank_search(test_anomaly_scores, labels, adjust_predict=True)#
        #     print(f"Results using best f1 score search:\n {bf_eval}")
        # else:
        #     bf_eval = None
        #
        # bf_eval_pointwise = bf_rank_search(test_anomaly_scores, labels, adjust_predict=False)#
        # print(f"Results using best pointwise f1 score search:\n {bf_eval_pointwise}")
        metric_results = self.compute_metrics(test_anomaly_scores, labels, metric_types=metric_types)
        print(metric_results)

        return metric_results

    def evaluate(self, test_prediction, train_prediction, labels, metric_types=["f1_pa", "f1", "f1_cp", "vus_pr", "vus_roc"], save_path=None):

        train_pred_df = self.convert_to_df(train_prediction, ref_prediction_dic=train_prediction)

        test_pred_df = self.convert_to_df(test_prediction, ref_prediction_dic=train_prediction)

        train_anomaly_scores = train_pred_df["score_global"].values
        test_anomaly_scores = test_pred_df["score_global"].values

        train_anomaly_scores = self.adjust_anomaly_scores(train_anomaly_scores, self.dataset, True)
        test_anomaly_scores = self.adjust_anomaly_scores(test_anomaly_scores, self.dataset, False)

        train_pred_df['score_global'] = train_anomaly_scores
        test_pred_df['score_global'] = test_anomaly_scores

        if self.use_mov_av:
            smoothing_window = 3
            #train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        # # p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, labels, q=self.q, level=self.level, dynamic=self.dynamic_pot)
        # # print(f"Results using peak-over-threshold method:\n {p_eval}")
        # bf_eval = bf_rank_search(test_anomaly_scores, labels, adjust_predict=True)#
        # print(f"Results using best f1 score search:\n {bf_eval}")
        #
        # bf_eval_pointwise = bf_rank_search(test_anomaly_scores, labels, adjust_predict=False)#
        # print(f"Results using best pointwise f1 score search:\n {bf_eval_pointwise}")
        #
        # for k, v in bf_eval.items():
        #     bf_eval[k] = float(v)
        #
        # # if self.eval_method == "pot":
        # #     eval = p_eval
        # if self.eval_method == "bf_search":
        #     eval = bf_eval
        # # elif self.eval_method == "epsilon":
        # #     eval = e_eval
        # if print_identified_events:
        #     test_preds_global = test_anomaly_scores > bf_eval["threshold"]
        #     test_preds_global = adjust_predicts(None, labels, bf_eval["threshold"], pred=test_preds_global)
        #     unidentified_events = get_unidentified_events(labels, test_preds_global)
        #     print("Unidentified events:", unidentified_events)

        metric_results = self.compute_metrics(test_anomaly_scores, labels, train_scores=train_anomaly_scores, metric_types=metric_types)
        print(metric_results)

        if save_path:
            # Save
            # summary = {"bf_result": metric_results}
            # with open(f"{save_path}/summary.txt", "w") as f:
            #     json.dump(summary, f, indent=2)
            self.save_output(train_pred_df, test_pred_df, labels, metric_results["pa_threshold"], save_path)
        return metric_results

    def save_output(self, train_pred_df, test_pred_df, labels, threshold, save_path):
        global_epsilon = threshold
        test_pred_df["true_label_global"] = labels
        #if train_pred_df:
        train_pred_df["thresh_global"] = global_epsilon
        train_pred_df[f"pred_label_global"] = (train_pred_df["score_global"].values >= global_epsilon).astype(int)

        test_pred_df["thresh_global"] = global_epsilon
        test_preds_global = (test_pred_df["score_global"].values >= global_epsilon).astype(int)
        # Adjust predictions according to evaluation strategy
        if labels is not None:
            test_preds_global = adjust_predicts(None, labels, global_epsilon, pred=test_preds_global)
        test_pred_df[f"pred_label_global"] = test_preds_global

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        print(f"Saving output to {save_path}/<train/test>_output.pkl")
        #if train_pred_df:
        train_pred_df.to_pickle(f"{save_path}/train_output.pkl")
        test_pred_df.to_pickle(f"{save_path}/test_output.pkl")

        print("-- Done.")

    def assign_causes(self, y_test, num_features):
        causes = self.ds_object.get_root_causes()
        if self.dataset.upper() in ["SWAT", "WADI", "DAMADICS"]:
            events = get_events(y_test)

            causes_array = np.zeros(shape=(len(y_test), num_features))
            for index, cause in enumerate(causes):
                if len(cause) > 0:
                    for c in cause:
                        causes_array[events[index+1][0]:events[index+1][1]+1, c]=1

            X = {}
            for i in range(num_features):
                X[f"cause_{i}"] = causes_array[:, i]

            cause_X = pd.DataFrame(X)
            return cause_X
        else:
            return pd.DataFrame({})

    def convert_to_df(self, prediction_dic, ref_prediction_dic):
        df = pd.DataFrame()

        for key, value in prediction_dic.items():
            if prediction_dic[key] is None:
                continue
            names = key.split("_")
            if names[1] == "tc":
                if names[0] == "score":
                    anomaly_scores = np.zeros_like(prediction_dic[key])
                    for i in range(prediction_dic[key].shape[1]):
                        a_score = prediction_dic[key][:, i]
                        if self.scale_scores:
                            ref_a_score = ref_prediction_dic[key][:, i]
                            q75, q25 = np.percentile(ref_a_score, [75, 25])
                            iqr = q75 - q25
                            median = np.median(ref_a_score)
                            epsilon = 1e-5
                            a_score = (a_score - median) / (epsilon + iqr)
                            #a_score = self.batch_normalize_errors_with_fixed_window_iqr(ref_a_score, a_score, 10000, 200)
                            # from sklearn.preprocessing import StandardScaler
                            #a_score = StandardScaler().fit(ref_a_score).transform(a_score)

                        if self.use_mov_av:
                            smoothing_window = 3  # int(self.batch_size * self.window_size * 0.05)
                            a_score = pd.DataFrame(a_score).ewm(
                                span=smoothing_window).mean().values.flatten()

                        anomaly_scores[:, i] = a_score

                        df[f"score_{i}"] = a_score

                else:
                    for i in range(prediction_dic[key].shape[1]):
                        df[f"{names[0]}_{i}"] = prediction_dic[key][:, i].tolist()

        if "score_t" not in prediction_dic or prediction_dic["score_t"] is None:
            anomaly_scores = np.mean(anomaly_scores, 1)
            df['score_global'] = anomaly_scores
        else:
            df['score_global'] = prediction_dic["score_t"]
        return df

    def normalize_errors_with_fixed_window(self, train_errors, test_errors, window_size, normalize_type):
        # Concatenate training and testing errors
        combined_errors = np.concatenate((train_errors, test_errors))

        # Create a sliding window view of the combined errors
        sliding_windows = sliding_window_view(combined_errors, window_size)

        # Compute the mean and standard deviation for each sliding window
        if normalize_type == "standard":
            means = np.mean(sliding_windows, axis=-1)
            stds = np.std(sliding_windows, axis=-1)
            test_indices = np.arange(len(train_errors), len(combined_errors))
            normalized_test_errors = (test_errors - means[test_indices - window_size + 1]) / stds[
                test_indices - window_size + 1]

            stds_non_zero = stds[test_indices - window_size + 1] != 0
            normalized_test_errors[~stds_non_zero] = test_errors[~stds_non_zero] - \
                                                     means[test_indices - window_size + 1][
                                                         ~stds_non_zero]
        elif normalize_type == "iqr":
            q1 = np.percentile(sliding_windows, 25, axis=-1)
            q3 = np.percentile(sliding_windows, 75, axis=-1)
            iqr = q3-q1
            median = np.median(sliding_windows, axis=-1)
            test_indices = np.arange(len(train_errors), len(combined_errors))
            normalized_test_errors = (test_errors - median[test_indices - window_size + 1]) / iqr[
                test_indices - window_size + 1]

            IQR_non_zero = iqr[test_indices - window_size + 1] != 0
            normalized_test_errors[~IQR_non_zero] = test_errors[~IQR_non_zero] - median[test_indices - window_size + 1][
                ~IQR_non_zero]

        return normalized_test_errors

    def batch_normalize_errors_with_fixed_window_iqr(self, train_errors, test_errors, window_size, batch_size):
        # Concatenate training and testing errors
        combined_errors = np.concatenate((train_errors, test_errors))

        # Initialize the array to store normalized test errors
        normalized_test_errors = np.zeros_like(test_errors)

        # Calculate the number of batches
        num_batches = int(np.ceil(len(test_errors) / batch_size))

        for batch in range(num_batches):
            # Determine the start and end indices of the batch
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, len(test_errors))

            # Determine the start and end indices of the context window
            context_end_idx = len(train_errors) + start_idx
            context_start_idx = max(0, context_end_idx - window_size)

            # Define the context window using the fixed window size
            context = combined_errors[context_start_idx:context_end_idx]

            # Compute the 25th and 75th percentiles (Q1 and Q3) for the context window
            Q1 = np.percentile(context, 25)
            Q3 = np.percentile(context, 75)

            # Compute the IQR for the context window
            IQR = Q3 - Q1

            # Normalize the current batch of test errors
            if IQR != 0:
                normalized_test_errors[start_idx:end_idx] = (test_errors[start_idx:end_idx] - Q1) / IQR
            else:
                normalized_test_errors[start_idx:end_idx] = test_errors[start_idx:end_idx] - Q1

        return normalized_test_errors

    def adjust_anomaly_scores(self, scores, dataset, is_train, lookback=0):
        """
        Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
        :param scores: anomaly_scores
        :param dataset: name of dataset
        :param is_train: if scores is from train set
        :param lookback: lookback (window size) used in model
        """

        # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)

        adjusted_scores = scores.copy()
        if dataset.upper() in ["ASD"]:
            root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                               "data", "asd")
            if is_train:
                with open(os.path.join(root, "processed", "asd_train_md.pkl"), "rb") as f:
                    sep_cuma_full = pickle.load(f)
            else:
                with open(os.path.join(root, "processed", "asd_test_md.pkl"), "rb") as f:
                    sep_cuma_full = pickle.load(f)

            sep_cuma = sep_cuma_full[:-1]
            buffer = np.arange(1, 20)
            i_remov = np.sort(np.concatenate((sep_cuma, np.array([i + buffer for i in sep_cuma]).flatten(),
                                              np.array([i - buffer for i in sep_cuma]).flatten())))
            i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
            i_remov = np.sort(np.unique(i_remov))
            if len(i_remov) != 0:
                adjusted_scores[i_remov] = 0

            # Normalize each concatenated part individually
            s = sep_cuma_full
            #s = [0] + sep_cuma.tolist()
            for c_start, c_end in [(s[i], s[i + 1]) for i in range(len(s) - 1)]:
                e_s = adjusted_scores[c_start: c_end + 1]
                e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
                adjusted_scores[c_start: c_end + 1] = e_s

        if dataset.upper() in ["MSL", "SMAP"]:
            root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                               "data", "smap_msl")
            if is_train:
                md = pd.read_csv(os.path.join(root, f'{dataset.lower()}_train_md.csv'))
            else:
                md = pd.read_csv(os.path.join(root, 'labeled_anomalies.csv'))
                md = md[md['spacecraft'] == dataset.upper()]

            md = md[md['chan_id'] != 'P-2']

            # Sort values by channel
            md = md.sort_values(by=['chan_id'])

            # Getting the cumulative start index for each channel
            sep_cuma = np.cumsum(md['num_values'].values) - lookback
            sep_cuma = sep_cuma[:-1]
            buffer = np.arange(1, 20)
            i_remov = np.sort(np.concatenate((sep_cuma, np.array([i + buffer for i in sep_cuma]).flatten(),
                                              np.array([i - buffer for i in sep_cuma]).flatten())))
            i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
            i_remov = np.sort(np.unique(i_remov))
            if len(i_remov) != 0:
                adjusted_scores[i_remov] = 0

            # Normalize each concatenated part individually
            sep_cuma = np.cumsum(md['num_values'].values) - lookback
            s = [0] + sep_cuma.tolist()
            for c_start, c_end in [(s[i], s[i + 1]) for i in range(len(s) - 1)]:
                e_s = adjusted_scores[c_start: c_end + 1]
                if np.isclose(np.max(e_s), np.min(e_s)):
                    e_s = e_s - np.min(e_s)
                else:
                    e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
                adjusted_scores[c_start: c_end + 1] = e_s

        return adjusted_scores

    def compute_metrics(self, scores, labels, train_scores=None, metric_types=["f1_pa", "f1", "f1_cp", "vus_pr", "vus_roc"]):
        results = {}
        if "f1_adratio" in metric_types:
            thresh = np.percentile(scores, 100 - self.anomaly_ratio)
            pred = (scores > thresh).astype(int)
            pred = adjust_predicts_(pred, labels, calc_latency=False)
            fscore = f1_score(labels, pred)
            prec = precision_score(labels, pred)
            rec = recall_score(labels, pred)
            results["f1_adratio"] = fscore
            results["precision_adratio"] = prec
            results["recall_adratio"] = rec
            results["adratio_threshold"] = thresh

        if "f1_pot" in metric_types:
            p_eval = pot_eval(train_scores, scores, labels, q=self.q, level=self.level, dynamic=self.dynamic_pot)
            results["f1_pot"] = p_eval["f1"]
            results["precision_pot"] = p_eval["precision"]
            results["recall_pot"] = p_eval["recall"]
            results["pot_threshold"] = p_eval["threshold"]

        if "f1_percentile" in metric_types:
            f1, precision, recall, threshold = get_percentile_fscore(scores, train_scores, labels, return_threshold=True, eval_fn_type="f1_adjust", percentile=self.anomaly_percentile)
            results["f1_percentile"] = f1
            results["precision_percentile"] = precision
            results["recall_percentile"] = recall
            results["percentile_threshold"] = threshold

        if "f1_zscore" in metric_types:
            f1, precision, recall, threshold = get_zscore_fscore(scores, train_scores, labels, return_threshold=True, eval_fn_type="f1_adjust", threshold_factor=self.anomaly_zscore_factor)
            results["f1_zscore"] = f1
            results["precision_zscore"] = precision
            results["recall_zscore"] = recall
            results["zscore_threshold"] = threshold

        if "f1_pa" in metric_types:
            pa_f1, pa_precision, pa_recall, pa_threshold = get_best_fscore_faster(scores, labels, return_threshold=True, eval_fn_type = "f1_adjust")
            results["f1_pa"] = pa_f1
            results["precision_pa"] = pa_precision
            results["recall_pa"] = pa_recall
            results["pa_threshold"] = pa_threshold

        if "f1" in metric_types:
            f1, precision, recall, pa_threshold = get_best_fscore_faster(scores, labels, return_threshold=True, eval_fn_type= "f1")
            results["f1"] = f1
            results["precision"] = precision
            results["recall"] = recall
            results["threshold"] = pa_threshold

        if "f1_cp" in metric_types:
            composite_f1, composite_precision, composite_recall, composite_threshold = get_best_fscore_faster(scores, labels, return_threshold=True, eval_fn_type= "f1_composite")
            results["f1_cp"] = composite_f1
            results["precision_cp"] = composite_precision
            results["recall_cp"] = composite_recall
            results["threshold_cp"] = composite_threshold

        if "vus_pr" in metric_types or "vus_roc" in metric_types:
            slidding_window = int(np.median([end - start for _, (start, end) in get_events(y_test=labels).items()]))
            vus_results = get_range_vus_roc(scores, labels, slidingWindow=slidding_window)
            r_auc_roc = vus_results["R_AUC_ROC"]
            r_auc_pr = vus_results["R_AUC_PR"]
            vus_roc = vus_results["VUS_ROC"]
            vus_pr = vus_results["VUS_PR"]
            results["r_auc_roc"] = r_auc_roc
            results["r_auc_pr"] = r_auc_pr
            results["vus_roc"] = vus_roc
            results["vus_pr"] = vus_pr

        return results
