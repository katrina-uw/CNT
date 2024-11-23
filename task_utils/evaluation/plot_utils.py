import numpy as np
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()
import os
from configs import n_dims
import json
import pandas as pd
import plotly as py
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


class Plotter:

    def __init__(self, model_name, dataset, post_fix=None):

        self.model_name = model_name
        self.dataset = dataset
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    '../outputs'))
        if post_fix is None:
            self.result_path = os.path.join(self.base_dir, self.model_name, self.dataset)
        else:
            self.result_path = os.path.join(self.base_dir, self.model_name, self.dataset, post_fix)
        self.train_output = None
        self.test_output = None
        self.labels_available = True
        self.pred_cols = None
        self._load_results()
        self.train_output["timestamp"] = self.train_output.index
        self.test_output["timestamp"] = self.test_output.index

        # config_path = f"{self.result_path}/config.txt"
        # with open(config_path) as f:
        #     self.lookback = json.load(f)["lookback"]

        if "smd" in self.result_path:
            self.pred_cols = [f"feat_{i}" for i in range(n_dims["smd"])]
        elif "swat" in self.result_path:
            self.pred_cols = [f"feat_{i}" for i in range(n_dims["swat"])]
        elif "wadi" in self.result_path:
            self.pred_cols = [f"feat_{i}" for i in range(n_dims["wadi"])]
        elif "smap" in self.result_path or "msl" in self.result_path or "jump-diffusion" in self.result_path:
            self.pred_cols = ["feat_1"]

    def _load_results(self):

        print(f"Loading results of {self.result_path}")
        train_output = pd.read_pickle(f"{self.result_path}/train_output.pkl")
        train_output.to_pickle(f"{self.result_path}/train_output.pkl")
        train_output["true_label_global"] = 0
        test_output = pd.read_pickle(f"{self.result_path}/test_output.pkl")

        # Because for SMAP and MSL only one feature is predicted
        if 'SMAP'.lower() in self.result_path or 'MSL'.lower() in self.result_path or 'jump-diffusion' in self.result_path:
            train_output[f'pred_label_0'] = train_output['pred_label_global']
            train_output[f'score_0'] = train_output['score_global']
            train_output[f'thresh_0'] = train_output['thresh_global']

            test_output[f'pred_label_0'] = test_output['pred_label_global']
            test_output[f'score_0'] = test_output['score_global']
            test_output[f'thresh_0'] = test_output['thresh_global']

        self.train_output = train_output
        self.test_output = test_output

    def result_summary(self):
        path = f"{self.result_path}/summary.txt"
        if not os.path.exists(path):
            print(f"Folder {self.result_path} do not have a summary.txt file")
            return
        try:
            print("Result summary:")
            with open(path) as f:
                result_dict = json.load(f)
                epsilon_result = result_dict["epsilon_result"]
                pot_result = result_dict["pot_result"]
                bf_results = result_dict["bf_result"]
                print(f'Epsilon:')
                print(f'\t\tprecision: {epsilon_result["precision"]:.2f}, recall: {epsilon_result["recall"]:.2f}, F1: {epsilon_result["f1"]:.2f}')
                print(f'POT:')
                print(f'\t\tprecision: {pot_result["precision"]:.2f}, recall: {pot_result["recall"]:.2f}, F1: {pot_result["f1"]:.2f}')
                print(f'Brute-Force:')
                print(f'\t\tprecision: {bf_results["precision"]:.2f}, recall: {bf_results["recall"]:.2f}, F1: {bf_results["f1"]:.2f}')
                return epsilon_result, pot_result, bf_results
        except FileNotFoundError as e:
            print(e)

    def get_series_color(self, y):
        if np.average(y) >= 0.95:
            return "black"
        elif np.average(y) == 0.0:
            return "black"
        else:
            return "black"


    def get_y_height(self, y):
        if np.average(y) >= 0.95:
            return 1.5
        elif np.average(y) == 0.0:
            return 0.1
        else:
            return max(y) + 0.1


    def get_anomaly_sequences(self, values):
        splits = np.where(values[1:] != values[:-1])[0] + 1
        if values[0] == 1:
            splits = np.insert(splits, 0, 0)

        a_seqs = []
        for i in range(0, len(splits) - 1, 2):
            a_seqs.append([splits[i], splits[i + 1] - 1])

        if len(splits) % 2 == 1:
            a_seqs.append([splits[-1], len(values) - 1])

        return a_seqs


    def create_shapes(self, ranges, sequence_type, _min, _max, plot_values, xref=None, yref=None):
        """
        Create shapes for regions to highlight in plotly (true and predicted anomaly sequences).

        :param ranges: tuple of start and end indices for anomaly sequences for a feature
        :param sequence_type: "predict" if predicted values else "true" if actual values. Determines colors.
        :param _min: min y value of series
        :param _max: max y value of series
        :param plot_values: dictionary of different series to be plotted

        :return: list of shapes specifications for plotly
        """

        if _max is None:
            _max = max(plot_values["errors"])

        if sequence_type is None:
            color = "blue"
        else:
            if sequence_type == "true":
                color = "red"
            elif sequence_type == "predicted":
                color = "blue"
            elif sequence_type == "individual_true":
                color = "green"
            #color = "red" if sequence_type == "true" "blue"
        shapes = []

        for r in ranges:
            w = 5
            x0 = r[0] - w
            x1 = r[1] + w
            shape = {
                "type": "rect",
                "x0": x0,
                "y0": _min,
                "x1": x1,
                "y1": _max,
                "fillcolor": color,
                "opacity": 0.08,
                "line": {
                    "width": 0,
                },
            }
            if xref is not None:
                shape["xref"] = xref
                shape["yref"] = yref

            shapes.append(shape)

        return shapes


    def plot_global_predictions(self, type="test"):
        if type == "test":
            data_copy = self.test_output.copy()
        else:
            data_copy = self.train_output.copy()

        fig, axs = plt.subplots(
            3,
            figsize=(30, 10),
            sharex=True,
        )
        axs[0].plot(data_copy[f"score_global"], c="r", label="anomaly scores")
        axs[0].plot(data_copy["thresh_global"], linestyle="dashed", c="black", label="threshold")
        axs[1].plot(data_copy["pred_label_global"], label="predicted anomalies", c="orange")
        if self.labels_available and type == "test":
            axs[2].plot(
                data_copy["true_label_global"],
                label="actual anomalies",
            )
        axs[0].set_ylim([0, 5 * np.mean(data_copy["thresh_global"].values)])
        fig.legend(prop={"size": 20})
        plt.show()

    def plotly_global_predictions(self, is_test=True):
        is_test = True
        if is_test is False:
            data_copy = self.train_output.copy()
            is_test = False
        else:
            data_copy = self.test_output.copy()

        tot_anomaly_scores = data_copy["score_global"].values
        pred_anomaly_sequences = self.get_anomaly_sequences(data_copy[f"pred_label_global"].values)
        threshold = data_copy['thresh_global'].values
        y_min = -0.1
        y_max = 5 * np.mean(threshold) # np.max(tot_anomaly_scores)
        shapes = self.create_shapes(pred_anomaly_sequences, "pred", y_min, y_max, None)
        if is_test:
            true_anomaly_sequences = self.get_anomaly_sequences(data_copy[f"true_label_global"].values)
            shapes2 = self.create_shapes(true_anomaly_sequences, "true", y_min, y_max, None)
            shapes.extend(shapes2)

        layout = {
            "title": f"{type} set | Total error, predicted anomalies in blue, true anomalies in red if available "
                     f"(making correctly predicted in purple)",
            "shapes": shapes,
            "yaxis": dict(range=[0, y_max]),
            "height": 400,
            "width": 1500
        }

        fig = go.Figure(
            data=[go.Scatter(x=data_copy["timestamp"], y=tot_anomaly_scores, name='Error', line=dict(width=1, color="red")),
                  go.Scatter(x=data_copy["timestamp"], y=threshold, name='Threshold', line=dict(color="black", width=1, dash="dash"))],
            layout=layout,
        )
        py.offline.iplot(fig)

    def plot_all_features_with_anomaly(self, start=None, end=None, type="test", feature_prefix=None):

        if type == "test":
            data_copy = self.test_output.copy()
        else:
            data_copy = self.train_output.copy()

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            data_copy = data_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            data_copy = data_copy.iloc[: end - start, :]

        data_copy = data_copy.drop(columns=['timestamp', 'score_global', 'thresh_global'])
        if feature_prefix is None:
            cols = [c for c in data_copy.columns if not (c.startswith('thresh_') or c.startswith('pred_label_')) or c.startswith("cause_")]
        else:
            cols = [c for c in data_copy.columns if c.startswith(feature_prefix)]
        data_copy = data_copy[cols]

        num_subplots = len(data_copy.columns)

        fig, axs = plt.subplots(
            num_subplots,
            figsize=(20, num_subplots),
            #sharex=True,
        )

        for i, col in enumerate(data_copy.columns):
            axs[i].plot(data_copy[col], c = "black")
            if self.labels_available and type == "test":
                axs[i].plot(
                    data_copy["true_label_global"],
                    label="actual anomalies",
                )
            #axs[0].set_ylim([0, 5 * np.mean(data_copy["thresh_global"].values)])
            #fig.legend(prop={"size": 20})
        plt.show()

    def plot_all_features(self, start=None, end=None, type="test", feature_prefix=None):
        """
        Plotting all features, using the following order:
            - forecasting for feature i
            - reconstruction for feature i
            - true value for feature i
            - anomaly score (error) for feature i
        """
        if type == "train":
            data_copy = self.train_output.copy()
        elif type == "test":
            data_copy = self.test_output.copy()

        data_copy = data_copy.drop(columns=['timestamp', 'score_global', 'thresh_global'])
        if feature_prefix is None:
            cols = [c for c in data_copy.columns if not (c.startswith('thresh_') or c.startswith('pred_label_')) or c.startswith("cause_")]
        else:
            cols = [c for c in data_copy.columns if c.startswith(feature_prefix)]
        data_copy = data_copy[cols]

        if start is not None and end is not None:
            assert start < end
        if start is not None:
            data_copy = data_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            data_copy = data_copy.iloc[: end - start, :]

        num_cols = data_copy.shape[1]
        plt.tight_layout()
        colors = ["gray", "gray", "gray", "r"] * (num_cols // 4) + ["b", "g"]
        data_copy.plot(subplots=True, figsize=(20, num_cols), style=colors)#, ylim=(0, 1.5), style=colors)
        plt.show()

    def plot_all_features(self, start=None, end=None, type="test", feature_prefix=None, indexes=None):
        """
        Plotting all features, using the following order:
            - forecasting for feature i
            - reconstruction for feature i
            - true value for feature i
            - anomaly score (error) for feature i
        """
        if type == "train":
            data_copy = self.train_output.copy()
        elif type == "test":
            data_copy = self.test_output.copy()

        data_copy = data_copy.drop(columns=['timestamp', 'score_global', 'thresh_global'])
        if feature_prefix is None:
            cols = [c for c in data_copy.columns if not (c.startswith('thresh_') or c.startswith('pred_label_')) or c.startswith("cause_")]
        else:
            cols = [c for c in data_copy.columns if c.startswith(feature_prefix)]

        if indexes is not None:
            available_cols = []
            for i in indexes:
                for col in cols:
                    if col.endswith(f"_{i}"):
                        available_cols.append(col)

            cols = available_cols
        data_copy = data_copy[cols]



        if start is not None and end is not None:
            assert start < end
        if start is not None:
            data_copy = data_copy.iloc[start:, :]
        if end is not None:
            start = 0 if start is None else start
            data_copy = data_copy.iloc[: end - start, :]

        num_cols = data_copy.shape[1]
        plt.tight_layout()
        colors = ["gray", "gray", "gray", "r"] * (num_cols // 4) + ["b", "g"]
        data_copy.plot(subplots=True, figsize=(20, num_cols), style=colors)#, ylim=(0, 1.5), style=colors)
        plt.show()

    def plot_anomaly_segments(self, type="test", num_aligned_segments=None, show_boring_series=False):
        """
        Finds collective anomalies, i.e. feature-wise anomalies that occur at the same time, and visualize them
        """
        is_test = True
        if type == "train":
            data_copy = self.train_output.copy()
            is_test = False
        elif type == "test":
            data_copy = self.test_output.copy()

        def get_pred_cols(df):
            pred_cols_to_remove = []
            col_names_to_remove = []
            for i, col in enumerate(self.pred_cols):
                y = df[f"true_{i}"].values
                if np.average(y) >= 0.95 or np.average(y) == 0.0:
                    pred_cols_to_remove.append(col)
                    cols = list(df.columns[4 * i : 4 * i + 4])
                    col_names_to_remove.extend(cols)

            df.drop(col_names_to_remove, axis=1, inplace=True)
            return [x for x in self.pred_cols if x not in pred_cols_to_remove]

        non_constant_pred_cols = self.pred_cols if show_boring_series else get_pred_cols(data_copy)

        fig = make_subplots(
            rows=len(non_constant_pred_cols),
            cols=1,
            vertical_spacing=0.4 / len(non_constant_pred_cols),
            shared_xaxes=True,
        )

        timestamps = None
        shapes = []
        annotations = []
        for i in range(len(non_constant_pred_cols)):
            new_idx = int(data_copy.columns[4 * i].split("_")[-1])
            values = data_copy[f"true_{new_idx}"].values

            anomaly_sequences = self.get_anomaly_sequences(data_copy[f"pred_label_{new_idx}"].values)

            y_min = -0.1
            y_max = 2  # 0.5 * y_max

            j = i + 1
            xref = f"x{j}" if i > 0 else "x"
            yref = f"y{j}" if i > 0 else "y"
            anomaly_shape = self.create_shapes(
                anomaly_sequences, None, y_min, y_max, None, xref=xref, yref=yref, is_test=is_test
            )
            shapes.extend(anomaly_shape)

            fig.append_trace(
                go.Scatter(x=timestamps, y=values, line=dict(color=self.get_series_color(values), width=1)), row=i + 1, col=1
            )
            fig.update_yaxes(range=[-0.1, self.get_y_height(values)], row=i + 1, col=1)

            annotations.append(
                dict(
                    # xref="paper",
                    xanchor="left",
                    yref=yref,
                    text=f"<b>{non_constant_pred_cols[i].upper()}</b>",
                    font=dict(size=10),
                    showarrow=False,
                    yshift=35,
                    xshift=(-523),
                )
            )

        colors = ["blue", "green", "red", "black", "orange", "brown", "aqua", "hotpink"]
        taken_shapes_i = []
        keep_segments_i = []
        corr_segments_count = 0
        for nr, i in enumerate(range(len(shapes))):
            corr_shapes = [i]
            shape = shapes[i]
            shape["opacity"] = 0.3
            shape_x = shape["x0"]

            for j in range(i + 1, len(shapes)):
                if j not in taken_shapes_i and shapes[j]["x0"] == shape_x:
                    corr_shapes.append(j)

            if num_aligned_segments is not None:
                if num_aligned_segments[0] == ">":
                    num = int(num_aligned_segments[1:])
                    keep_segment = len(corr_shapes) >= num
                else:
                    num = int(num_aligned_segments)
                    keep_segment = len(corr_shapes) == num

                if keep_segment:
                    keep_segments_i.extend(corr_shapes)
                    taken_shapes_i.extend(corr_shapes)
                    if len(corr_shapes) != 1:
                        for shape_i in corr_shapes:
                            shapes[shape_i]["fillcolor"] = colors[corr_segments_count % len(colors)]
                        corr_segments_count += 1

        if num_aligned_segments is not None:
            shapes = np.array(shapes)
            shapes = shapes[keep_segments_i].tolist()

        fig.update_layout(
            height=1800,
            width=1200,
            shapes=shapes,
            template="simple_white",
            annotations=annotations,
            showlegend=False)

        fig.update_yaxes(ticks="", showticklabels=False, showline=True, mirror=True)
        fig.update_xaxes(ticks="", showticklabels=False, showline=True, mirror=True)
        py.offline.iplot(fig)