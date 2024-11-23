import os
import logging
import numpy as np
import pandas as pd
import abc

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class Dataset:

    def __init__(self, name: str, entity: str = None):
        self.name = name
        # self.processed_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                                                    '../data/processed/', file_name))

        self._data = None
        self.logger = logging.getLogger(__name__)
        self.train_starts = np.array([])
        self.test_starts = np.array([])
        if entity is None:
            entity = self.name
        self.entity = entity
        self.verbose = False
        self.test_anom_frac_entity = None
        self.test_anom_frac_avg = None
        self.y_test = None

    def get_anom_frac_entity(self):
        if self.y_test is None:
            self.load()
        test_anom_frac_entity = (np.sum(self.y_test)) / len(self.y_test)
        return test_anom_frac_entity

    def data(self) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
        """Return data, load if necessary"""
        if self._data is None:
            self.load()
        return self._data

    @abc.abstractmethod
    def load(self):
        """Load data"""

    def format_data(self, train_df, test_df, outlier_class=1, verbose=False):
        train_only_cols = set(train_df.columns).difference(set(test_df.columns))
        if verbose:
            print("Columns {} present only in the training set, removing them")
        train_df = train_df.drop(train_only_cols, axis=1)

        test_only_cols = set(test_df.columns).difference(set(train_df.columns))
        if verbose:
            print("Columns {} present only in the test set, removing them")
        test_df = test_df.drop(test_only_cols, axis=1)

        train_anomalies = train_df[train_df["y"] == outlier_class]
        test_anomalies: pd.DataFrame = test_df[test_df["y"] == outlier_class]
        print("Total Number of anomalies in train set = {}".format(len(train_anomalies)))
        print("Total Number of anomalies in test set = {}".format(len(test_anomalies)))
        print("% of anomalies in the test set = {}".format(len(test_anomalies) / len(test_df) * 100))
        print("number of anomalous events = {}".format(len(get_events(y_test=test_df["y"].values))))
        print(f"number of features = {len(train_df.columns)-1}")
        print(f"len of train set: {len(train_df)}, len of test set: {len(test_df)}")
        print("average event length = {}".format(np.median([end-start for _, (start, end) in get_events(y_test=test_df["y"].values).items()])))
        # Remove the labels from the data
        X_train = train_df.drop(["y"], axis=1)
        y_train = train_df["y"]
        X_test = test_df.drop(["y"], axis=1)
        y_test = test_df["y"]
        self.y_test = y_test
        return X_train, y_train, X_test, y_test

    @staticmethod
    def standardize(X_train, X_test, remove=False, verbose=False, max_clip=5, min_clip=-4, normalization_type = "minmax"):
        if normalization_type == "robust":
            scaler = RobustScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        elif normalization_type == "standard":
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        elif normalization_type == "minmax":
            min_value, max_value = X_train.min(), X_train.max()
            for col in X_train.columns:
                if max_value[col] != min_value[col]:
                    X_train[col] = (X_train[col] - min_value[col]) / (max_value[col] - min_value[col])
                    X_test[col] = (X_test[col] - min_value[col]) / (max_value[col] - min_value[col])
                    X_test[col] = np.clip(X_test[col], a_min=min_clip, a_max=max_clip)
                else:
                    assert X_train[col].nunique() == 1
                    if remove:
                        if verbose:
                            print("Column {} has the same min and max value in train. Will remove this column".format(col))
                        X_train = X_train.drop(col, axis=1)
                        X_test = X_test.drop(col, axis=1)
                    else:
                        if verbose:
                            print("Column {} has the same min and max value in train. Will scale to 1".format(col))
                        if min_value[col] != 0:
                            X_train[col] = X_train[col] / min_value[col]  # Redundant operation, just for consistency
                            X_test[col] = X_test[col] / min_value[col]
                        if verbose:
                            print("After transformation, train unique vals: {}, test unique vals: {}".format(
                            X_train[col].unique(),
                            X_test[col].unique()))

        elif normalization_type == "minmax_original":
            scaler = MinMaxScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        return X_train, X_test

    def get_start_position(self, is_train):
        return []

    def get_target_dims(self):
        """
        :param dataset: Name of dataset
        :return: index of data dimension that should be modeled (forecasted and reconstructed),
                         returns None if all input dimensions should be modeled
        """
        dataset = self.name.upper()
        if "SMAP" in dataset or "MSL" in dataset:
            return [0]
        else:
            return None


def get_dataset(dataset, normalize, seed, entity=None, normalize_type="minmax"):
    from data_utils.hai_entity import Hai_entity
    from data_utils.smd_entity import Smd_entity
    from data_utils.swat import Swat
    from data_utils.smap import SMAP
    from data_utils.msl import MSL
    from data_utils.wadi import Wadi

    if dataset == "hai":
        ds = Hai_entity(seed=seed, entity=entity, normalize=True, reload=False, normalize_type=normalize_type)
    elif dataset == "swat":
        ds = Swat(seed=seed, shorten_long=False, normalize=normalize, reload=False, normalize_type=normalize_type)
    elif dataset == "smd":
        ds = Smd_entity(seed=seed, entity=entity, normalize=normalize, normalize_type=normalize_type)
    elif dataset == "smap":
        ds = SMAP(seed=seed, normalize=normalize, normalize_type=normalize_type)
    elif dataset == "msl":
        ds = MSL(seed=seed, normalize=normalize, normalize_type=normalize_type)
    elif dataset == "wadi":
        ds = Wadi(seed=seed, normalize=normalize, normalize_type=normalize_type)
    return ds


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


def get_unidentified_events(true_y, predict_y):
    events = get_events(true_y)
    unidentified_events = []
    for index in np.where((true_y - true_y * predict_y) > 0)[0]:

        for event_num, (event_start, event_end) in events.items():
            if index >= event_start and index <= event_end:
                unidentified_events.append(event_num)

    unidentified_events = set(unidentified_events)
    return unidentified_events


def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler
