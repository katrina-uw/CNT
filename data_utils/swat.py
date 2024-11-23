import pandas as pd
import numpy as np
import os
from data_utils.dataset import Dataset
from configs import CATEGORICAL_COLUMNS

"""
@author: Astha Garg 10/19
"""


class Swat(Dataset):

    def __init__(self, seed: int, reload=True, sample=True, shorten_long=False, remove_unique=False, normalize=True, verbose=False, one_hot=False, normalize_type="minmax"):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        if shorten_long:
            name = "swat"
        else:
            name = "swat-long"
        super().__init__(name=name)
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "data", "swat")
        self.raw_path_train = os.path.join(root, "SWaT_Dataset_Normal_v1.csv")
        self.raw_path_test = os.path.join(root, "SWaT_Dataset_Attack_v0.csv")

        if shorten_long:
            self.process_path_train = os.path.join(root, "processed", "train_short.csv")
            self.process_path_test = os.path.join(root, "processed", "test_short.csv")
        else:
            self.process_path_train = os.path.join(root, "processed", "train.csv")
            self.process_path_test = os.path.join(root, "processed", "test.csv")

        self.reload = reload
        self.seed = seed
        self.shorten_long = shorten_long
        self.remove_unique = remove_unique
        self.verbose = verbose
        self.one_hot = one_hot
        self.sample = sample
        self.normalize = normalize
        self.normalize_type = normalize_type

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        if self.reload:
            test_df: pd.DataFrame = pd.read_csv(self.raw_path_test, skiprows=1)
            train_df: pd.DataFrame = pd.read_csv(self.raw_path_train, skiprows=1)

            train_df = train_df.rename(columns={col: col.strip() for col in train_df.columns})
            test_df = test_df.rename(columns={col: col.strip() for col in test_df.columns})

            train_df["y"] = train_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
            train_df = train_df.drop(columns=["Normal/Attack"], axis=1)
            test_df["y"] = test_df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
            test_df = test_df.drop(columns=["Normal/Attack"], axis=1)

            # one-hot-encoding stuff
            if self.one_hot:
                keywords = {col_name: "".join([s for s in col_name if not s.isdigit()]) for col_name in train_df.columns}
                cat_cols = [col for col in keywords.keys() if keywords[col] in ["P", "MV", "UV"]]
                one_hot_cols = [col for col in cat_cols if train_df[col].nunique() >= 3 or test_df[col].nunique() >= 3]
                print(one_hot_cols)
                one_hot_encoded = Dataset.one_hot_encoding(pd.concat([train_df, test_df], axis=0, join="inner"),
                                                           col_names=one_hot_cols)
                train_df = one_hot_encoded.iloc[:len(train_df)]
                test_df = one_hot_encoded.iloc[len(train_df):]

            # shorten the extra long anomaly to 550 points
            if self.shorten_long:
                long_anom_start = 227828
                long_anom_end = 263727
                test_df = test_df.drop(test_df.loc[(long_anom_start + 551):(long_anom_end + 1)].index,
                                      axis=0).reset_index(drop=True)

            if self.sample:
                train_df = format_data(train_df)#.dropna()
                test_df = format_data(test_df).dropna()
                # train_df = train_df.drop(columns=["Timestamp"], axis=1)
                # test_df = test_df.drop(columns=["Timestamp"], axis=1)
                test_df["y"] = test_df.y.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
                train_df["y"] = train_df.y.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
            # else:
            #     train_df = train_df.drop(columns=["Timestamp"], axis=1)
            #     test_df = test_df.drop(columns=["Timestamp"], axis=1)
            train_df.to_csv(self.process_path_train, index=False)
            test_df.to_csv(self.process_path_test, index=False)
        else:
            train_df = pd.read_csv(self.process_path_train)
            test_df = pd.read_csv(self.process_path_test)

        if self.sample:
            train_df = train_df.iloc[2060:len(train_df),:]

        # self.train_data_stamp = train_df["Timestamp"]
        # self.test_data_stamp = test_df["Timestamp"]
        train_df = train_df.set_index("Timestamp")#drop(columns=["Timestamp"], axis=1)
        test_df = test_df.set_index("Timestamp")#drop(columns=["Timestamp"], axis=1)

        self.causes_channels_names = [["MV101"], ["P102"], ["LIT101"], [], ["AIT202"], ["LIT301"], ["DPIT301"],
                                 ["FIT401"], [], ["MV304"], ["MV303"], ["LIT301"], ["MV303"], ["AIT504"],
                                 ["AIT504"], ["MV101", "LIT101"], ["UV401", "AIT502", "P501"], ["P602", "DPIT301",
                                                                                                "MV302"],
                                 ["P203", "P205"], ["LIT401", "P401"], ["P101", "LIT301"], ["P302", "LIT401"],
                                 ["P201", "P203", "P205"], ["LIT101", "P101", "MV201"], ["LIT401"], ["LIT301"],
                                 ["LIT101"], ["P101"], ["P101", "P102"], ["LIT101"], ["P501", "FIT502"],
                                 ["AIT402", "AIT502"], ["FIT401", "AIT502"], ["FIT401"], ["LIT301"]]

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        if self.normalize:
            X_train, X_test = Dataset.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose, normalization_type=self.normalize_type)

        matching_col_names = np.array([col.split("_1hot")[0] for col in train_df.columns])
        self.causes = []
        for event in self.causes_channels_names:
            event_causes = []
            for chan_name in event:
                event_causes.extend(np.argwhere(chan_name == matching_col_names).ravel())
            self.causes.append(event_causes)

        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes


def convert_time(row):
    d, afternoon = row.Timestamp, row.afternoon
    if afternoon == 0 and d.hour == 12:
        d = d.replace(hour=0)
    return d


def format_data(df, is_test=True):
    if is_test:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"].apply(lambda x: x.strip()), format="%d/%m/%Y %I:%M:%S %p")

    def middle_value(series):
        if len(series) == 0:
            return np.nan
        arr = series.to_numpy()
        arr.sort()
        middle = arr[len(arr) // 2]
        return middle

    agg_funcs = {}
    for col in df.columns:
        if col not in ["Timestamp", "y"]:
            if list(df.columns).index(col)-1 not in CATEGORICAL_COLUMNS["swat"]:
                agg_funcs[col] = "median"
            else:
                agg_funcs[col] = middle_value

    df = df.resample("10s", on="Timestamp").agg({**agg_funcs, "y": pd.Series.mode})#.reset_index(drop=True)
    return df.reset_index()



def main():
    seed = 0
    swat = Swat(seed=seed, shorten_long=False, normalize=True, reload=False, sample=True)
    x_train, y_train, x_test, y_test = swat.data()


if __name__ == '__main__':
    main()
