import pandas as pd
import numpy as np
import os
from data_utils.dataset import Dataset
"""
@author: Katrina Chen 11/10
"""


class Hai_entity(Dataset):

    def __init__(self, seed: int, entity=None, reload=True, sample=True, remove_unique=False, normalize=True, normalize_type="minmax", verbose=False, one_hot=False):
        """
        :param seed: for repeatability
        :param entity: for compatibility with multi-entity datasets. Value provided will be ignored. self.entity will be
         set to same as dataset name for single entity datasets
        """
        name = "hai"
        super().__init__(name=name)
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "data", "hai")
        self.raw_path_train = os.path.join(root, f"train{entity}.csv")
        self.raw_path_test = os.path.join(root, f"test{entity}.csv")
        self.process_path_train = os.path.join(root, "processed", f"train_{entity}.csv")
        self.process_path_test = os.path.join(root, "processed", f"test_{entity}.csv")
        if not os.path.exists(os.path.join(root, "processed")):
            os.mkdir(os.path.join(root, "processed"))

        self.reload = reload
        self.seed = seed
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
            test_df: pd.DataFrame = pd.read_csv(self.raw_path_test, skiprows=0)
            train_df: pd.DataFrame = pd.read_csv(self.raw_path_train, skiprows=0)

            train_df = train_df.rename(columns={col: col.strip() for col in train_df.columns})
            test_df = test_df.rename(columns={col: col.strip() for col in test_df.columns})

            train_df["y"] = train_df["attack"]
            train_df = train_df.drop(columns=["attack", "attack_P1", "attack_P2", "attack_P3"], axis=1)
            test_df["y"] = test_df["attack"]
            test_df = test_df.drop(columns=["attack", "attack_P1", "attack_P2", "attack_P3"], axis=1)

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

            if self.sample:
                train_df = format_data(train_df)#.dropna()
                test_df = format_data(test_df).dropna()
                # train_df = train_df.groupby(np.arange(len(train_df)) // 10).agg(
                #     {**{col: "median" for col in train_df.columns[:-1]}, "y": pd.Series.mode})
                # test_df = test_df.groupby(np.arange(len(test_df)) // 10).agg(
                #     {**{col: "median" for col in test_df.columns[:-1]}, "y": pd.Series.mode})

                # test_df["y"] = test_df.y.apply(lambda x:  x[0] if isinstance(x, np.ndarray)  else x)
                # train_df["y"] = train_df.y.apply(lambda x:  x[0] if isinstance(x, np.ndarray) else x)

                train_df = train_df.drop(columns=["time"], axis=1)
                test_df = test_df.drop(columns=["time"], axis=1)
                test_df["y"] = test_df.y.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
                train_df["y"] = train_df.y.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
            else:
                train_df = train_df.drop(columns=["time"], axis=1)
                test_df = test_df.drop(columns=["time"], axis=1)
            train_df.to_csv(self.process_path_train, index=False)
            test_df.to_csv(self.process_path_test, index=False)
        else:
            train_df = pd.read_csv(self.process_path_train)
            test_df = pd.read_csv(self.process_path_test)

        # if self.sample:
        #     train_df = train_df.iloc[2060:len(train_df),:]

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


def main():
    seed = 0
    hai = Hai_entity(seed=seed, entity=1, remove_unique=False, reload=True, sample=True)
    x_train, y_train, x_test, y_test = hai.data()
    hai = Hai_entity(seed=seed, entity=2, remove_unique=False, reload=True, sample=True)
    x_train, y_train, x_test, y_test = hai.data()
    hai = Hai_entity(seed=seed, entity=3, remove_unique=False, reload=True, sample=True)
    x_train, y_train, x_test, y_test = hai.data()


def convert_time(row):
    #print(row)
    d, afternoon = row.Timestamp, row.afternoon
    if afternoon == 0 and d.hour == 12:
        d = d.replace(hour=0)
    return d


def format_data(df, is_test=True):
    if is_test:
        df["time"] = pd.to_datetime(df["time"].apply(lambda x: x.strip()), format="%Y-%m-%d %H:%M:%S")

    df = df.resample("10s", on="time").agg({**{col: "median" for col in df.columns[:-1]}, "y": pd.Series.mode})
    return df


if __name__ == '__main__':
    main()
