import pandas as pd
import numpy as np
import os
from data_utils.dataset import Dataset
from data_utils.dataset import get_events


class Smd_entity(Dataset):

    def __init__(self, seed: int, entity="machine-1-1", remove_unique=False, verbose=False, normalize=True, normalize_type="minmax"):
        """
        :param seed: for repeatability
        """
        name = "smd-" + entity
        super().__init__(name=name)
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "data", "ServerMachineDataset")
        self.seed = seed
        self.remove_unique = remove_unique
        self.entity = entity
        self.verbose = verbose
        self.normalize = normalize
        self.normalize_type = normalize_type

    @staticmethod
    def one_hot_encoding(df, col_name):
        with_dummies = pd.get_dummies(df[col_name])
        with_dummies = with_dummies.rename(columns={name: col_name + "_" + str(name) for name in with_dummies.columns})
        new_df = pd.concat([df, with_dummies], axis=1).drop(col_name, axis=1)
        return new_df

    def load(self):
        # 1 is the outlier, all other digits are normal
        OUTLIER_CLASS = 1
        train_df = pd.read_csv(os.path.join(self.base_path, "train", self.entity + ".txt"), header=None, sep=",",
                               dtype=np.float32)
        test_df = pd.read_csv(os.path.join(self.base_path, "test", self.entity + ".txt"), header=None, sep=",",
                              dtype=np.float32)
        train_df["y"] = np.zeros(train_df.shape[0])

        # Get test anomaly labels
        test_labels = np.genfromtxt(os.path.join(self.base_path, "test_label", self.entity + ".txt"), dtype=np.float32,
                                    delimiter=',')
        test_df["y"] = test_labels

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, OUTLIER_CLASS, verbose=self.verbose)
        if self.normalize:
            X_train, X_test = Dataset.standardize(X_train, X_test, remove=self.remove_unique, verbose=self.verbose, normalization_type=self.normalize_type)

        # Retrieve for each anomalous sequence the set of root causes of the anomaly
        self.causes = []
        causes_df = pd.read_csv(os.path.join(self.base_path, "interpretation_label", self.entity + ".txt"), header=None,
                                sep=":", names=["duration", "causes"])
        causes_df["starts"] = causes_df["duration"].str.split("-").map(lambda x: int(x[0]))
        events_df = pd.DataFrame(list(get_events(y_test=y_test).values()), columns=["starts", "ends"])
        merged_df = pd.DataFrame.merge(events_df, causes_df, how="outer", left_on="starts", right_on="starts")\
            .sort_values(by="starts")
        starts_with_missing_causes = merged_df["starts"][pd.isna(merged_df["causes"])]
        if len(starts_with_missing_causes) > 0:
            print("Events starting at {} don't have root causes. Will be filled with all channels as root cause".format(
                starts_with_missing_causes.values))
        # if an event was present but root causes not provided, assign all channels to true root cause
        merged_df["causes"] = merged_df["causes"].fillna(str(list(range(1, X_test.shape[1]+1))).replace(" ", "").replace(
            "[", "").replace("]", ""))
        for row in merged_df["causes"]:
            event_causes = [int(cause) - 1 for cause in row.split(",")]
            self.causes.append(event_causes)

        self._data = tuple([X_train.astype(float), y_train, X_test.astype(float), y_test])

    def get_root_causes(self):
        return self.causes


def average_statistics():
    seed = 0
    file_lists = ["machine-1-1", "machine-1-2", "machine-1-3", "machine-1-4", "machine-1-5", "machine-1-6", "machine-1-7", "machine-1-8", \
                 "machine-2-1", "machine-2-2", "machine-2-3", "machine-2-4", "machine-2-5", "machine-2-6", "machine-2-7", "machine-2-8", "machine-2-9",\
                 "machine-3-1", "machine-3-2", "machine-3-3", "machine-3-4", "machine-3-5", "machine-3-6", "machine-3-7", "machine-3-8", "machine-3-9", "machine-3-10", "machine-3-11"]

    len_events = 0
    len_train = 0
    len_test = 0
    for entity in file_lists:
        smd = Smd_entity(seed=seed, remove_unique=False, entity=entity)
        x_train, y_train, x_test, y_test = smd.data()
        events = get_events(y_test)
        len_events+=len(events)
        len_train+=len(x_train)
        len_test+=len(x_test)

    print("average_events",len_events/28)
    print("average_train",len_train/28)
    print("average_test",len_test/28)

if __name__ == '__main__':
    average_statistics()
