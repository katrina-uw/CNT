import pandas as pd
import numpy as np
import os
from data_utils.dataset import Dataset
import pickle
from csv import reader
from ast import literal_eval
from pickle import dump

"""
@author: Katrina Chen 07/28
"""


class SMAP(Dataset):

    def __init__(self, seed: int, verbose=False, normalize=False, normalize_type="minmax"):
        """
        :param seed: for repeatability
        """
        name = "smap"
        super().__init__(name=name)
        self.root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "data", "smap_msl")
        self.processed_path_train = os.path.join(self.root, "processed", "smap_train.pkl")
        self.processed_path_test = os.path.join(self.root, "processed", "smap_test.pkl")
        self.processed_path_test_label = os.path.join(self.root, "processed", "smap_test_label.pkl")
        self.outlier_class = 1
        self.causes = None
        self.seed = seed
        self.verbose = verbose
        self.normalize = normalize
        self.normalize_type = normalize_type

    def load(self):

        with open(self.processed_path_train,"rb") as f:
            train_data = pickle.load(f).reshape((-1, 25))

        with open(self.processed_path_test, "rb") as f:
            test_data = pickle.load(f).reshape((-1, 25))

        with open(self.processed_path_test_label, "rb") as f:
            test_label = pickle.load(f).reshape((-1))

        train_df = pd.DataFrame(train_data)
        print("train df len {}".format(len(train_df)))

        test_df = pd.DataFrame(test_data)
        print("test df len {}".format(len(test_df)))
        train_df["y"] = np.zeros(len(train_df))
        test_df["y"] = test_label

        X_train, y_train, X_test, y_test = self.format_data(train_df, test_df, self.outlier_class, verbose=self.verbose)
        if self.normalize:
            X_train, X_test = Dataset.standardize(X_train, X_test, verbose=self.verbose, normalization_type=self.normalize_type)
        self._data = tuple([X_train, y_train, X_test, y_test])

    def get_root_causes(self):
        return self.causes

    def get_start_position(self, is_train):
        if is_train:
            md = pd.read_csv(os.path.join(self.root, 'smap_train_md.csv'))
        else:
            md = pd.read_csv(os.path.join(self.root, 'labeled_anomalies.csv'))
            md = md[md['spacecraft'] == "SMAP"]

        md = md[md['chan_id'] != 'P-2']

        # Sort values by channel
        md = md.sort_values(by=['chan_id'])

        # Getting the cumulative start index for each channel
        sep_cuma = np.cumsum(md['num_values'].values)
        #sep_cuma = sep_cuma[:-1]
        return sep_cuma

    def preprocess(self):
        output_folder = os.path.join(self.root, "processed")
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(self.root, "labeled_anomalies.csv"), "r") as file:
            csv_reader = reader(file, delimiter=",")
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        data_info = [row for row in res if row[1] == self.name.upper() and row[0] != "P-2"]
        #self.entities = [row[0] for row in data_info]
        labels = []
        for row in data_info:
            anomalies = literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool_)
            for anomaly in anomalies:
                label[anomaly[0] : anomaly[1] + 1] = True
            labels.extend(label)

        labels = np.asarray(labels)
        print(self.name, "test_label", labels.shape)

        with open(os.path.join(output_folder, self.name + "_" + "test_label" + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(self.root, category, filename + ".npy"))
                data.extend(temp)
            data = np.asarray(data)
            print(self.name, category, data.shape)
            with open(os.path.join(output_folder, self.name + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ["train", "test"]:
            concatenate_and_save(c)


def main():
    #from algorithms.autoencoder import AutoEncoder
    seed = 0
    ds = SMAP(seed=seed)
    ds.preprocess()
    x_train, y_train, x_test, y_test = ds.data()
    # x_test = x_test[:1000]
    # y_test = y_test[:1000]
    # model = AutoEncoder(sequence_length=100, num_epochs=1, hidden_size=15, lr=1e-4, gpu=1)
    # model.fit(x_train)
    # error = model.predict(x_test)["error_tc"]
    # print(error.shape)


if __name__ == '__main__':
    main()