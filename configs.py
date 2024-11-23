best_configs = {}
strides = {"swat": 1, "hai": 1, "wadi": 1, "msl": 1, "smap": 1}
n_dims = {"swat": 51, "wadi": 123, "msl": 55, "smap": 25, "hai": 79}
reg_levels = {"msl": 0, "smap":0, "swat":0, "wadi":0, "hai":0}
anomaly_ratio_dict = {"msl": 1, "smap": 1, "swat": 10, "wadi": 1, "hai": 1}
anomaly_percentile_dict = {"swat": 99, "wadi": 99, "msl": 99, "smap": 99, "hai": 99}
anomaly_zscore_factor_dict = {"swat": 3, "wadi": 3, "msl": 3, "smap": 3, "hai": 3}
level_q_dict = {
    "smap": (0.9, 0.005),
    "msl": (0.9, 0.001),
    "smd": (0.9925, 0.001),
    "skab": (0.9, 0.001),
    "swat": (0.9, 0.001),
    "asd": (0.9, 0.001),
    "psm": (0.9, 0.001),
    "wadi": (0.9, 0.001),
    "hai": (0.9, 0.001),
    "todsuni": (0.9, 0.001),
}
pca_expl_var = 0.9

batch_size = 64
num_epochs = 50
patience = 10
lr = 1e-3
train_val_pc = 0.2

NUMERICAL_COLUMNS = {'smap': (0,),
                     'msl': (0,),
                     'smd': tuple(list(range(7)) + list(range(8, 38)))
                     }

CATEGORICAL_COLUMNS = {'smap': range(1, 25),
                       'msl': range(1, 55),
                       'smd': (7,),
                       'swat': tuple([2, 3, 4, 9] + list(range(11, 16)) + list(range(19, 25)) \
                                     + list(range(29, 34)) + [42, 43, 48, 49, 50]),
                       'wadi': tuple([6, 7] + list(range(9, 19)) + list(range(47, 59)) \
                                     + list(range(68, 81)) + [82, 84, 87] + list(range(91, 97)) \
                                     + [111] + list(range(113, 120)) + [121])
                       }

IGNORED_COLUMNS = {'swat' : (10,),
                   'wadi' : (102,)
                  }

NUMERICAL_COLUMNS['swat'] = tuple([i for i in range(0, 51) if (i not in CATEGORICAL_COLUMNS['swat'])\
                                   and (i not in IGNORED_COLUMNS['swat'])])
NUMERICAL_COLUMNS['wadi'] = tuple([i for i in range(0, 123) if (i not in CATEGORICAL_COLUMNS['wadi'])\
                                   and (i not in IGNORED_COLUMNS['wadi'])])


def get_best_config(algo_name, ds_name):

    if algo_name == "CNT":
        win_size_dict = {"msl": 5, "swat": 5, "wadi": 5, "smap": 5, "hai": 5}
        temperature_dict = {"msl": 0.1, "swat": 0.1, "wadi": 0.1, "smap": 0.1, "hai": 0.1}
        transformation_dict = {"msl": 6, "swat": 6, "wadi": 6, "smap": 6, "hai": 6}
        n_layer_dict = {"msl": 8, "swat": 8, "wadi": 8, "smap": 8, "hai": 8}
        hidden_dim = {"msl": 64, "swat": 64, "wadi": 64, "smap": 64, "hai": 64}
        seq_lens_static = {"msl": 30, "swat": 30, "wadi": 30, "smap": 30, "hai": 30}
        normalize_dict = {"msl": False, "swat": False, "wadi": True, "smap": False, "hai": False}
        best_configs["CNT"] = {
            "win_size": win_size_dict[ds_name],
            "seq_len": seq_lens_static[ds_name],
            "stride": strides[ds_name],
            "patience": 10,
            "train_val_percentage": train_val_pc,
            "batch_size": 256,
            "num_epochs": 30,  # num_epochs,
            "enc_nlayers": 6,
            "enc_bias": False,
            "trans_nlayers": 5,
            "batch_norm": False,
            "trans_type": "mul",
            "feature_dim": n_dims[ds_name],
            "n_layers": n_layer_dict[ds_name],
            "is_skip": False,
            "is_residual": True,
            "is_graph_conv": False,
            "lr": 1e-3,
            "topK": 20,
            "hidden_dim": hidden_dim[ds_name],
            "dropout": 0,
            "dilation_exp": 2,
            "loss_type": "contextual_OCC_featureNTL",
            "temperature": temperature_dict[ds_name],
            "n_transformations": transformation_dict[ds_name],
            "error_type": "all",
            "normalize": normalize_dict[ds_name],
            "normalize_type": "minmax",
            "num_workers": 0,
            "opt": "adam",
            "opt_scheduler": "step",
            "opt_restart": 0,
            "opt_decay_step": 10,
            "opt_decay_rate": 0.1,
            "weight_decay": 0.,
        }
    else:
        return {"batch_size": batch_size,
                "num_epochs": 50,
                "lr": 0.0001,
                "patience": patience,
                "train_val_percentage": train_val_pc,
                "dropout": 0.0,
                "stride": strides[ds_name],
                "feature_dim": n_dims[ds_name],
                "sequence_length": 100}

    return best_configs[algo_name]

