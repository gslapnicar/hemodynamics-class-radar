import data_preparation, experiment_design_in_memory
import tensorflow as tf
import datetime
import pandas as pd
import random
from pathlib import Path
import warnings
import itertools

FS_RE       = 100
WIN_WIDTH   = 10
WHICH_DATA  = "contact"

NET_HYPERPARAMS = {
    "N_HIDDEN_LAYERS":  random.choice([1, 2, 3]),
    "N_UNITS":          random.choice([32, 64, 128]),
    "N_FILTERS":        random.choice([32, 64, 128]),
    "KERNEL_SIZE":      random.choice([4, 8, 16]),
    "LOSS":             "categorical_crossentropy",
    "ACTIVATION":       random.choice(["relu", "tanh"]),
    "LAST_ACTIVATION":  "softmax",
    "DROPOUT":          random.choice([0.2, 0.3, 0.4]),
    "LR":               random.choice([0.005, 0.01, 0.05]),
    "INITIALIZER":      "GlorotNormal",
    "OPTIMIZER":        random.choice(["adam", "sgd", "adadelta", "rmsprop"]),
    "REGULARIZER":      random.choice(["l1", "l2"]),
    "REGULAR":          random.choice([0.001, 0.005, 0.01])
}

EXP_HYPERPARAMS = {
    "N_CLASSES":        5,
    "BATCH_SIZE":       random.choice([64, 128]),
    "N_EPOCHS":         100,
    "Fs":               FS_RE,
    "WINDOW_LEN":       WIN_WIDTH,
    "TRAIN_AMNT":       random.choice([0.7, 0.8]),
    "VALIDATION":       random.choice([0.2, 0.3]),
    "VERBOSE":          0,
    "N_FOLDS":          5,
    "SHUFFLE":          True,
    "STANDARD":         True,
    "OVERSAMPLE":       True,
    "FFT_ONLY":         False,
    "FFT_ADDED":        False,
    "WHICH_DATA":       WHICH_DATA,
    "WHICH_MODEL":      "fully_connected",
    "WHICH_EXP":        "kfold",
    "SELECTED_CLASSES": ["Apnea", "Valsalva", "Resting", "TiltUp", "TiltDown"],
    "RUN_ID":           datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
}

print(EXP_HYPERPARAMS)
exclude_keys = ['SELECTED_CLASSES']
test = pd.DataFrame(data={k: EXP_HYPERPARAMS[k] for k in set(list(EXP_HYPERPARAMS.keys())) - set(exclude_keys)}, index=[0])
test.to_csv("test.csv")
print(test)

for i in range(5):
    test.to_csv("test.csv", mode="a", header=False)

'''
Path("../results/" + EXP_HYPERPARAMS["WHICH_DATA"] + "_kfold/" + EXP_HYPERPARAMS["RUN_ID"]).mkdir(parents=True, exist_ok=True)
FILENAME = "../results/" + EXP_HYPERPARAMS["WHICH_DATA"] + "_kfold/" + EXP_HYPERPARAMS["RUN_ID"] + "/results_" + EXP_HYPERPARAMS["WHICH_MODEL"] + ".xlsx"

clmns = ["accuracy"] + list(NET_HYPERPARAMS.keys()) + list(EXP_HYPERPARAMS.keys())
overall_df = pd.DataFrame(columns=clmns)

overall_df.loc[0, "accuracy"] = 0.7
for k, v in NET_HYPERPARAMS.items():
    overall_df.loc[0, k] = v
for k, v in EXP_HYPERPARAMS.items():
    overall_df.loc[0, k] = v
print(overall_df)

#overall_df.to_excel("../results/massive1.xlsx")
'''
