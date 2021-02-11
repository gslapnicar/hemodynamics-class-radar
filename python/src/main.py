import data_preparation, experiment_design_in_memory, helper_functions
import tensorflow as tf
import datetime
import pandas as pd
import numpy as np
import random
from pathlib import Path
import logging
tf.get_logger().setLevel(logging.ERROR)


PIPELINE_STEPS = {
    "data_preparation" : True,
    "train_test_exp"   : False,
    "kFold_exp"        : True,
    "loo_exp"          : False
}

CLASS_MAP = {
    #"Other":    0,
    "Resting":  0,
    "Valsalva": 1,
    "Apnea":    2,
    "TiltUp":   3,
    "TiltDown": 4
}

SELECTED_CLASSES = ["Resting", "Valsalva", "Apnea", "TiltUp", "TiltDown"]

METRICS = [
    tf.keras.metrics.CategoricalCrossentropy(name="loss"),
    tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

overall_df  = None
iteration   = 0
for WHICH_DATA in ["contact", "radar"]:
    for WIN_WIDTH in [10, 20]: # in seconds
        if PIPELINE_STEPS["data_preparation"]:
            print("[START] Begun data preparation...")

            # Set some basic parameters for the preparation, set the data location and run the function.
            FILE_PATH   = "../../matlab/existing_code_starting_point/outputs/output_full_data_re.mat"
            FS_RE       = 100  # everything in _re files downsampled to 100 Hz (done in matlab, using shape-preserving piecewise cubic interpolation)
            PLOT        = False
            NORMALISE   = True

            SIGNAL_NAME_MAP = {
                "contact_bp":           0,
                "contact_ecg1":         1,
                "contact_ecg2":         2,
                "contact_icg":          3,
                "contact_respiration":  4,
                "contact_z0":           5,
                "radar_dist":           6,
                "radar_heartSound":     7,
                "radar_i":              8,
                "radar_pulse":          9,
                "radar_q":              10,
                "radar_resp":           11
            }

            data_preparation.prepare_and_save(FILE_PATH, FS_RE, WIN_WIDTH, CLASS_MAP, SIGNAL_NAME_MAP, NORMALISE, PLOT)

            # Read the full data
            X_all = np.load("../model_inputs_and_targets/kfold/X_all_" + WHICH_DATA + ".npy")
            Y_all = np.load("../model_inputs_and_targets/kfold/Y_all.npy")
            print("Original shape:", X_all.shape, Y_all.shape)

            # Make subsets of data of selected classes
            if len(SELECTED_CLASSES) < 5:
                print("Taking subsets of data for classes:", SELECTED_CLASSES)
                X_new, Y_new = [], []
                for selected_class in SELECTED_CLASSES:
                    idx = (Y_all == CLASS_MAP[selected_class]).nonzero()
                    X_new.append(X_all[idx[0], :, :])
                    Y_new.append(Y_all[idx])

                X_all = np.concatenate(X_new, axis=0)
                Y_all = np.concatenate(Y_new, axis=0)
                Y_all = np.subtract(Y_all, np.min(Y_all)).astype(int)
                print("Subset shapes:", X_all.shape, Y_all.shape)

                class_map_new = {selected_class: CLASS_MAP[selected_class] for selected_class in SELECTED_CLASSES}
                #for k, v in class_map_new.items():
                #    class_map_new[k] = v-1
                class_map = class_map_new
            else:
                class_map = CLASS_MAP

            # Oversample minority classes in a smart way using SMOTE
            helper_functions.show_dist(Y_all)
            X_all, Y_all = helper_functions.oversample_smote(X_all, Y_all)

            print("[END] Finished data preparation.")

        for i in range(25):
            if PIPELINE_STEPS["kFold_exp"]:
                print()
                print("=== " + str(iteration) + " [START] Begun kFold split ML experiment...")

                NET_HYPERPARAMS = {
                    "N_HIDDEN_LAYERS": random.choice([1, 2]),
                    "N_UNITS": random.choice([32, 64]),
                    "N_FILTERS": random.choice([32, 64]),
                    "KERNEL_SIZE": random.choice([4, 8, 16]),
                    "LOSS": "categorical_crossentropy",
                    "ACTIVATION": random.choice(["relu", "tanh"]),
                    "LAST_ACTIVATION": "softmax",
                    "DROPOUT": random.choice([0.2, 0.3]),
                    "LR": random.choice([0.005, 0.01, 0.05]),
                    "INITIALIZER": "GlorotNormal",
                    "OPTIMIZER": random.choice(["adam", "sgd", "adadelta", "rmsprop"]),
                    "REGULARIZER": random.choice(["l1", "l2"]),
                    "REGULAR": random.choice([0.001, 0.005, 0.01])
                }

                EXP_HYPERPARAMS = {
                    "N_CLASSES": len(SELECTED_CLASSES),
                    "BATCH_SIZE": random.choice([64, 128]),
                    "N_EPOCHS": 30,
                    "Fs": FS_RE,
                    "WINDOW_LEN": WIN_WIDTH,
                    "TRAIN_AMNT": random.choice([0.7, 0.8]),
                    "VALIDATION": random.choice([0.2, 0.3]),
                    "VERBOSE": 0,
                    "N_FOLDS": 5,
                    "SHUFFLE": True,
                    "STANDARD": True,
                    "OVERSAMPLE": True,
                    "FFT_ONLY": False,
                    "FFT_ADDED": True,
                    "WHICH_DATA": WHICH_DATA,
                    "WHICH_MODEL": "fully_connected",
                    "WHICH_EXP": "kfold",
                    "SELECTED_CLASSES": SELECTED_CLASSES,
                    "RUN_ID": datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
                }

                all_results = experiment_design_in_memory.kfold_cv_experiment(X_all, Y_all, EXP_HYPERPARAMS, NET_HYPERPARAMS, METRICS, class_map=class_map)
                print(all_results)

                # Save results
                Path("../results/" + EXP_HYPERPARAMS["WHICH_DATA"] + "_kfold/" + EXP_HYPERPARAMS["RUN_ID"]).mkdir(parents=True, exist_ok=True)
                FILENAME = "../results/" + EXP_HYPERPARAMS["WHICH_DATA"] + "_kfold/" + EXP_HYPERPARAMS["RUN_ID"] + "/results_" + EXP_HYPERPARAMS["WHICH_MODEL"] + ".xlsx"
                all_results.to_excel(FILENAME, sheet_name="results")
                writer = pd.ExcelWriter(FILENAME, engine='openpyxl', mode='a')
                exclude_keys = ['SELECTED_CLASSES']
                pd.DataFrame(data={k: EXP_HYPERPARAMS[k] for k in set(list(EXP_HYPERPARAMS.keys())) - set(exclude_keys)}, index=[0]).T.to_excel(writer, sheet_name="exp_hyperparams")
                pd.DataFrame(data=NET_HYPERPARAMS, index=[0]).T.to_excel(writer, sheet_name="net_hyperparams")
                writer.save()

                print("=== " + str(iteration) + " [END] Finished kFold split ML experiment.")

            if iteration == 0:
                clmns = ["accuracy"] + list(NET_HYPERPARAMS.keys()) + list(EXP_HYPERPARAMS.keys())
                df = pd.DataFrame(columns=clmns)
                df.loc[iteration, "accuracy"] = all_results["accuracy"].mean()
                for k, v in NET_HYPERPARAMS.items():
                    df.loc[iteration, k] = v
                for k, v in EXP_HYPERPARAMS.items():
                    df.loc[iteration, k] = v
                df.to_csv("../results/cumulative_results_"+WHICH_DATA+"_FFT_and_contact.csv")
            else:
                df.loc[iteration, "accuracy"] = all_results["accuracy"].mean()
                for k, v in NET_HYPERPARAMS.items():
                    df.loc[iteration, k] = v
                for k, v in EXP_HYPERPARAMS.items():
                    df.loc[iteration, k] = v
                df.to_csv("../results/cumulative_results_"+WHICH_DATA+"_FFT_and_contact.csv", mode="a", header=False)
            iteration += 1

