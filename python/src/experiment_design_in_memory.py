import gc
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
from scipy.fft import fft
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

import helper_functions


# The function below executes a single fold of the cross validation procedure, obtaining a set of results for the current iteration.
def execute_single_fold(train_index, test_index, X_all, Y_all, exp_hyperparams, net_hyperparams, metrics, iteration, class_map):
    print("TRAIN:", train_index, "TEST:", test_index)

    # Take n-1 folds for train and 1 for test, as given by sklearn indices
    X_train = X_all[train_index, :, :]
    X_test = X_all[test_index, :, :]
    Y_train = Y_all[train_index]
    Y_test = Y_all[test_index]
    print("Train shapes:", X_train.shape, Y_train.shape, "Test shapes:", X_test.shape, Y_test.shape)

    # Check distributions
    helper_functions.show_dist(Y_train)
    helper_functions.show_dist(Y_test)

    # Random shuffling within train test (no overall)
    if exp_hyperparams["SHUFFLE"]:
        X_train, Y_train, X_test, Y_test = helper_functions.shuffle_within_train_test(X_train, Y_train, X_test, Y_test)

    # Standardize/scale the data
    if exp_hyperparams["STANDARD"]:
        X_train, X_test = helper_functions.standardize_input(X_train, X_test)

    # Define fully-connected (contact or radar) signal model
    model = helper_functions.get_model(X_train, exp_hyperparams, net_hyperparams, metrics)

    if not os.path.exists("../model_checkpoints/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration)) or \
            not os.path.exists("../model_training_logs/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration)) or \
            not os.path.exists("../results/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration)):
        os.makedirs("../model_checkpoints/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration))
        os.makedirs("../model_training_logs/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration))
        os.makedirs("../results/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration))

    # Train the model
    history = model.fit(
        x=[np.squeeze(x) for x in np.split(X_train, X_train.shape[2], axis=2)],  # split individual signals
        y=to_categorical(Y_train, num_classes=exp_hyperparams["N_CLASSES"]),  # one hot encode discrete labels
        batch_size=exp_hyperparams["BATCH_SIZE"],
        epochs=exp_hyperparams["N_EPOCHS"],
        verbose=exp_hyperparams["VERBOSE"],
        validation_split=exp_hyperparams["VALIDATION"],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath="../model_checkpoints/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration) + "/best_model.hdf5", monitor='val_loss', mode='min', save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir="../model_training_logs/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration)),
            tf.keras.callbacks.LearningRateScheduler(helper_functions.scheduler),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=exp_hyperparams["VERBOSE"], patience=50)
        ]
    )

    # Evaluate the model on left-out test data
    trained_model = load_model(filepath="../model_checkpoints/" + exp_hyperparams["WHICH_DATA"] + "_kfold/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration) + "/best_model.hdf5")
    results = helper_functions.evaluate_model(trained_model, X_test, Y_test, exp_hyperparams, class_map, iteration)

    # Cleanup for GPU memory
    del model
    del trained_model
    gc.collect()
    K.clear_session()
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    return results


# The function below calls all other functions to read the data, split it, define and train a model and evaluate it in k-fold cross validation experiment.
# Finally, it does some memory cleanup.
def kfold_cv_experiment(X_all, Y_all, exp_hyperparams, net_hyperparams, metrics, class_map):
    # Clean old things from previous run (in newer versions, dont run this, since each run is saved by timestamp)
    # helper_functions.clear_old_dirs(which_data, "kfold")
    n_folds = exp_hyperparams["N_FOLDS"]

    if exp_hyperparams["FFT_ONLY"]:
        print("Transforming temporal data to 1D FFT (abs squared)...")
        for signal in range(X_all.shape[2]):
            i = 0
            for row in X_all[:, :, signal]:
                X_all[i, :, signal] = np.abs(fft(row)) ** 2
                i += 1

    if exp_hyperparams["FFT_ADDED"]:
        # FFT
        print("Adding 1D FFT (abs squared) to temporal data...")
        X_all_fft = np.copy(X_all)
        for signal in range(X_all.shape[2]):
            i = 0
            for row in X_all[:, :, signal]:
                X_all_fft[i, :, signal] = np.abs(fft(row)) ** 2
                i += 1

        X_all = np.concatenate([X_all[:, :, :], X_all_fft[:, :, :]], axis=2)
        print(X_all.shape)

    # Group_by class (for visualisation purposes)
    idx = np.argsort(Y_all)
    X_all = X_all[idx]
    Y_all = Y_all[idx]

    # Prepare kFold CV
    skf = StratifiedKFold(n_splits=n_folds)
    print("Number of splits:", skf.get_n_splits(X_all, Y_all))

    # Sanity plot of index splits
    #fig, ax = plt.subplots()
    #helper_functions.plot_cv_indices(skf, X_all, Y_all, ax, n_folds)
    #plt.show()

    iteration = 0
    all_results = pd.DataFrame(columns=["loss", "accuracy", "precision", "recall", "auc"])
    for train_index, test_index in skf.split(X_all, Y_all):
        print("====================== FOLD:", iteration, "======================")

        # Sanity check
        print("Train:", train_index)
        print("Test:", test_index)
        print("Intersection:", list(set(train_index) & set(test_index)))

        results = execute_single_fold(train_index, test_index, X_all, Y_all, exp_hyperparams, net_hyperparams, metrics, iteration, class_map)
        all_results = all_results.append(results)

        iteration += 1

    return all_results
