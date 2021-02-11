import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import models


# This function clears old logs and checkpoints used in previous runs (to save space).
def clear_old_dirs(which_data, which_experiment):
    tf.keras.backend.clear_session()

    if which_data in ["contact", "radar", "fusion"] and which_experiment in ["train_test", "kfold", "loo"]:
        try:
            for d in os.listdir("../model_checkpoints/" + which_data + "_" + which_experiment + "/"):
                os.remove("../model_checkpoints/" + which_data + "_" + which_experiment + "/" + d)
            for d in os.listdir("../model_training_logs/" + which_data + "_" + which_experiment + "/"):
                shutil.rmtree("../model_training_logs/" + which_data + "_" + which_experiment + "/" + d)
            for d in os.listdir("../results/" + which_data + "_" + which_experiment + "/"):
                os.remove("../results/" + which_data + "_" + which_experiment + "/" + d)
        except:
            print("Attempted to delete old DIRs, but failed.")
    else:
        print("Unknown DATA or EXPERIMENT.\n"
              "Possible values for data are 'contact', 'radar', 'fusion'.\n"
              "Possible experiments are 'train_test, 'kfold', 'loo'.")


# This function decreases the learning rate in accordance with some mathematical function.
def scheduler(epoch, lr=0.1, decay=0.001):
    if epoch <= 10:
        return lr
    else:
        return lr * 1 / (1 + decay * epoch)


# This function shows the distribution of (class) values of an array.
def show_dist(input_array):
    print(pd.DataFrame(input_array, columns=["class"]).value_counts(normalize=True))


# This function oversamples the data using Synthetic Minority Oversampling Technique, or SMOTE for short. It achieves equal class distribution in the dataset.
# It returns the oversampled dataset.
def oversample_smote(X_all, Y_all):
    oversample_X, oversample_Y = [], []
    for i in range(X_all.shape[2]):
        print("Oversampling signal", i)
        oversample = SMOTE(n_jobs=-1)
        X_t, Y_t = oversample.fit_resample(X_all[:, :, i], Y_all)
        oversample_X.append(X_t)
        oversample_Y.append(Y_t)

    X_all = np.stack(oversample_X, axis=2)
    Y_all = Y_t
    print("Shapes after oversampling:", X_all.shape, Y_all.shape)

    return X_all, Y_all


# The function below shuffles the data within the train and test split separately, as to not cause any neighbouring instances to appear in train and test.
# It returns the shuffled train and test data.
def shuffle_within_train_test(X_train, Y_train, X_test, Y_test):
    print("Shuffling WITHIN train/test, NOT overall!")

    indices_train = np.arange(Y_train.shape[0])
    np.random.shuffle(indices_train)
    X_train = X_train[indices_train]
    Y_train = Y_train[indices_train]

    indices_test = np.arange(Y_test.shape[0])
    np.random.shuffle(indices_test)
    X_test = X_test[indices_test]
    Y_test = Y_test[indices_test]

    return X_train, Y_train, X_test, Y_test


# Simple standard scaler (normalization) from sklearn.
def standardize_input(X_train, X_test):
    print("Scaling!")
    scaler = StandardScaler()
    X_train[np.isinf(X_train)] = np.mean(X_train)
    X_test[np.isinf(X_test)] = np.mean(X_test)
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)
    for i in range(X_train.shape[2]):
        scaler = scaler.fit(X_train[:, :, i])
        X_train[:, :, i] = scaler.transform(X_train[:, :, i])
        X_test[:, :, i] = scaler.transform(X_test[:, :, i])

    return X_train, X_test


# This function creates a selected deep learning ANN architecture model and returns it.
def get_model(input_X, exp_hyperparams, net_hyperparams, metrics):
    if exp_hyperparams["WHICH_MODEL"] == "fully_connected":
        model = models.fully_connected(
            input_data=input_X,
            exp_hyperparams=exp_hyperparams,
            net_hyperparams=net_hyperparams,
            metrics=metrics
        )
    elif exp_hyperparams["WHICH_MODEL"] == "1d_cnn":
        model = models.cnn1d(
            input_data=input_X,
            exp_hyperparams=exp_hyperparams,
            net_hyperparams=net_hyperparams,
            metrics=metrics
        )
    else:
        print("Unknown model type!")
        return None

    return model


# This function evaluates a given model on a given (separate) test dataset.
# It returns a table with results and saves a confusion matrix.
def evaluate_model(trained_model, X_test, Y_test, exp_hyperparams, class_map, iteration):
    # Table of numeric results
    all_results = pd.DataFrame(columns=["loss", "accuracy", "precision", "recall", "auc"])

    results = trained_model.evaluate(
        x=[np.squeeze(x) for x in np.split(X_test, X_test.shape[2], axis=2)],  # split individual signals
        y=to_categorical(Y_test, num_classes=exp_hyperparams["N_CLASSES"]),  # one hot encode discrete labels
        batch_size=exp_hyperparams["BATCH_SIZE"],
        verbose=exp_hyperparams["VERBOSE"]
    )
    all_results.loc[0] = results[1:]

    # Confusion matrix
    predictions = trained_model.predict([np.squeeze(x) for x in np.split(X_test, X_test.shape[2], axis=2)])
    Y_predicted = np.argmax(predictions, axis=1).astype('float')

    inv_class_map = {v: k for k, v in class_map.items()}
    if np.unique(Y_test).size >= np.unique(Y_predicted).size:
        larger = Y_test
    else:
        larger = Y_predicted
    plot_classes = []
    for val in np.unique(larger):
        plot_classes.append(inv_class_map[int(val)])

    # Plot confusion matrix
    cm = confusion_matrix(y_true=Y_test, y_pred=Y_predicted)
    df_cm = pd.DataFrame(cm / np.sum(cm), index=plot_classes, columns=plot_classes)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("PREDICTED")
    plt.ylabel("TRUE")

    if exp_hyperparams["WHICH_EXP"] == "kfold" or exp_hyperparams["WHICH_EXP"] == "loo":
        plt.savefig("../results/" + exp_hyperparams["WHICH_DATA"] + "_" + exp_hyperparams["WHICH_EXP"] + "/" + exp_hyperparams["RUN_ID"] + "/" + str(iteration) + "/confusion_matrix.png", bbox_inches='tight')
        plt.close()
    else:
        plt.savefig("../results/" + exp_hyperparams["WHICH_DATA"] + "_" + exp_hyperparams["WHICH_EXP"] + "/confusion_matrix.png", bbox_inches='tight')
        plt.close()

    return all_results


# The function below plots exact split of indices in the kFold CV, for sanity check.
def plot_cv_indices(cv, X, y, ax, n_splits, lw=10):
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices), c=indices, marker='_', lw=lw, cmap=plt.cm.coolwarm, vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X), c=y, marker='_', lw=lw, cmap=plt.cm.Paired)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class']
    ax.set(yticks=np.arange(n_splits + 1) + .5, yticklabels=yticklabels, xlabel='Sample index', ylabel="CV iteration", ylim=[n_splits + 2.2, -.2])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax
