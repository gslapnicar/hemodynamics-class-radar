import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

import helper_functions


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels, which_data, batch_size, dim, n_channels=6, n_classes=6, shuffle=False):
        'Initialization'
        self.which_data = which_data
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, :, :, :] = np.load("model_inputs_and_targets/spectrograms/" + self.which_data + "/" + ID)

            # Store class
            y[i] = self.labels[indexes[i]]

        return ([x for x in np.split(X, X.shape[3], axis=3)], keras.utils.to_categorical(y, num_classes=self.n_classes))


def spectrograms_experiment(which_data, which_model, hyperparams, metrics, class_map, n_folds, selected_classes=None):
    # Clean old things from previous run (in newer versions, dont run this, since each run is saved by timestamp)
    # helper_functions.clear_old_dirs(which_data, "spectrograms")

    # Read full data
    base_path = "model_inputs_and_targets/spectrograms/"
    X_dict, Y_dict = {}, {}
    for class_name in os.listdir(base_path + which_data):
        X_dict[class_name] = np.array([class_name + "/" + file for file in os.listdir(base_path + which_data + "/" + class_name)])
        Y_dict[class_name] = np.array([class_map[class_name] for file in os.listdir(base_path + which_data + "/" + class_name)])

    # Prepare kFold CV
    X_full = np.concatenate([x for x in X_dict.values()])
    Y_full = np.concatenate([y for y in Y_dict.values()])
    skf = StratifiedKFold(n_splits=n_folds)
    print("Number of splits:", skf.get_n_splits(X_full), skf.get_n_splits(Y_full), "\n")

    iteration = 0
    all_results = pd.DataFrame(columns=["loss", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "auc"])
    for train_idx, test_idx in skf.split(X_full, Y_full):
        print("========== Iteration:", iteration, "==========")
        X_train_files = X_full[train_idx[0:int(0.7 * train_idx.shape[0])]]
        X_val_files = X_full[train_idx[int(0.7 * train_idx.shape[0]):]]
        X_test_files = X_full[test_idx]
        Y_train = Y_full[train_idx[0:int(0.7 * train_idx.shape[0])]]
        Y_val = Y_full[train_idx[int(0.7 * train_idx.shape[0]):]]
        Y_test = Y_full[test_idx]

        print(X_test_files)
        print(Y_test)

        # Make a model
        example_instance = np.load(base_path + which_data + "/" + X_train_files[0])
        model = models.create_2d_CNN_small(example_instance, kernel_size=(3, 3), n_classes=6)

        # Generator
        train_gen = DataGenerator(X_train_files, Y_train, which_data, hyperparams["BATCH_SIZE"], dim=example_instance.shape)
        val_gen = DataGenerator(X_val_files, Y_val, which_data, hyperparams["BATCH_SIZE"], dim=example_instance.shape)
        test_gen = DataGenerator(X_test_files, Y_test, which_data, hyperparams["BATCH_SIZE"], dim=example_instance.shape)

        model.fit(x=train_gen,
                  validation_data=val_gen,
                  use_multiprocessing=True,
                  workers=8,
                  epochs=hyperparams["N_EPOCHS"],
                  verbose=hyperparams["VERBOSE"],
                  callbacks=[
                      tf.keras.callbacks.ModelCheckpoint(filepath="model_checkpoints/" + which_data + "_spectrograms/best_model.hdf5", monitor='val_loss', mode='min', save_best_only=True),
                      tf.keras.callbacks.TensorBoard(log_dir="model_training_logs/" + which_data + "_spectrograms/"),
                      tf.keras.callbacks.LearningRateScheduler(helper_functions.scheduler),
                      tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
                  ])

        iteration += 1

    return all_results
