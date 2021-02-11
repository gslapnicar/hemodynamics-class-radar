import tensorflow as tf
import time

import tensorflow as tf
from keras import optimizers, regularizers
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout, Dense
from keras.layers import Input, LSTM
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils import plot_model

print("Available GPU devices:", tf.config.list_physical_devices('GPU'))
# Reproducibility
tf.random.set_seed(3)


# Due to the fact that branched models cannot use the Keras functional API in traditional way (model.add()),
# we create functions that define each branch individually based on given hyperparameters.
def single_fully_conn_branch(input_shape, hyperparams):
    # Always create an input layer + 1 Dense layer + 1 Dropout for each branch
    my_input = Input(shape=(input_shape,))

    # Add hidden dense layers + dropout to each branch
    for hidden_layer in range(hyperparams["N_HIDDEN_LAYERS"]):
        if hidden_layer == 0:
            n_units = hyperparams["N_UNITS"] / 2  # triangular shape, but be careful -> units must not fall under 2
            l = Dense(units=n_units,
                      activation=hyperparams["ACTIVATION"],
                      kernel_initializer=hyperparams["INITIALIZER"],
                      activity_regularizer=hyperparams["REGULARIZER"])(my_input)
        else:
            n_units = n_units / 2 # triangular shape, but be careful -> units must not fall under 2
            l = Dense(units=n_units,
                      activation=hyperparams["ACTIVATION"],
                      kernel_initializer=hyperparams["INITIALIZER"],
                      activity_regularizer=hyperparams["REGULARIZER"])(l)
        l = Dropout(hyperparams["DROPOUT"])(l)

    final_branch_layer = Dense(units=n_units / 2,
                               activation=hyperparams["ACTIVATION"],
                               kernel_initializer=hyperparams["INITIALIZER"],
                               activity_regularizer=hyperparams["REGULARIZER"])(l)

    return my_input, final_branch_layer


# Using the branch function defined above, we create a simple fully connected ANN
def fully_connected(input_data, exp_hyperparams, net_hyperparams, metrics):
    input_shape = input_data.shape[1]
    n_branches = input_data.shape[2]
    n_classes = exp_hyperparams["N_CLASSES"]

    # Create each branch for each input
    branch_inputs = []
    branch_outputs = []
    for branch in range(n_branches):
        curr_input, curr_output = single_fully_conn_branch(input_shape, net_hyperparams)
        branch_inputs.append(curr_input)
        branch_outputs.append(curr_output)

    # Always merge the branches
    l = concatenate(branch_outputs)

    # Additional dense layers
    for hidden_layer in range(net_hyperparams["N_HIDDEN_LAYERS"]):
        if hidden_layer == 0:
            n_units = net_hyperparams["N_UNITS"] / 2  # triangular shape, but be careful -> units must not fall under 2
            l = Dense(units=n_units,  # triangular shape, but be careful -> units must not fall under 2
                      activation=net_hyperparams["ACTIVATION"],
                      kernel_initializer=net_hyperparams["INITIALIZER"],
                      activity_regularizer=net_hyperparams["REGULARIZER"])(l)
            l = Dropout(net_hyperparams["DROPOUT"])(l)
        else:
            n_units = n_units / 2  # triangular shape, but be careful -> units must not fall under 2
            l = Dense(units=n_units,  # triangular shape, but be careful -> units must not fall under 2
                      activation=net_hyperparams["ACTIVATION"],
                      kernel_initializer=net_hyperparams["INITIALIZER"],
                      activity_regularizer=net_hyperparams["REGULARIZER"])(l)
            l = Dropout(net_hyperparams["DROPOUT"])(l)

    # Always have output
    final_output = Dense(units=n_classes,
                         activation=net_hyperparams["LAST_ACTIVATION"],
                         kernel_initializer=net_hyperparams["INITIALIZER"],
                         activity_regularizer=net_hyperparams["REGULARIZER"])(l)

    # Define model
    model = Model(inputs=branch_inputs, outputs=final_output)

    model.compile(optimizer=net_hyperparams["OPTIMIZER"], loss=net_hyperparams["LOSS"], metrics=metrics)
    #model.summary()
    plot_model(model, to_file="../sanity_model_plots/" + exp_hyperparams["WHICH_DATA"] + "_" + exp_hyperparams["RUN_ID"] + ".png", show_layer_names=True, show_shapes=True)

    # history = model.fit(x=[np.squeeze(x_branch) for x_branch in np.split(x, x.shape[2], axis=2)],
    #                    y=to_categorical(y, num_classes=n_classes),
    #                    validation_data=([np.squeeze(x_branch) for x_branch in np.split(x_val, x_val.shape[2], axis=2)], to_categorical(y_val, num_classes=n_classes)),
    #                    batch_size=hyperparams["batch_size"],
    #                    epochs=hyperparams["epochs"],
    #                    # callbacks=[ta.Live()],
    #                    verbose=0)

    return model


# We use similar idea to the one above, creating a single CONVOLUTIONAL branch this time
def single_1d_cnn_branch(input_shape, hyperparams):
    # Always the same TinySleepNet idea for each branch
    my_input = Input(shape=(input_shape, 1,))

    # Representation learning (convolutional part)
    cnn_1 = Conv1D(filters=hyperparams["N_FILTERS"], kernel_size=hyperparams["Fs"] // 2, strides=hyperparams["Fs"] // 4, padding="same")(my_input)
    maxpool_1 = MaxPooling1D(pool_size=8, strides=8)(cnn_1)
    dropout_1 = Dropout(hyperparams["DROPOUT"])(maxpool_1)
    cnn_2 = Conv1D(filters=hyperparams["N_FILTERS"], kernel_size=hyperparams["KERNEL_SIZE"], strides=1, padding="same")(dropout_1)
    cnn_3 = Conv1D(filters=hyperparams["N_FILTERS"], kernel_size=hyperparams["KERNEL_SIZE"], strides=1, padding="same")(cnn_2)
    cnn_4 = Conv1D(filters=hyperparams["N_FILTERS"], kernel_size=hyperparams["KERNEL_SIZE"], strides=1, padding="same")(cnn_3)
    maxpool_2 = MaxPooling1D(pool_size=4, strides=4)(cnn_4)
    dropout_2 = Dropout(hyperparams["DROPOUT"])(maxpool_2)

    # Sequence learning (LSTM part)
    lstm_1 = LSTM(units=hyperparams["N_UNITS"])(dropout_2)
    final_branch_layer = Dropout(hyperparams["DROPOUT"])(lstm_1)

    # final_branch_layer = Dense(units=hyperparams["N_UNITS"], activation=hyperparams["ACTIVATION"])(dropout_2)

    return my_input, final_branch_layer


# Now we again use the branch function to create a network
def cnn1d(input_data, exp_hyperparams, net_hyperparams, metrics):
    input_shape = input_data.shape[1]
    n_branches = input_data.shape[2]
    n_classes = exp_hyperparams["N_CLASSES"]

    # Create each branch for each input
    branch_inputs = []
    branch_outputs = []
    for branch in range(n_branches):
        curr_input, curr_output = single_1d_cnn_branch(input_shape, net_hyperparams)
        branch_inputs.append(curr_input)
        branch_outputs.append(curr_output)

    # Always merge the branches
    l = concatenate(branch_outputs)
    # l = Flatten()(l) # this needed if no LSTM in the branch function

    # Additional dense layers on top of merged branches potentially
    # for dense_layer in range(hyperparams["N_HIDDEN_LAYERS"]):
    #    l = Dense(units=hyperparams["N_UNITS"]//2, activation=hyperparams["ACTIVATION"])(l)
    #    l = Dropout(hyperparams["DROPOUT"])(l)

    # Always have output
    final_output = Dense(units=n_classes, activation=net_hyperparams["LAST_ACTIVATION"])(l)

    # Define model
    model = Model(inputs=branch_inputs, outputs=final_output)

    opt = optimizers.Adam(learning_rate=net_hyperparams["LR"])
    model.compile(optimizer=opt, loss=net_hyperparams["LOSS"], metrics=metrics)
    model.summary()
    plot_model(model, to_file="../sanity_model_plots/" + exp_hyperparams["WHICH_DATA"] + "_" + exp_hyperparams["RUN_ID"] + ".png", show_layer_names=True, show_shapes=True)

    # history = model.fit(x=[np.squeeze(x_branch) for x_branch in np.split(x, x.shape[2], axis=2)],
    #                    y=to_categorical(y, num_classes=n_classes),
    #                    validation_split=hyperparams["validation_pct"],
    #                    batch_size=hyperparams["batch_size"],
    #                    epochs=hyperparams["epochs"],
    #                    #callbacks=[ta.Live()],
    #                    verbose=1)

    return model
