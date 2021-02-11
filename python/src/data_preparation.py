import numpy as np
import math
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from scipy.signal import butter, lfilter


def remap_ground_truth(class_array_raw, class_idx, scenario, subject_id, class_map, plot=False):
    class_array_new = np.zeros(class_array_raw.shape)

    if len(class_idx) % 2 != 0:
        print("[WARNING!] Odd number of indices!")
    else:
        if scenario == "Resting":
            class_array_new[:] = class_map[scenario]
        elif scenario == "Valsalva" or scenario == "Apnea":
            for i1, i2 in zip(class_idx[::2], class_idx[1::2]):
                class_array_new[i1:i2] = class_map[scenario]
        elif scenario == "TiltUp" or scenario == "TiltDown":
            if class_idx.size == 0:
                class_array_new[:] = class_map[scenario]
            elif class_idx.size == 1:
                class_array_new[class_idx[0]:] = class_map[scenario]
            elif class_idx.size == 2:
                if class_idx[0] == 0 and class_idx[1] == len(class_array_raw):
                    class_array_new[:] = class_map[scenario]
                elif class_idx[0] != 0 and class_idx[1] == len(class_array_raw):
                    class_array_new[class_idx[0]:] = class_map[scenario]
                else:
                    class_array_new[class_idx[1]:] = class_map[scenario]

    if plot:
        if not os.path.exists("../sanity_signal_plots/" + subject_id + "/" + scenario):
            os.makedirs("../sanity_signal_plots/" + subject_id + "/" + scenario)

        # print(class_idx)
        fig = plt.figure(figsize=(16, 12))
        l1, = plt.plot(class_array_raw.tolist())
        l2 = plt.vlines(class_idx, 0, 5, colors='r')
        l3, = plt.plot(class_array_new.tolist(), 'k')
        plt.title(subject_id + " " + scenario + " - class value")
        plt.ylim([-1, 6])
        plt.legend([l1, l2, l3], ['Electrical signal', 'Button presses', 'New label'])
        plt.savefig("../sanity_signal_plots/" + subject_id + "/" + scenario + "/class.png")
        plt.close(fig)

    return class_array_new


# This function below reshapes the time series into a 2D matrix:
# - **rows**: each row is one window or instance for ML, which is of selected *window length*
# - **columns**: each column is a single sample of the signal
#
# Overlap is chosen, currently at 1/2 of the window length.
def get_instances_sliding_window(array, window_width, overlap):
    return skimage.util.view_as_windows(array, window_width, step=overlap)


# Bandpass filtering
def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_band(signal_name):
    # Set filtering band based on type of signal
    if signal_name == "contact_bp":
        band = [0.1, 4.0]  # bp
    elif signal_name == "contact_ecg1":
        band = [0.1, 20.0]  # ecg1
    elif signal_name == "contact_ecg2":
        band = [0.1, 20.0]  # ecg2
    elif signal_name == "contact_icg":
        band = [0.1, 10.0]  # icg
    elif signal_name == "contact_respiration":
        band = [0.1, 1.0]  # respiration
    elif signal_name == "contact_z0":
        band = [0.1, 1.0]  # z0, related to respiration
    elif signal_name == "radar_dist":
        band = [0.1, 1.0]  # distance
    elif signal_name == "radar_heartSound":
        band = [0.1, 10.0]  # heart sound
    elif signal_name == "radar_i":
        band = [0.1, 5.0]  # i
    elif signal_name == "radar_pulse":
        band = [0.1, 3.0]  # pulse
    elif signal_name == "radar_q":
        band = [0.1, 5.0]  # q
    elif signal_name == "radar_resp":
        band = [0.1, 1.0]  # respiration

    return band


# The function below does the majority of work when going from many long 1D signals to getting short windows formatted in 3D arrays.
#
# Input: It takes the prepared data from the MATLAB scripts (a single dict, where 1st level are subjects, 2nd level are scenarios and 3rd level are obtained signals as 1D time series).
#
# Functionality: Based on the button presses it remaps the scenarios to discrete integer values. Then the windowing is done and the most common label in a window is selected as the singular label. Remapped label graphs are saved.
# Additionally, the spectrograms are computed and saved.
#
# Output: The result are dicts (for contact and contact-free signals), where 1st level keys are subject IDs and values are 3D arrays (rows = instances / windows, columns = signal samples, 3rd dimension = different signals).
# Additionally, a dict of ground-truth scenario labels is also provided, where keys are subject IDs and values are 1D arrays (temporal label).
def prepare_data(file_path, fs, win_width, class_map, signal_name_map, normalise, plot):
    X_contact_3d, X_radar_3d, Y_3d, plot_labels_contact, plot_labels_radar = {}, {}, {}, {}, {}

    # read the original data
    f = h5py.File(file_path, 'r')
    full_data = f.get('output_re')

    # iterate across subjects
    for (subject_id, scenarios) in full_data.items():
        #print("=", subject_id)

        # iterate across scenarios for each subject
        scenarios_contact_sigs, scenarios_radar_sigs, scenarios_labels, scenarios_contact_signal_names, scenarios_radar_signal_names = [], [], [], [], []
        for scenario, signals in scenarios.items():
            #print("===", scenario)

            # remap scenario "signal" into descrete class values based on button presses
            class_array_raw = signals.get("class")[()].flatten()
            class_idx = signals.get("class_idx")[()].flatten(order='F').astype(int)
            class_array_discrete = remap_ground_truth(class_array_raw, class_idx, scenario, subject_id, class_map, plot=plot)

            # then cut it into sliding windows with half a window overlap
            class_matrix_windowed = get_instances_sliding_window(
                class_array_discrete, win_width * fs, math.floor((win_width * fs) / 2)
            )

            scenarios_labels.append(pd.DataFrame(class_matrix_windowed).mode(axis=1).values[:, 0])

            # iterate across all signals for each scenario for each subject
            contact_signals, radar_signals, signal_contact_names, signal_radar_names = [], [], [], []
            for signal_name, data in signals.items():
                if signal_name not in ["class", "class_idx"]:
                    if normalise:
                        signal = (data[()].flatten() - np.min(data[()].flatten())) / np.ptp(data[()].flatten())
                        band = get_band(signal_name)
                        signal = butter_bandpass_filter(signal, band[0], band[1], fs)
                    else:
                        signal = data[()].flatten()
                        band = get_band(signal_name)
                        signal = butter_bandpass_filter(signal, band[0], band[1], fs)

                    # cut it into sliding windows with half a window overlap
                    signal_matrix_windowed = get_instances_sliding_window(
                        signal, win_width * fs, math.floor((win_width * fs) / 2)
                    )

                    if "contact" in signal_name:
                        contact_signals.append(signal_matrix_windowed)
                        signal_contact_names.append(np.full(signal_matrix_windowed.shape, signal_name_map[signal_name]))
                    elif "radar" in signal_name:
                        radar_signals.append(signal_matrix_windowed)
                        signal_radar_names.append(np.full(signal_matrix_windowed.shape, signal_name_map[signal_name]))
                    else:
                        print("Anomaly!?")

            scenarios_contact_sigs.append(np.stack(contact_signals, axis=2))
            scenarios_radar_sigs.append(np.stack(radar_signals, axis=2))
            scenarios_contact_signal_names.append(np.stack(signal_contact_names, axis=2))
            scenarios_radar_signal_names.append(np.stack(signal_radar_names, axis=2))

        # Save the concatenated arrays to corresponding subject IDs
        X_contact_3d[subject_id] = np.concatenate(scenarios_contact_sigs)
        X_radar_3d[subject_id] = np.concatenate(scenarios_radar_sigs)
        Y_3d[subject_id] = np.concatenate(scenarios_labels)
        plot_labels_contact[subject_id] = np.concatenate(scenarios_contact_signal_names)
        plot_labels_radar[subject_id] = np.concatenate(scenarios_radar_signal_names)
        #print()

    return X_contact_3d, X_radar_3d, Y_3d, plot_labels_contact, plot_labels_radar


# Check some output shapes, if they match what is expected, if each dimension is of expected shape, some visual inspection, etc.
def sanity_check_signal(subject_id, random_segment, signal_X_3d, signal_Y_3d, signal_names, name_map, plot=False):
    #print("We have X data for these\n", signal_X_3d.keys(), "\n")
    #print("We have Y data for these\n", signal_Y_3d.keys(), "\n")

    #print("Shape of X data for selected subject", subject_id, signal_X_3d[subject_id].shape)
    #print("Shape of Y data for selected subject", subject_id, signal_Y_3d[subject_id].shape, "\n")

    if plot:
        for i in range(signal_names[subject_id].shape[2]):
            plt.figure()
            plt.plot(signal_X_3d[subject_id][random_segment, :, i])
            plt.title(subject_id + " " + name_map[signal_names[subject_id][random_segment, :, i][0]])
            plt.show()


# This function will output which subjects are being traveresed, and which scenarios of each subject exists in the original data.
# Result: We expect to obtain 3 dictionaries, one for contact, second for contact-free and third for the ground-truth labels.
# Additionally, some extra dicts (for plotting purposes) are provided.
def prepare_and_save(filepath, fs, win_width, class_map, signal_name_map, normalise, plot):
    X_contact_3d, X_radar_3d, Y_3d, plot_labels_contact, plot_labels_radar = prepare_data(
        file_path=filepath,
        fs=fs,
        win_width=win_width,
        class_map=class_map,
        signal_name_map=signal_name_map,
        normalise=normalise,
        plot=plot
    )

    # Take a random subject and random instance and plot the signals
    random_id = random.randint(1, 30)
    if random_id <= 9:
        SUBJECT_ID = "GDN000" + str(random_id)
    else:
        SUBJECT_ID = "GDN00" + str(random_id)

    random_segment = random.randint(0, X_contact_3d[SUBJECT_ID].shape[0])
    inv_name_map = {v: k for k, v in signal_name_map.items()}

    #print("============================ CONTACT: ============================\n")
    #sanity_check_signal(SUBJECT_ID, random_segment, X_contact_3d, Y_3d, plot_labels_contact, inv_name_map, plot)
    #print("============================ RADAR: ============================\n")
    #sanity_check_signal(SUBJECT_ID, random_segment, X_radar_3d, Y_3d, plot_labels_radar, inv_name_map, plot)

    # Distributions (and shape checks) of per-subject data
    # for subject1, subject2, subject3 in zip(X_contact_3d, X_radar_3d, Y_3d):
    #    print(subject1, subject2, subject3)
    #    print(X_contact_3d[subject1].shape, X_radar_3d[subject1].shape, Y_3d[subject1].shape)
    #    print(pd.DataFrame(Y_3d[subject1], columns=["class"]).value_counts())

    # Check full data as later formatted for DL
    X_all_contact = np.concatenate(list(X_contact_3d.values()), axis=0)
    X_all_radar = np.concatenate(list(X_radar_3d.values()), axis=0)
    Y_all = np.concatenate(list(Y_3d.values()), axis=0)
    plot_titles_contact = np.concatenate(list(plot_labels_contact.values()), axis=0)
    plot_titles_radar = np.concatenate(list(plot_labels_radar.values()), axis=0)

    # Show the full distribution of classes
    #print("============================ CLASS DISTRIBUTION: ============================\n")
    inv_class_map = {v: k for k, v in class_map.items()}
    x_ticks = [int(x[0]) for x in pd.DataFrame(Y_all, columns=["class"]).value_counts().keys().to_list()]
    x_labels = [inv_class_map[x] for x in x_ticks]

    dist = pd.DataFrame(Y_all, columns=["class"]).value_counts().rename(index=inv_class_map)
    dist = dist.reset_index()
    dist.columns = ["class", "counts"]
    dist[["int_class"]] = x_ticks
    dist[["percentage"]] = round(dist.counts / dist.counts.sum(), 2) * 100
    #print(dist)

    # Save the concatenated files for train/test split and kFold
    np.save("../model_inputs_and_targets/train_test/X_all_contact.npy", X_all_contact)
    np.save("../model_inputs_and_targets/train_test/X_all_radar.npy", X_all_radar)
    np.save("../model_inputs_and_targets/train_test/Y_all.npy", Y_all)

    np.save("../model_inputs_and_targets/kfold/X_all_contact.npy", X_all_contact)
    np.save("../model_inputs_and_targets/kfold/X_all_radar.npy", X_all_radar)
    np.save("../model_inputs_and_targets/kfold/Y_all.npy", Y_all)

    # Save the per-subject dictionary files for leave one out split
    np.save("../model_inputs_and_targets/leave_one_out/X_loo_contact.npy", X_contact_3d)
    np.save("../model_inputs_and_targets/leave_one_out/X_loo_radar.npy", X_radar_3d)
    np.save("../model_inputs_and_targets/leave_one_out/Y_loo.npy", Y_3d)
