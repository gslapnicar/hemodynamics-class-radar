%% Author: Gasper Slapnicar
% Copyright (C) 2020  Gasper Slapnicar

%%%%%% IMPORTANT %%%%%%
% This is a modified script of the original code provided as supplementary
% material of the paper "A dataset of clinically recorded radar vital signs 
% with synchronised reference sensor signals" by Sven Schellenberger and 
% Kilin Shi. It relies on auxiliary functions also provided by Sven Schellenberger and 
% Kilin Shi. Original code available at: https://gitlab.com/sven_schellenberger/scidata_phase1
%%%%%%%%%%%%%%%%%%%%%%%

%% Plot data
% This script generates exemplary plots of selected subject-ID(s) and
% scenario(s)
% Data is loaded, processed and then plotted

%% Init
clear;
clc;
close all;
addpath(genpath('utils/'))
format long g;

%% Options
%%%% Choose Subject-ID(s) - can be singular integer (from 01 to 30) or a
%%%% range like 3:25
IDrange             = 1:30;
cut                 = 0; % parameter for cutting out only periods between button presses (1 = cut out, 0 = take all data including movement)
fs_target_resample  = 100;
butterworth_filt_order = 4;
radar_cutoffs       = [0.01, 20.0];
bp_cutoffs          = [0.5, 4.0];
ecg_cutoffs         = [0.1, 20.0];
icg_cutoffs         = [0.1, 20.0];
resp_cutoffs        = [0.1, 1.0];
z0_cutoffs          = [0.1, 1.0];

%%%% Choose scnerio(s) 
% possible scenarios are {'Resting' 'Valsalva' 'Apnea' 'TiltUp' 'TiltDown'}
scenarios = {'Resting', 'Valsalva', 'Apnea', 'TiltUp', 'TiltDown'};

%%%% Set path to data
path = '../../data'; 

%% Extract and plot the data
% Manually selected ECG channel for each subject (Position in ECG_CHANNEL vector corresponds to ID)
ECG_CHANNEL = [2 2 2 2 2 1 2 2 2 2 2 2 2 2 1 2 2 2 2 2 1 1 2 2 2 2 2 2 2 2];

for indx = 1:length(IDrange)
    %% Iterate all subject IDs
    ID = sprintf('GDN%04d',IDrange(indx));
    fprintf('----------- Loading %s ------------\n', ID);

    for sz = 1:length(scenarios)
        %% Iterate all existing subject scenarios
        scenario = scenarios{sz};
        fprintf('--- Scenario %s\n', scenario);
        
        % Check if scenario exists for this subject
        path_id = [path,'/',ID];
        files_synced_mat = dir([path_id,'/*.mat']);
        found = [];
        for j = 1:length(files_synced_mat)
            found = strfind(files_synced_mat(j).name,scenario);
            if ~isempty(found)
                % load file
                load([path_id,'/',files_synced_mat(j).name]);
                break;
            end
        end
        if isempty(found)
            fprintf('---- skipped, because this scenario not found\n');
            continue
        end
        
        %% Basic data cleaning
        output_raw.(ID).(scenario) = struct;
        output_f.(ID).(scenario) = struct;
        output_re.(ID).(scenario) = struct;
        
        % Correction of signal potentially being NaN sometimes
        if sum(isnan(radar_i(:))) > 0
            radar_i = fillmissing(radar_i,'pchip');
        end
        if sum(isnan(radar_q(:))) > 0
            radar_q = fillmissing(radar_q,'pchip');
        end
        if sum(isnan(tfm_bp(:))) > 0
            tfm_bp = fillmissing(tfm_bp,'pchip');
        end
        if sum(isnan(tfm_ecg1(:))) > 0
            tfm_ecg1 = fillmissing(tfm_ecg1,'pchip');
        end
        if sum(isnan(tfm_ecg2(:))) > 0
            tfm_ecg2 = fillmissing(tfm_ecg2,'pchip');
        end
        if sum(isnan(tfm_icg(:))) > 0
            tfm_icg = fillmissing(tfm_icg,'pchip');
        end
        if sum(isnan(tfm_z0(:))) > 0
            tfm_z0 = fillmissing(tfm_z0,'pchip');
        end
        if sum(isnan(tfm_intervention(:))) > 0
            tfm_intervention = fillmissing(tfm_intervention,'pchip');
        end
        
        % ECG channel selection
        if ECG_CHANNEL(IDrange(indx)) == 1
            tfm_ecg = tfm_ecg1;
        else
            tfm_ecg = tfm_ecg2;
        end
        
        %% Compute distance from raw radar data
        % This ellipse fitting and compensation is not trivial, details are here: 
        % https://www.nature.com/articles/s41598-018-29984-5
        % NOTE: source code is not accessable!
        [~,~,~,radar_dist] = elreko(radar_i,radar_q,measurement_info{1},0);
        % Usage of elreko:
        % [radar_i_compensated,radar_q_compensated,phase_compensated,radar_dist] = elreko(radar_i,radar_q,measurement_info{1}(timestamp of dataset),0(Plot flag -> 1: plots on, 0: plots off));
        
        %% Compute respiration & heartsound signal from raw radar data
        % NOTE: source code is not accessable!
        [radar_respiration, radar_pulse, radar_heartsound, tfm_respiration] = getVitalSigns(radar_dist, fs_radar, tfm_z0, fs_z0);
        
        %% Get indices of button presses - explanations in GetInterventions()
        sc_sec = getInterventions(scenario,tfm_intervention,fs_intervention);
        sc_new = round(sc_sec ./ (fs_intervention / fs_target_resample));
        
        %% [ML] Preprocessing
        % 1.) Filter the data to remove some obvious noise
        
        % contact-free data
        radar_i_f           = filtButter(radar_i,fs_radar,butterworth_filt_order,radar_cutoffs,'bandpass');
        radar_q_f           = filtButter(radar_q,fs_radar,butterworth_filt_order,radar_cutoffs,'bandpass');
        radar_dist_f        = filtButter(radar_dist,fs_radar,butterworth_filt_order,radar_cutoffs,'bandpass');
        radar_resp_f        = filtButter(radar_respiration,fs_radar,butterworth_filt_order,radar_cutoffs,'bandpass');
        radar_pulse_f       = filtButter(radar_pulse,fs_radar,butterworth_filt_order,radar_cutoffs,'bandpass');
        radar_heartsound_f  = filtButter(radar_heartsound,fs_radar,butterworth_filt_order,radar_cutoffs,'bandpass');
        % contact data
        % dont filter BP atm, because this implementation influences ampl.
        %tfm_bp_f = filtButter(tfm_bp,fs_bp,butterworth_filt_order,bp_cutoffs,'bandpass');
        tfm_ecg1_f          = filtButter(tfm_ecg1,fs_ecg,butterworth_filt_order,ecg_cutoffs,'bandpass');
        tfm_ecg2_f          = filtButter(tfm_ecg2,fs_ecg,butterworth_filt_order,ecg_cutoffs,'bandpass');
        tfm_icg_f           = filtButter(tfm_icg,fs_icg,butterworth_filt_order,icg_cutoffs,'bandpass');
        tfm_respiration_f   = filtButter(tfm_respiration,fs_z0,butterworth_filt_order,resp_cutoffs,'bandpass');
        tfm_z0_f            = filtButter(tfm_z0,fs_z0,butterworth_filt_order,z0_cutoffs,'bandpass');
        
        % Sanity check plots after resampling
        %{
        t_radar = 1/fs_radar:1/fs_radar:length(radar_dist)/fs_radar;
        
        figure;
        ax(1) = subplot(6,1,1);
        hold on;
        plot(t_radar, radar_i, 'k--');
        plot(t_radar, radar_i_f);
        title('Radar I')
        ax(2) = subplot(6,1,2);
        hold on;
        plot(t_radar, radar_q, 'k--');
        plot(t_radar, radar_q_f);
        title('Radar Q')
        ax(3) = subplot(6,1,3);
        hold on;
        plot(t_radar, radar_dist.*1000, 'k--');
        plot(t_radar, radar_dist_f.*1000);
        title('Radar distance')
        ax(4) = subplot(6,1,4);
        hold on;
        plot(t_radar, radar_respiration, 'k--');
        plot(t_radar, radar_resp_f);
        title('Radar respiration')
        ax(5) = subplot(6,1,5);
        hold on;
        plot(t_radar, radar_pulse, 'k--');
        plot(t_radar, radar_pulse_f);
        title('Radar pulse')
        ax(6) = subplot(6,1,6);
        hold on;
        plot(t_radar, radar_heartsound, 'k--');
        plot(t_radar, radar_heartsound_f);
        title('Radar heart sound')
        linkaxes(ax, 'x');
        
        figure;
        ax(1) = subplot(6,1,1);
        hold on;
        plot(1/fs_bp:1/fs_bp:length(tfm_bp)/fs_bp,tfm_bp, 'k--');
        plot(1/fs_bp:1/fs_bp:length(tfm_bp)/fs_bp,tfm_bp);
        title('Blood Pressure (we dont filter BP atm due to amplitude influence)')
        ax(2) = subplot(6,1,2);
        hold on;
        plot(1/fs_ecg:1/fs_ecg:length(tfm_ecg1)/fs_ecg,tfm_ecg1, 'k--');
        plot(1/fs_ecg:1/fs_ecg:length(tfm_ecg1)/fs_ecg,tfm_ecg1_f);
        title('ECG1')
        ax(3) = subplot(6,1,3);
        hold on;
        plot(1/fs_ecg:1/fs_ecg:length(tfm_ecg2)/fs_ecg,tfm_ecg2, 'k--');
        plot(1/fs_ecg:1/fs_ecg:length(tfm_ecg2)/fs_ecg,tfm_ecg2_f);
        title('ECG2')
        ax(4) = subplot(6,1,4);
        hold on;
        plot(1/fs_icg:1/fs_icg:length(tfm_icg)/fs_icg,tfm_icg, 'k--');
        plot(1/fs_icg:1/fs_icg:length(tfm_icg)/fs_icg,tfm_icg_f);
        title('ICG')
        ax(5) = subplot(6,1,5);
        hold on;
        plot(1/fs_z0:1/fs_z0:length(tfm_respiration)/fs_z0,tfm_respiration, 'k--');
        plot(1/fs_z0:1/fs_z0:length(tfm_respiration)/fs_z0,tfm_respiration_f);
        title('Respiration')
        ax(6) = subplot(6,1,6);
        hold on;
        plot(1/fs_z0:1/fs_z0:length(tfm_z0)/fs_z0,tfm_z0, 'k--');
        plot(1/fs_z0:1/fs_z0:length(tfm_z0)/fs_z0,tfm_z0_f);
        title('Z0')
        linkaxes(ax, 'x');
        return;
        %}
        
        % 2.) Resample all to preserve space... consider other resampling
        % methods, for higher precision? Currently using MATLAB default
        % which is: "resample applies an FIR Antialiasing Lowpass Filter to
        % x and compensates for the delay introduced by the filter"
        
        % contact-free data
        radar_i_re          = resample(radar_i_f, fs_target_resample, fs_radar);
        radar_q_re          = resample(radar_q_f, fs_target_resample, fs_radar);
        radar_dist_re       = resample(radar_dist_f, fs_target_resample, fs_radar);
        radar_resp_re       = resample(radar_resp_f, fs_target_resample, fs_radar);
        radar_pulse_re      = resample(radar_pulse_f, fs_target_resample, fs_radar);
        radar_heartsound_re = resample(radar_heartsound_f, fs_target_resample, fs_radar);
        time_radar_re       = 1/fs_target_resample:1/fs_target_resample:length(radar_dist_re)/fs_target_resample;    
        % contact data
        tfm_bp_re           = resample(tfm_bp, fs_target_resample, fs_bp);
        tfm_ecg1_re         = resample(tfm_ecg1_f, fs_target_resample, fs_ecg);
        tfm_ecg2_re         = resample(tfm_ecg2_f, fs_target_resample, fs_ecg);
        tfm_icg_re          = resample(tfm_icg_f, fs_target_resample, fs_icg);
        tfm_respiration_re  = resample(tfm_respiration_f, fs_target_resample, fs_z0);
        tfm_z0_re           = resample(tfm_z0_f, fs_target_resample, fs_z0);
        tfm_intervention_re = resample(tfm_intervention, fs_target_resample, fs_intervention);
        time_contact_re     = 1/fs_target_resample:1/fs_target_resample:length(tfm_respiration_re)/fs_target_resample;
        
        % Sanity check plots after resampling
        %{
        t_radar = 1/fs_radar:1/fs_radar:length(radar_dist)/fs_radar
        
        figure;
        ax(1) = subplot(6,1,1);
        hold on;
        plot(t_radar, 'k--');
        plot(time_radar_re,radar_i_re);
        title('Radar I')
        ax(2) = subplot(6,1,2);
        hold on;
        plot(t_radar,radar_q, 'k--');
        plot(time_radar_re,radar_q_re);
        title('Radar Q')
        ax(3) = subplot(6,1,3);
        hold on;
        plot(t_radar,radar_dist.*1000, 'k--');
        plot(time_radar_re,radar_dist_re.*1000);
        title('Radar distance')
        ax(4) = subplot(6,1,4);
        hold on;
        plot(t_radar,radar_respiration, 'k--');
        plot(time_radar_re,radar_resp_re);
        title('Radar respiration')
        ax(5) = subplot(6,1,5);
        hold on;
        plot(t_radar,radar_pulse, 'k--');
        plot(time_radar_re,radar_pulse_re);
        title('Radar pulse')
        ax(6) = subplot(6,1,6);
        hold on;
        plot(t_radar,radar_heartsound, 'k--');
        plot(time_radar_re,radar_heartsound_re);
        title('Radar heart sound')
        linkaxes(ax, 'x');
        
        figure;
        ax(1) = subplot(6,1,1);
        hold on;
        plot(1/fs_bp:1/fs_bp:length(tfm_bp)/fs_bp,tfm_bp, 'k--');
        plot(time_contact_re,contact_bp_re);
        title('Blood Pressure')
        ax(2) = subplot(6,1,2);
        hold on;
        plot(1/fs_ecg:1/fs_ecg:length(tfm_ecg1)/fs_ecg,tfm_ecg1, 'k--');
        plot(time_contact_re,contact_ecg1_re);
        title('ECG1')
        ax(3) = subplot(6,1,3);
        hold on;
        plot(1/fs_ecg:1/fs_ecg:length(tfm_ecg2)/fs_ecg,tfm_ecg2, 'k--');
        plot(time_contact_re,contact_ecg2_re);
        title('ECG2')
        ax(4) = subplot(6,1,4);
        hold on;
        plot(1/fs_icg:1/fs_icg:length(tfm_icg)/fs_icg,tfm_icg, 'k--');
        plot(time_contact_re,contact_icg_re);
        title('ICG')
        ax(5) = subplot(6,1,5);
        hold on;
        plot(1/fs_z0:1/fs_z0:length(tfm_respiration)/fs_z0,tfm_respiration, 'k--');
        plot(time_contact_re,contact_respiration_re);
        title('Respiration')
        ax(6) = subplot(6,1,6);
        hold on;
        plot(1/fs_z0:1/fs_z0:length(tfm_z0)/fs_z0,tfm_z0, 'k--');
        plot(time_contact_re,contact_z0_re);
        title('Z0')
        linkaxes(ax, 'x');
        return;
        %}
        %figure;
        %plot(tfm_intervention_re);
        %hold on;
        %for g=1:numel(sc_new)
        %    xline(sc_new(g));
        %end
        %title(scenario);
        %disp(sc_new)
        
        %% Detect heart beats from both radar and ECG
        % Heartbeat detection via hidden semi-Markov models (we get R peak
        % locations)
        % NOTE: source code is not accessable!
        radar_hs_states = getHsmmStates(radar_heartsound, fs_radar);
        radar_hs_locsR = statesToLocs(radar_hs_states, 1);
        [~,tfm_ecg_locsR] = twaveend(tfm_ecg(1:length(radar_hs_states)),fs_ecg,32*(fs_ecg/200),'p');
        
        %% Save chosen (relevant) data into a single variable
        %% [Raw] gather contact-free data
        output_raw.(ID).(scenario).radar_i          = radar_i;
        output_raw.(ID).(scenario).radar_q          = radar_q;
        output_raw.(ID).(scenario).radar_dist       = radar_dist;
        output_raw.(ID).(scenario).radar_resp       = radar_respiration;
        output_raw.(ID).(scenario).radar_pulse      = radar_pulse;
        output_raw.(ID).(scenario).radar_heartSound = radar_heartsound;
        % [Raw] gather contact data
        output_raw.(ID).(scenario).contact_bp       = tfm_bp;
        output_raw.(ID).(scenario).contact_ecg1     = tfm_ecg1;
        output_raw.(ID).(scenario).contact_ecg2     = tfm_ecg2;
        output_raw.(ID).(scenario).contact_icg      = tfm_icg;
        output_raw.(ID).(scenario).contact_respiration = tfm_respiration;
        output_raw.(ID).(scenario).contact_z0       = tfm_z0;
        output_raw.(ID).(scenario).class            = tfm_intervention;
        output_raw.(ID).(scenario).class_idx        = sc_sec;
        
        %% [Filtered] gather contact-free data
        output_f.(ID).(scenario).radar_i            = radar_i_f;
        output_f.(ID).(scenario).radar_q            = radar_q_f;
        output_f.(ID).(scenario).radar_dist         = radar_dist_f;
        output_f.(ID).(scenario).radar_resp         = radar_resp_f;
        output_f.(ID).(scenario).radar_pulse        = radar_pulse_f;
        output_f.(ID).(scenario).radar_heartSound   = radar_heartsound_f;
        % [Filtered] gather contact data
        output_f.(ID).(scenario).contact_bp         = tfm_bp;
        output_f.(ID).(scenario).contact_ecg1       = tfm_ecg1_f;
        output_f.(ID).(scenario).contact_ecg2       = tfm_ecg2_f;
        output_f.(ID).(scenario).contact_icg        = tfm_icg_f;
        output_f.(ID).(scenario).contact_respiration = tfm_respiration_f;
        output_f.(ID).(scenario).contact_z0         = tfm_z0_f;
        output_f.(ID).(scenario).class              = tfm_intervention;
        output_f.(ID).(scenario).class_idx          = sc_sec;
        
        %% [Resampled] gather contact-free data
        output_re.(ID).(scenario).radar_i           = radar_i_re;
        output_re.(ID).(scenario).radar_q           = radar_q_re;
        output_re.(ID).(scenario).radar_dist        = radar_dist_re;
        output_re.(ID).(scenario).radar_resp        = radar_resp_re;
        output_re.(ID).(scenario).radar_pulse       = radar_pulse_re;
        output_re.(ID).(scenario).radar_heartSound  = radar_heartsound_re;
        % [Resampled] gather contact data
        output_re.(ID).(scenario).contact_bp        = tfm_bp_re;
        output_re.(ID).(scenario).contact_ecg1      = tfm_ecg1_re;
        output_re.(ID).(scenario).contact_ecg2      = tfm_ecg2_re;
        output_re.(ID).(scenario).contact_icg       = tfm_icg_re;
        output_re.(ID).(scenario).contact_respiration = tfm_respiration_re;
        output_re.(ID).(scenario).contact_z0        = tfm_z0_re;
        output_re.(ID).(scenario).class             = tfm_intervention_re;
        output_re.(ID).(scenario).class_idx         = sc_new;
    end
end

if cut == 1
    disp('TODO');
else
    save('outputs/output_full_data_raw.mat','output_raw', '-v7.3');
    save('outputs/output_full_data_f.mat','output_f', '-v7.3');
    save('outputs/output_full_data_re.mat','output_re', '-v7.3');
end
