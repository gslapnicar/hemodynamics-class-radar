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

%% Options
%%%% Choose Subject-ID(s) - can be singular integer (from 01 to 30) or a
%%%% range like 3:25
IDrange = 01;

%%%% Choose scnerio(s) 
% possible scenarios are {'Resting' 'Valsalva' 'Apnea' 'TiltUp' 'TiltDown'}
scenarios = {'Resting', 'Valsalva', 'Apnea', 'TiltUp', 'TiltDown'};

%%%% Set path to data
path = '../../data'; 

scrsz = get(groot,'ScreenSize'); % Screen size for plotting

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
        
        %% Prepare data
        output.(ID).(scenario) = struct;
        % Some black magic here, this ellipse fitting and compensation is
        % not trivial, details are here: https://www.nature.com/articles/s41598-018-29984-5
        % This code gets us from I and Q raw components to distance.
        % NOTE: source code is not accessable!
        [~,~,~,radar_dist] = elreko(radar_i,radar_q,measurement_info{1},0);
        % Usage of elreko
        % [radar_i_compensated,radar_q_compensated,phase_compensated,radar_dist] = elreko(radar_i,radar_q,measurement_info{1}(timestamp of dataset),0(Plot flag -> 1: plots on, 0: plots off));
        
        % Radar
        [radar_respiration, radar_pulse, radar_heartsound, tfm_respiration] = getVitalSigns(radar_dist, fs_radar, tfm_z0, fs_z0);
        
        % TFM
        if ECG_CHANNEL(IDrange(indx)) == 1
            tfm_ecg = fillmissing(tfm_ecg1,'constant',0); % Sometimes ECG is NaN -> set all occurrences to 0
        else
            tfm_ecg = fillmissing(tfm_ecg2,'constant',0); % Sometimes ECG is NaN -> set all occurrences to 0
        end
        % I assume this is a 4th order Butterworth bandpass filter with
        % cutoffs 1 Hz and 20 Hz.
        tfm_ecg = filtButter(tfm_ecg,fs_ecg,4,[1 20],'bandpass');
        
        % Resample radar respiration to match tfm respiration
        radar_respiration_re = resample(radar_respiration,fs_z0,fs_radar);
        radar_dist_re = resample(radar_dist,fs_z0,fs_radar);
        
        if length(radar_respiration_re) > length(tfm_respiration)
            radar_respiration_re = radar_respiration_re(1:length(tfm_respiration));
            radar_dist_re = radar_dist_re(1:length(tfm_respiration));
        elseif length(radar_respiration_re) < length(tfm_respiration)
            tfm_respiration = tfm_respiration(1:length(radar_respiration_re));
        end
        
        % Get scenario "signal" aka. ground truth, when something was
        % happening
        sc_sec = getInterventions(scenario,tfm_intervention,fs_intervention);
        
        %% Evaluate data
        % Heartbeat detection via hidden semi-Markov models (we get R peak
        % locations)
        % NOTE: source code is not available!
        radar_hs_states = getHsmmStates(radar_heartsound, fs_radar);
        radar_hs_locsR = statesToLocs(radar_hs_states, 1);
        [~,tfm_ecg_locsR] = twaveend(tfm_ecg(1:length(radar_hs_states)),fs_ecg,32*(fs_ecg/200),'p');
        
        
        %% Time vectors
        time_radar = 1/fs_radar:1/fs_radar:length(radar_dist)/fs_radar;
        time_ecg = 1/fs_ecg:1/fs_ecg:length(tfm_ecg)/fs_ecg;
        time_icg = 1/fs_icg:1/fs_icg:length(tfm_icg)/fs_icg;
        time_bp = 1/fs_bp:1/fs_bp:length(tfm_bp)/fs_bp;
        time_z0 = 1/fs_z0:1/fs_z0:length(tfm_z0)/fs_z0;
        time_respiration = 1/fs_z0:1/fs_z0:length(radar_respiration_re)/fs_z0;
        
        %% Plot all data
        figure('Position',[1 1 scrsz(3) scrsz(4)-80]);
        ax(1) = subplot(4,1,1);
        hold on;
        %plot(time_radar,radar_dist.*1000,'k-');
        plot(time_radar,radar_i,'b');
        plot(time_radar,radar_q,'r');
        set(gca, 'FontName', 'Arial');
        title('Radar', 'FontSize', 18);
        legend({'In-Phase component', 'Quadrature component'}, 'FontSize', 16);
        ylabel('Rel. Distance(mm)');
        % xlabel('Time(s)')

        ax(2) = subplot(4,1,2);
        plot(time_z0,tfm_z0,'k-');
        set(gca, 'FontName', 'Arial');
        title('Impedance', 'FontSize', 18)
        ylabel('Voltage (mV)');
        % xlabel('Time(s)')

        ax(3) = subplot(4,1,3);
        plot(time_ecg,tfm_ecg1+4,'k-');
        hold on;
        plot(time_ecg,tfm_ecg2+2,'b-');
        plot(time_icg,tfm_icg,'r-');
        set(gca, 'FontName', 'Arial');
        legend({'Lead 1','Lead 2', 'ICG'}, 'FontSize', 16)
        title('Electrocardiogram & Impedance cardiogram', 'FontSize', 18)
        ylabel('norm. Amplitude');
        % xlabel('Time(s)')

        ax(4) = subplot(4,1,4);
        plot(time_bp,tfm_bp,'k-')
        title('Blood pressure', 'FontSize', 18)
        ylabel('norm. Amplitude');
        xlabel('Time(s)', 'FontSize', 16)
        
        linkaxes(ax,'x');
        xlim([time_radar(1) time_radar(end)]);

        %% Compare radar vital signs to reference raw signals
        %{
        figure('Position',[1 1 scrsz(3) scrsz(4)-80]);
        ax2(1) = subplot(2,1,1);
        plot(time_respiration,radar_respiration_re.*1000,'k-');
        title('Compare respiration')
        ylabel('Rel. Distance(mm)');
        yyaxis right;
        plot(time_respiration,tfm_respiration);
        ylabel('Impedance')
        xlabel('Time(s)')

        ax2(2) = subplot(2,1,2);
        plot(time_radar,radar_heartsound.*10^6,'k-')
        hold on;
        set(vline(radar_hs_locsR(1)/fs_radar,'b-'),'HandleVisibility','on'); % Turn the legend on for vline
        vline(radar_hs_locsR./fs_radar,'b-');
        set(vline(tfm_ecg_locsR(1)/fs_ecg,'r-'),'HandleVisibility','on'); % Turn the legend on for vline
        vline(tfm_ecg_locsR./fs_radar,'r-');
        title('Compare heartbeat detection')
        ylabel('Rel. Distance(um)');
        xlabel('Time(s)')
        legend('Radar heart sound','Radar S1','ECG R-peak')
        
        linkaxes(ax2,'x');
        xlim([time_radar(1) time_radar(end)]);
        %}
        return;
        
        if IDrange(indx) > 4
        
            %% Compare Radar to TFM aggregated data
            figure('Position',[1 1 scrsz(3) scrsz(4)-80]);
            ax3(1) = subplot(2,1,1);
            plot(time_radar,radar_dist.*1000);
            title('Radar distance')
            ylabel('Rel. Distance(mm)');
            ax3(2) = subplot(2,1,2);
            plot(radar_hs_locsR(1:end-1)./fs_radar,60./(diff(radar_hs_locsR)./fs_radar));
            hold on;
            plot(tfm_param_time,tfm_param.HR);
            legend('Radar','TFM','Location','south east');
            title('Heart rate comparison');
            xlabel('Time(s)');
            ylabel('Heart rate(BPM)');
            linkaxes(ax3,'x')
            xlim([time_radar(1) time_radar(end)]);
    %         xlim([110 140]); %GDN0023 Apnea

            %% Show some TFM aggregated data
            figure('Position',[1 1 scrsz(3) scrsz(4)-80]);
            ax4(1) = subplot(4,1,1);
            plot(tfm_param_time,tfm_param.HR);
            ylabel('HR(BPM)')
            yyaxis right;
            plot(tfm_param_time,tfm_param.LVET);
            title('HR & LVET')
            ylabel('LVET(ms)')

            ax4(2) = subplot(4,1,2);
            plot(tfm_param_time,tfm_param.SV);
            ylabel('SV(ml)')
            yyaxis right;
            plot(tfm_param_time,tfm_param.HZV);
            title('SV & HZV')
            ylabel('HZV(l/min)')

            ax4(3) = subplot(4,1,3);
            plot(tfm_param_time,tfm_param.TPR);
            ylabel('TPR(dyne*s/cm^5)')
            yyaxis right;
            plot(tfm_param_time,tfm_param.TFC);
            title('TPR & TFC')
            ylabel('TFC(1/Ohm)')

            ax4(4) = subplot(4,1,4);
            plot(tfm_param_time,tfm_param.sBP);
            ylabel('sBP(mmHg)')
            yyaxis right;
            plot(tfm_param_time,tfm_param.dBP);
            title('sBP & dBP')
            ylabel('dBP(mmHg)')
            xlabel('Time(s)');

            linkaxes(ax4,'x');
            xlim([tfm_param_time(1) tfm_param_time(end)]);
            
        end
    end
end

