% function springer_options = default_Springer_HSMM_options()
%
% The default options to be used with the Springer segmentation algorithm.
% USAGE: springer_options_advanced = default_Advanced_HSMM_options

%% Authors: Kilin Shi, Sven Schellenberger
% Copyright (C) 2017  Kilin Shi, Sven Schellenberger

function springer_options_advanced = default_Advanced_HSMM_options()

%% Wavelet level and name for getSpringerPCGFeatures
%  rbio3.9 with level 3 is unstable sometimes, use db10 level 5 from paper
%  "Performance Comparison of Denoising Methods for
%  Heart Sound Signal" instead
springer_options_advanced.wavelet_level = 5;
springer_options_advanced.wavelet_name = 'rbio3.9';

%% Filter settings for getSpringerPCGFeatures
springer_options_advanced.fc_high_getSpringer = 80;
springer_options_advanced.fc_low_getSpringer = 16;

%% Filter settings for getHeartRateSchmidt
springer_options_advanced.fc_high_Schmidt = 80;
springer_options_advanced.fc_low_Schmidt = 16;

%% Autocorrelation settings for getHeartRateSchmidtHRV
springer_options_advanced.min_indexHRV = 0.45;
springer_options_advanced.max_indexHRV = 1.45;

%% Filter settings for getHeartRateSchmidtHRV
springer_options_advanced.fc_high_SchmidtHRV = 80;
springer_options_advanced.fc_low_SchmidtHRV = 16;

%% Autocorrelation settings for getHeartRateSchmidt
springer_options_advanced.min_index = 0.45;
springer_options_advanced.max_index = 1.45;

%% Filter settings for getFilteredCutSignals
springer_options_advanced.fc_high_signals = 80;
springer_options_advanced.fc_low_signals = 16;

%% Std of diastole in get_duration_distributions
springer_options_advanced.std_diastole_var = 0.15;

