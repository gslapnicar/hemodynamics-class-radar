function [signal_norm] = norm1to1(signal)
% function [signal_norm] = norm1to1(signal)
% normalise signal between -1 and 1
% input: signal
% output: normalised signal
%% Authors: Kilin Shi, Sven Schellenberger
% Copyright (C) 2017  Kilin Shi, Sven Schellenberger

signal_norm = 2*(signal - min(signal))/(max(signal)-min(signal))-1;
