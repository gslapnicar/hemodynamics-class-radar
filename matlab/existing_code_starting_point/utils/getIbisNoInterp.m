function [ibi_ekg, ibi_test] = getIbisNoInterp(locsR, locs_test, medfilt_size, smooth_size, Fs)
%% Authors: Kilin Shi, Sven Schellenberger
% Copyright (C) 2017  Kilin Shi, Sven Schellenberger

%% Returns the IBIs
% locsR: location of locs of ECG in samples
% locs_test: location of test locs in samples

locsR = locsR./Fs;
locs_test = locs_test./Fs;

% IBI ECG
ibi_ekg = diff(locsR);
ibi_ekg = medfilt1(ibi_ekg,medfilt_size,'truncate');
ibi_ekg = smooth(ibi_ekg,smooth_size);
ibi_ekg = interp1(locsR(2:end),ibi_ekg,locsR(2):1/Fs:locsR(end));
% ibi_ekg = medfilt1(ibi_ekg,medfilt_size*Fs,'truncate');
% ibi_ekg = smooth(ibi_ekg,smooth_size*Fs);
% ibi_ekg = ibi_ekg';

% IBI Test
ibi_test = diff(locs_test);
ibi_test = medfilt1(ibi_test,medfilt_size,'truncate');
ibi_test = smooth(ibi_test,smooth_size);
ibi_test = interp1(locs_test(2:end),ibi_test,locs_test(2):1/Fs:locs_test(end));
% ibi_test = medfilt1(ibi_test,medfilt_size*Fs,'truncate');
% ibi_test = smooth(ibi_test,smooth_size*Fs);
% ibi_test = ibi_test';

% Calculate RMSE
if locsR(2) > locs_test(2)
    ibi_test_cut = ibi_test(round((locsR(2)-locs_test(2))*Fs):end);
    ibi_ekg_cut = ibi_ekg;
elseif locsR(2) < locs_test(2)
    ibi_test_cut = ibi_test;
    ibi_ekg_cut = ibi_ekg(round((locs_test(2)-locsR(2))*Fs):end);
else
    ibi_ekg_cut = ibi_ekg;
    ibi_test_cut = ibi_test;
end
if length(ibi_test_cut) > length(ibi_ekg_cut)
    ibi_test_cut = ibi_test_cut(1:length(ibi_ekg_cut));
elseif length(ibi_test_cut) < length(ibi_ekg_cut)
    ibi_ekg_cut = ibi_ekg_cut(1:length(ibi_test_cut));
end

time_overall = floor((length(ibi_test_cut)-1)/Fs);
% Take IBI Value at every second:
ibi_ekg = ibi_ekg_cut(1:Fs:(time_overall*Fs)+1)';
ibi_test = ibi_test_cut(1:Fs:(time_overall*Fs)+1)';

end

