clear; close all; clc;

%{
SUBJECT = "GDN0016";
SCENARIO = "TiltDown";
SIGNAL = "contact_respiration";
fps = 100;

data_table = readtable(strcat('../python/csv_data/', SUBJECT, '/', SCENARIO, '.csv'));
data = data_table.(SIGNAL);
%}

data = readmatrix("../python/test.csv");
fps = 100;
spec_plot_l = 200;
P = data(1:end-1)';

% bandpass filtering
band = [0.1, 4.0];
[n,d]  = butter(1, band/(fps/2),'bandpass');
P = filtfilt(n, d, P);

% spectrogram
figure();

ax(1) = subplot(3,1,1);
spec = abs(spectrogram(P, 1*fps, 1*fps-1, 4096-1, 'yaxis'));
imagesc(spec(1:end,:));
set(gca,'YDir','normal');
ylabel('BR (BPM)')
title("MATLAB (built in)");

ax(2) = subplot(3,1,2);
spec_w = get_spec(P, fps);
imagesc(spec_w(1:end,:));
set(gca,'YDir','normal');
ylabel('BR (BPM)')
title("MATLAB (custom method)");

ax(3) = subplot(3,1,3);
hold on;
plot(data(1:end-1)');
plot(P);
%{
ax(3) = subplot(4,1,3);
test = load('../python/test.mat');
spec2 = test.Sxx;
imagesc(spec2(1:50,:));
set(gca,'YDir','normal');
ylabel('BR (BPM)')
title("SCIPY SPECTROGRAM");

ax(4) = subplot(4,1,4);
test2 = load('../python/test2.mat');
spec3 = test2.Zxx;
imagesc(spec3(1:50,:));
set(gca,'YDir','normal');
ylabel('BR (BPM)')
title("SCIPY STFT");
%}
function spec = get_spec(signal, fps)
    
    % define parameter
    L = fps * 1; % 10 sec window
    s = zeros(size(signal,2) - L + 1, L);
    
    for idx = 1 : size(signal,2) - L + 1
        p = signal(1, idx : idx + L - 1);
        s(idx,:) = hamming(L)' .* (p - mean(p)) / (eps + std(p)); %eps should be set accordingly for normalization "intensity"
    end
    
    s = abs(fft(s, 4096, 2));
    size(s)
    spec = s(:, 1:4096/2)';

    % Matlab function
%     spc = abs(spectrogram(signal,(L),L-1,4096-1,'yaxis'));
end