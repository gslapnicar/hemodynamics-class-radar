ROOT_PATH = '../data/';
CUT_LOW = 0.01;
CUT_HIGH = 5.0;
FILTER_ORDER = 2;

FILTERING = 1;
PLOT = 1;

fileList = dirPlus(ROOT_PATH, 'FileFilter', '.*\_new.mat$');

for i=1 : length(fileList)
    [filepath, name, ext] = fileparts(fileList{i});
    data = load(fileList{i}).full_data;
    
    signals = data.signal;
    fs = data.fs;
    raw_data = data.data;
    lengths = data.length;
    
    radar_I_Q = zeros(2, lengths{1});
    
    if PLOT
        figure;
        n_subplots = length(signals);
    end
    
    for k=1 : length(signals)
        if strcmp(signals{k}, 'radar_i')
            radar_I_Q(1, :) = raw_data{k};
        elseif strcmp(signals{k}, 'radar_q')
            radar_I_Q(2, :) = raw_data{k};
        end
        
        if FILTERING
            [b,a] = butter(FILTER_ORDER, [CUT_LOW, CUT_HIGH]/(double(fs{k})/2), 'bandpass');
            filtered_signal = filtfilt(b, a, raw_data{k});
        end
        
        if PLOT && i>1
            ax(k) = subplot(n_subplots,1,k);
            t = (1:length(raw_data{k}))./double(fs{k});
            plot(t, raw_data{k}, 'b', 'LineWidth', 2);
            if FILTERING
                hold on;
                plot(t, filtered_signal, 'r', 'LineWidth', 2);
            end
            title(signals{k});
            
            if k == length(signals)
                radar_A = sqrt((radar_I_Q(1, :).^2) + (radar_I_Q(2, :).^2));
                linkaxes(ax, 'x');
            end
        end
    end
    if i > 1
        return;
    end
end