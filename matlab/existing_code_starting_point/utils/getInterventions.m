function y = getInterventions(scenario,intervention,fs)
    % y = getInterventions(scenario,intervention,fs)
    % Extract the necessary intervention points from the signal
    % Each scenario has its own intervention sequence
    %
    % Input:    scenario:      e.g.'Valsalva'
    %           intervention:  intervention signal
    %           fs:            sampling frequency
    %
    % Output:   y:             array with corresponding indices

    % Authors: Sven Schellenberger, Kilin Shi
    % Copyright (C) 2020  Sven Schellenberger, Kilin Shi
    %%

    st = 1;
    en = length(intervention);
    y = [st en];
    locs = getTriggerlocs(intervention,0,scenario);
    
    % This is for resting... expected empty locs (no presses) so full
    % signal is taken
    if isempty(locs)
        if ~strcmp(scenario,'Resting')
            warning('No triggerlocs found')
        end
        return;
    end

    %% Distinguish scenarios
    if strcmp(scenario,'Valsalva')
        % Valsalva maneuver is executed 3 times, start and end points are
        % marked with a button press (falling and rising edge)
        % That means typically 6 button presses, but it can happen that not all are
        % detected
        if length(locs) < 6
            warning('less than 6 intervention points (2 per Valsalva)')
            if mod(length(locs),2)
                warning('Odd number of edges, something is missing')
                return;
            else
                warning('even number')
            end
        elseif length(locs) > 6
            warning('too many intervention points')
            return;
        end
        y = reshape(locs,2,[])'; % this stacks the vector into matrix of n x 2
        durations = diff(y,1,2)./fs;
        if any(durations > 27*fs)
           warning('Valsalva maneuver too long') 
        end

    elseif strcmp(scenario,'TiltUp')
        % During TiltUp the table is tilted upright. At the beginning of the
        % tilting action and at the end position the button is pressed.
        % At the beginning of the study the measurement was started with an
        % upright table, later on only the upright position was marked and then
        % the start and end of the tilting action
        % That means: 0, 1 and 2 button presses are possible
        if length(locs) == 2
            % GASPER CHANGE
            %y(1) = locs(2);
            y(1) = locs(1);
            y(2) = locs(2);
        elseif length(locs) == 1
            y(1) = locs(1);
            warning('only one intervention. assuming it is the upright position');
        elseif isempty(locs)
            warning('no interventions found')
            return;
        else
            warning('the number of interventions for this scenario is too high')
            return;
        end

    elseif strcmp(scenario,'TiltDown')
        % During TiltDown the table is tilted back in the horizontal position. At the beginning of the
        % tilting action and at the end position the button is pressed.
        % At the beginning of the study the measurement was started with an
        % horizontal table, later on only the horizontal position was marked and then
        % the start and end of the tilting action
        % That means: 0, 1 and 2 button presses are possible
        if length(locs) == 2
            % GASPER CHANGE
            %y(1) = locs(2);
            y(1) = locs(1);
            y(2) = locs(2);
        elseif length(locs) == 1
            y(1) = locs(1);
            warning('only one intervention. assuming it is the horizontal position');
        else
            warning('the number of interventions for this scenario is too high')
            return;
        end

    elseif strcmp(scenario,'Apnea')
        % Subjects press the button as long as they hold their breath
        % That means the section from falling to rising edge is an apnea
        % section. Typically the subjects do the apnea part two times, one time
        % with breathing in and one time with breathing out before holding
        % breath
        locs_start = getTriggerlocs(intervention,2,scenario); % falling edge is start
        locs_end = getTriggerlocs(intervention,1,scenario); % rising edge is end
        if length(locs_start) == length(locs_end)
            y = [locs_start' locs_end'];
        else
            warning('odd number of falling/rising edges');
            return;
        end

    else
        warning('No interventions expected for this scenario')
    end
end



function locs = getTriggerlocs(intervention,mode,scenario)
    % function locs = getTriggerlocs(intervention,mode)
    % return trigger locations
    % mode: 0(default): mid rising-falling (1) rising edge (2) falling edge
    %% 
    if nargin < 2
       mode = 0; 
    end

    minimum = min(intervention);
    if minimum > 2
        locs = [];
        return
    end

    % Convert the analog signal voltage into a binary signal and find edges
    ext_norm = norm1to1(intervention);
    ext_sign = sign(ext_norm)';
    %figure;
    %plot(ext_sign);
    %title(scenario);
    ext_sign(ext_sign == 0) = ext_sign(ext_sign == 0) -1;
    falling = strfind(ext_sign,[1 -1]);
    rising = strfind(ext_sign,[-1 1]);

    % Pushing the button leads to a falling and rising edge in the signal.
    % Some errors could lead to a rising or falling edge at the beginning or
    % end, respectively
    if ~isempty(rising)
        if rising(1) == 1
            rising(1) = [];
            warning('deleted rising at beginning at first bin')
        end
    end
    if ~isempty(falling)
        if falling(1) == 1
            falling(1) = [];
            warning('deleted falling at beginning at first bin')
        end
    end
    if length(falling) ~= length(rising)
        margin = length(intervention)-length(intervention)*0.01;
        if falling(end) > margin
            falling(end) = [];
            warning('deleted falling edge near end')
        elseif rising(end) > margin
            rising(end) = [];
            warning('deleted rising edge near end')
        else
            error('falling and rising edge count not equal');
        end
    end

    % Delete short glitches
    glitch = rising - falling;
    glitch_ind = find(glitch < 3);
    if glitch_ind
       falling(glitch_ind) = []; 
       rising(glitch_ind) = [];
    end

    % Return indices of edges
    switch mode
        case 0
            locs = ceil((rising+falling)./2);
        case 1
            locs = rising;
        case 2
            locs = falling;
    end

end