function locs = statesToLocs(states, type)
% function locs = statesToLocs(states)
% get the start locations for detected S1 or S2
% input: state vector from runSpringer, type means if S1 or S2 is returned
% as output
% output: start of S1 or S2 locations
%% Authors: Kilin Shi, Sven Schellenberger
% Copyright (C) 2017  Kilin Shi, Sven Schellenberger

if isrow(states)
    states = states';
end

if type == 1
    states_1 = states;
    states_1(states_1~=1) = 0; % delete all other states
    locs = strfind(states_1',[0 1])'; % state 1 is S1
    locs = locs + 1;
elseif type == 2
    states_3 = states;
    states_3(states_3~=3) = 0; % delete all other states
    locs = strfind(states_3',[0 3])'; % state 3 is S1
    locs = locs + 1;
else
    error('input is invalid!');
end
    
end

