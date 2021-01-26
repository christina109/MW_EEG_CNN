function eeg = loadData2(sub, timewin, chans)
% eeg = loadData2(sub, timewin = [all], chans = [all])
% Load data of Study2
% timewin: in ms
% eeg.data: nChan x nPnt x nTrial

f_main = fullfile(fileparts(which('mind_wandering_root')),'3data2');

% load pars
load(fullfile(f_main,'pars_EEG'), 'times', 'chanlocs')

% chans id
if nargin < 3 || isempty(chans)
    chans = {chanlocs.labels};
    chansid = 1:length(chanlocs);
else
    [~,chansid] = ismember(lower(chans), lower({chanlocs.labels}));
end

% subset times
if nargin < 2 || isempty(timewin)
    subtimes = times;
    timewinid = 1:length(times);
else 
    timewinid = dsearchn(times', timewin');
    subtimes = times(timewinid(1):timewinid(end));
end

% load
EEG = pop_loadset(fullfile(f_main,['preprocessing\\', num2str(sub), '_epochs_ica_a2.set']));
    
% event id in EEG.event
[~, eventsallid] = unique([EEG.event.epoch]); % suppose first event in each epoch is time = 0

% check true event count
if length(eventsallid) ~= EEG.trials
    error('Incorrect event structure!')
end

% subset: all trials
eeg.data = EEG.data(chansid, min(timewinid):max(timewinid), :);
eeg.urevent = [EEG.event(eventsallid).urevent];

% add info
eeg.sub   = sub;
eeg.times = subtimes;   
eeg.chans = chans;
eeg.srate = EEG.srate;

end