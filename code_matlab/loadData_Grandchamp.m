function eeg = loadData_Grandchamp(sub, session)
% eeg structure includes:
% .data: nChan x nPnt x nTrial
% .sub,  .chanlocs, .srate

f_dat = fullfile('data_Grandchamp', 'preprocessing');

% load
EEG_raw = pop_loadset(fullfile(f_dat, [num2str(sub),'_', num2str(session), '_epochs_ica_a2.set']));

% label: 0-ot, 1-mw
labels = [];
% ot trials [0 8] time locked to trigger 50
EEG = pop_epoch(EEG_raw, {'condition 50'}, [0 8]);
dat = EEG.data;
labels = [labels repelem([0], EEG.trials)];
% mw trials [-10 -2] time locked to trigger 30
EEG = pop_epoch(EEG_raw, {'30'}, [-10 -2]); 
dat(:,:, end+1:end+size(EEG.data,3)) = EEG.data;
labels = [labels repelem([1], EEG.trials)];

% add info
eeg.data  = dat;
eeg.labels = labels;
eeg.chanlocs   = EEG.chanlocs;
eeg.srate = EEG.srate;

end