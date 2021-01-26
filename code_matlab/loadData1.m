function eeg = loadData1(sub, timewin, chans)
% eeg = loadData1(sub, timewin = [all], chans = [all])
% Load data of Study1(both sessions)
% timewin: in ms
% eeg structure includes:
% .data: nChan x nPnt x nTrial
% .sub, .times, .chans, .srate

f_main = fullfile(fileparts(which('mind_wandering_root')),'3data');

% load pars
load(fullfile(f_main,'pars'), 'times', 'chanlocs')

% defaults
if nargin < 2 || isempty(timewin)
    timewin = [times(1) times(end)];
    disp('No TIMEWIN specified. Load all.')
end
if nargin < 3 || isempty(chans)
    chans = {chanlocs.labels};
    disp('No CHANS specified. Load all.')
end

% subset by channel
% try the orignial labelling first (A-D 1-32)
[~, chansid] = ismember(lower(chans), lower({chanlocs.labels}));  % take the id instead of logical vector for remaining the order
chansid = chansid(chansid>0);  % subset for matched ones

if length(chansid) < length(chans)*.5 % match less than 50%? 
    % maybe they are 10-20 IS labelings.
    disp('No matching channels. Interpreting using 10-20 system.')
    load('bs2is','bs2is')
    [~, tempid] = ismember(lower(chans), lower({bs2is{:,2}}));
    tempid = tempid(tempid>0);
    if length(tempid) < length(chans)*.5  % still no match?
        disp('Invalid CHANS specified. Load all.')
        chans = {chanlocs.labels};
        chansid = 1:length(chanlocs);
    else 
        chansBS = bs2is(tempid,1);
        disp('Conversion done.')
    end
    
    % try chansBS with the orignial labelling
    [~,chansid] = ismember(lower(chansBS), lower({chanlocs.labels}));
    
    % no match after converting from 10-20 system?
    if length(chansid) < length(chans)*.5
        disp('Invalid CHANS specidied. Load all.')
        chans = {chanlocs.labels};
        chansid = 1:length(chanlocs);
    end
end

% subset by time
timewin = [min(timewin) max(timewin)]; % formatting
timewinid = dsearchn(times', timewin');
subtimes = times(timewinid(1):timewinid(end));

% loop over sessions
nsession = 2;
for ssi = 1:nsession
    
    % load
    EEG = pop_loadset(fullfile(f_main,'preprocessing',[num2str(sub),'_', num2str(ssi), '_epochs_ica_o_a.set']));
    
    % checking
    if size(EEG.data,3) ~= length(EEG.event)
        error('Each trial in EEG.data contains multiple events. Further selection required.')
    end
    
    % excluding artifacts
    trials = ~EEG.reject.rejmanual; 

    % subset
    tpd = EEG.data(chansid, timewinid(1):timewinid(2), trials);
    tpe = [EEG.event(trials).urevent];
    tpe_mat = zeros(length(tpe), nsession);
    tpe_mat(:,ssi) = tpe;
    
    if ssi == 1
        data = tpd;
        urevent = tpe_mat;
    else 
        data(:,:,end+1:end+sum(trials)) = tpd;
        urevent(end+1:end+sum(trials),:) = tpe_mat;
    end

end % loop: sessions (ssi)

% add info
eeg.data  = data;
eeg.urevent = urevent;
eeg.sub   = sub;
eeg.times = subtimes;   
eeg.chans = chans;  % output the specified chans
eeg.srate = EEG.srate;

end