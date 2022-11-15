%% Oct 23, 2022
% Extract Features

%% pars
f_dat = fullfile('data_Grandchamp', 'preprocessing');
f_out  = fullfile('feats_matfile', 'Grandchamp');

% - general - 
winlen = 1; % in second 
load('pars_stfeats', 'chans')  % chans of study 2 for both 
%features = {'raw'};
features = {'raw', 'power', 'label'};
subjects = 1:2;
sessions = 1:11;

% - TF analysis setting - 
fRange_power = [4 40];
fRange_ISPC = [4 40];
nFreq_power = 35;
nFreq_ISPC = 35;
nCycleRange_power = [3 7];
nCycleRange_ISPC = [3 7];
baselineCorr_power = {}; 
baselineCorr_ISPC = {};
scaling_power = 'log';
winlenRange_ISPC = [1.5 2.5];

% - storage setting - 
downsamplize_arr = [256 50];
%downsamplize_arr = [0 0 50 50]; % only in temporal resolution (x-axis), in Hz; 0-OFF
overwrite =1 ; % can be changed in the pop-up dialogue box

% - print setting -
if overwrite 
    disp('OVERWRITE is ON. Old files will be overwritten.')
end

%% main 

% loop over studies, subs, features to compute feature trial-by-trial 
ci = 0;
wb = waitbar(0, 'Processing datasets...');
for si = 1:length(subjects)
    for ssi = 1:length(sessions)
        sub = subjects(si);
        session = sessions(ssi);
        eeg = loadData_Grandchamp(sub, session);
        
        wb2 = waitbar(0, 'Extracting features...');
        tfi = 0;
        for feati = 1:length(features)

            feature = features{feati};
            if ~strcmpi(feature, 'label')
                downsample = downsamplize_arr(feati);
            end

            % feature computing & saving: trial-by-trial
            nTrial = size(eeg.data,3);
            
            for triali = 1:nTrial
                % check feature file
                trial = triali;
                trialname = arrayfun(@(s) pad(num2str(s),4,'left', '0'), trial, 'UniformOutput',false);
                trialname = strcat(trialname{:});
                f_file = fullfile(f_out, num2str(sub), pad(num2str(session),2,'left','0'), trialname);
                if exist(f_file, 'file') ~= 7
                    mkdir(f_file)
                end

                f_feat = fullfile(f_file, [lower(feature), '.mat']);
                if exist(f_feat, 'file') == 2 && ~overwrite  % file exists and overwrite is off -> skip the trial
                    continue
                end

                % subset   
                raw = eeg.data(:,:,triali);

                % feature get
                if strcmpi(feature, 'label')
                    data = eeg.labels(triali);
                elseif strcmpi(feature, 'raw')
                    data = resample(double(raw'), downsample, eeg.srate);   % column as independent channel
                    data = data';

                elseif ismember(lower(feature), {'power'})

                    % loop over channels
                    tpnloopchan = size(raw,1);
                    for chani = 1: tpnloopchan % skip the last channel if feature == 'ispc'

                        if strcmpi(feature, 'power')
          
                            tpstruc = waveletConv(raw(chani,:), eeg.srate, fRange_power, nFreq_power, nCycleRange_power, scaling_power, 0,0,0,0);
                            tpd = computePower(tpstruc.data,  {}, baselineCorr_power, 1, {0, tpstruc.frex}, 0);
                        end 

                        if ~strcmpi(feature, 'ispc')

                            % downsamplize                      
                            if downsample
                                tpd = resample(tpd', downsample, eeg.srate);  
                                tpd = tpd';
                            end

                            % intialize
                            if chani == 1
                                data = zeros([size(tpd), size(raw,1)]);                      
                                data = single(data);
                            end

                            % register
                            data(:,:, chani) = tpd;
                        end 

                    end  % loop: chani
                end % feature selection

                % save
                save(f_feat, 'data')
                tfi = tfi+1;
                waitbar(tfi/length(features)/nTrial, wb2, 'Extracting features...')
            end  % loop: triali 
        end  % loop: feati (features)
        close(wb2)
        ci = ci+1;
        waitbar(ci/length(subjects)/length(sessions), wb, 'Processing datasets...')
    end % loop: session
end  % loop: subject
close(wb)

