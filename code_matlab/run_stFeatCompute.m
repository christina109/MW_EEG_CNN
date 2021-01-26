%% Oct 29, 2019
% Extract Features

%% pars
ROSAon = 0;
gpu2use = 0;  % 0-OFF, 1~4 GPU number on ROSA
testOn = 0;  % turn ON/OFF time elapse estimate for a comparison between ROSA vs local
pbOn = 0; % turn ON/OFF progress bar 

if ROSAon
    addpath(genpath('/home/p279421/matlab programs/eeglab14_1_1b'))
    addpath '/home/p279421/TOPIC_mind wandering'
    addpath '/home/p279421/TOPIC_mind wandering/3data3'
    addpath '/home/p279421/TOPIC_mind wandering/3data3/code_matlab'
    addpath '/home/p279421/files/TOOL_code'
    
    if gpu2use > 0
        gpuDevice(gpu2use)
    end   
end

f_main = fullfile(fileparts(which('mind_wandering_root')),'3data3');
f_out  = fullfile(f_main, 'feats_matfile');
cd(f_main)

% - general - 
studies = 2;
%studies = [1 2];
baseline = [-200 0];  % [] to turn off
timewin = [-400 1000];
load('pars_stfeats', 'chans')  % chans of study 2 for both 
features = {'raw'};
%features = {'raw', 'Wst', 'power', 'ISPC'};
% load(fullfile(fileparts(which('mind_wandering_root')),'3data','pars_data3'),'subs2use')
%subs = {subs2use};
%subs = {1:30, 301:330}; % following the study order
subs = {301:330};

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
downsamplize_arr = 0;
%downsamplize_arr = [0 0 50 50]; % only in temporal resolution (x-axis), in Hz; 0-OFF
overwrite = 0; % can be changed in the pop-up dialogue box

% - legacy checking -
if length(downsamplize_arr) ~= length(features)
    error('Incorrect downsamplize_arr length. Check with the input features.')
end

% - print setting -
if overwrite 
    disp('OVERWRITE is ON. Old files will be overwritten.')
end

%% main 

% initialize the progress monitor
if pbOn 
    n_total = sum(cellfun(@length, subs)) * length(features);
    i_current = 0;
    progressbar('Computing features...', 'Progress on trials...', 'Progress on channels...');   
else 
    dispstat('', 'init')
end

% time elapse est intialize
if testOn; nTest = 0; end

% loop over studies, subs, features to compute feature trial-by-trial 
for studi = 1:length(studies)
    study = studies(studi);
    subs_tp = subs{studi};
    
    for si = 1:length(subs_tp)
        sub = subs_tp(si);
        eval(['eeg = loadData', num2str(study), '(sub, timewin, chans);'])
        
        if ~pbOn
            dispstat(sprintf('Process subject %d/%d of study %d.', si, length(subs_tp), study), 'keepprev', 'keepthis', 'timestamp')
        end
                    
        for feati = 1:length(features)
            
            % for testi = 1:10  % for time estimate purposes
            tic
            feature = features{feati};
            downsample = downsamplize_arr(feati);
            if ~pbOn; dispstat(sprintf('Process feature(%d/%d) %s.', feati, length(features), upper(feature)), 'keepprev', 'keepthis', 'timestamp'); end
            
            % feature computing & saving: trial-by-trial
            nTrial = size(eeg.data,3);
            for triali = 1:nTrial
                
                if triali == 1; time_trial_arr = []; end
                
                % progress monitor
                if ~pbOn; dispstat(sprintf('Progress over trials: %d%%, estimating %d minutes remaining', ...
                        round(triali/nTrial*100), mean(time_trial_arr)/60*(nTrial - triali + 1))); end
                
                % check feature file
                if study == 1
                    trial = eeg.urevent(triali,:);  % trial id
                elseif study == 2
                    trial = eeg.urevent(triali);
                end
                trialname = arrayfun(@(s) pad(num2str(s),4,'left', '0'), trial, 'UniformOutput',false);
                trialname = strcat(trialname{:});
                f_file = fullfile(f_out, num2str(study), pad(num2str(sub),3,'left','0'), trialname);
                if exist(f_file, 'file') ~= 7
                    mkdir(f_file)
                end
                
                f_feat = fullfile(f_file, [lower(feature), '.mat']);
                if exist(f_feat, 'file') == 2 && ~overwrite  % file exists and overwrite is off -> skip the trial
                    continue
                end

                tic
                                             
                % subset
                if testOn; tic; end                
                raw = eeg.data(:,:,triali);
                if gpu2use > 0; raw = gpuArray(raw); end
                if testOn; time_read = toc; end
                if testOn; disp(['Elapsed time to READ one trial data is ', num2str(time_read), ' seconds.']); end

                % baseline corr
                if testOn; tic; end
                if ~isempty(baseline)
                    baselineidx = dsearchn(eeg.times', baseline');
                    raw = bsxfun(@minus, raw, mean(raw(:,baselineidx),2)); % single-trial baseline correction                
                end
                if testOn; time_bcorr = toc; end
                if testOn; disp(['Elapsed time to CORRECT BASELINE is ', num2str(time_bcorr), ' seconds.']); end
             
                % feature get
                if strcmpi(feature, 'raw')
                    data = ifelse(gpu2use>0, gather(raw), raw);
                
                elseif ismember(lower(feature), {'wst', 'power', 'ispc'})
                    
                    % loop over channels
                    tpnloopchan = ifelse(strcmpi(feature,'ispc'), size(raw,1)-1,size(raw,1));
                    for chani = 1: tpnloopchan % skip the last channel if feature == 'ispc'
                        
                        if strcmpi(feature, 'wst')
                            if testOn; tic; end
                            if testOn && nTest == 0; time_wst = 0; end                               
                            tpd = computeWst(raw(chani,:), eeg.times, 0, 0, gpu2use);
                            if testOn; time_wst = (time_wst*nTest + toc)/(nTest+1); end
                            if testOn; disp(['Mean elapsed time for WST convolution is ', num2str(time_wst), ' seconds.']); end
                            
                        elseif strcmpi(feature, 'power')
                            
                            if testOn; tic; end
                            if testOn && nTest == 0; time_conv = 0; end                            
                            tpstruc = waveletConv(raw(chani,:), eeg.srate, fRange_power, nFreq_power, nCycleRange_power, scaling_power, 0,0,0,0, gpu2use);
                            if testOn; time_conv = (time_conv*nTest + toc)/(nTest+1); end
                            if testOn; disp(['Mean elapsed time for WAVELET CONVOLUTION is ', num2str(time_conv), ' seconds.']); end
                            
                            if testOn, tic, end
                            if testOn && nTest == 0; time_power = 0; end
                            tpd = computePower(tpstruc.data, eeg.times, baselineCorr_power, 1, {0, tpstruc.frex}, 0);
                            if testOn; time_power = (time_power*nTest + toc)/(nTest+1); end
                            if testOn; disp(['Mean elapsed time for POWER EXTRACTION is ', num2str(time_power), ' seconds.']); end
                            
                        elseif strcmpi(feature, 'ispc')
                            
                            if testOn; tic; end
                            if testOn && nTest == 0; time_conv = 0; end
                            tpstruc = waveletConv(raw(chani,:), eeg.srate, fRange_power, nFreq_power, nCycleRange_power, scaling_power, 0,0,0,0, gpu2use);
                            if testOn; time_conv = (time_conv*nTest + toc)/(nTest+1); end
                            if testOn; disp(['Mean elapsed time for WAVELET CONVOLUTION is ', num2str(time_conv), ' seconds.']); end
                            
                            for chanj = chani+1:size(raw,1)
                                if testOn; tic; end
                                if testOn && nTest == 0; time_convj = 0; end
                                tpstruc2 = waveletConv(raw(chanj,:), eeg.srate, fRange_power, nFreq_power, nCycleRange_power, scaling_power, 0,0,0,0,gpu2use);
                                if testOn; time_convj = (time_convj *nTest +toc)/(nTest+1); end
                                if testOn; disp(['Mean elapsed time for WAVELET CONVOLUTION of channal j is ', num2str(time_convj), ' seconds.']); end
                                
                                if testOn; tic; end
                                if testOn && nTest == 0; time_ispc = 0; end
                                tpd = computeISPC(tpstruc.data, tpstruc2.data, tpstruc.frex, eeg.times, winlenRange_ISPC, 0, baselineCorr_ISPC, 1,0,0,gpu2use);
                                if testOn; time_ispc = (time_ispc*nTest + toc)/(nTest+1); end
                                if testOn; disp(['Mean elapsed time for ISPC computation is ', num2str(time_ispc), ' seconds.']); end
                                
                                if testOn; tic; end
                                if gpu2use >0; tpd = gather(tpd); end
                                if downsample
                                    tpd = resample(tpd', downsample, eeg.srate);  
                                    tpd = tpd';
                                end
                                
                                if chani == 1 && chanj == 2
                                    data = zeros([size(tpd), size(raw,1)*(size(raw,1)-1)/2]);
                                    data = single(data);
                                end
                                
                                data(:,:,sum(size(raw,1)-1:-1:size(raw,1)-chani+1)+chanj-chani) = tpd;
                                if testOn && nTest ==0; time_ds = 0; end
                                if testOn; time_ds = (time_ds *nTest + toc)/(nTest+1); end
                                if testOn; disp(['Mean elapsed time for DOWNSMAPLIZATION is ', num2str(time_ds), ' seconds.']);end
                                
                                if testOn; nTest = nTest + 1; end
                            end  % loop: chanj
                        end % if: feature selection
                        
                        if ~strcmpi(feature, 'ispc')
                            
                            if testOn, tic, end
                            if testOn && nTest == 0; time_ds = 0; end
                            if gpu2use > 0; tpd = gather(tpd); end
                            
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
                            if testOn; time_ds = (time_ds * nTest + toc)/(nTest+1); end
                            if testOn; disp(['Mean elapsed time for DOWNSAMPLIZATION is ', num2str(time_ds), ' seconds.']); end
                        end
                        
                        % progress
                        if pbOn; progressbar([],[], chani/tpnloopchan); end
                        if testOn && ~strcmpi(feature,'ispc'); nTest = nTest + 1; end
                                               
                    end  % loop: chani
                end % feature selection
                
                % save
                if testOn, tic, end
                save(f_feat, 'data')

                    %if ~overwrite  % if overwrite is OFF, ask for permission
                    %    msg = 'Feature has been computed before. Saving the new feature will overwrite previous file.';
                    %    title = 'Confirm Save';
                    %    selection = questdlg(msg,title, 'overwrite', 'cancel', 'cancel');            
                    %    if strcmpi(selection, 'overwrite')
                    %        disp('Feature files will be OVERWRITTEN.')
                    %        overwrite = 1;  % overwrite ON 
                    %    elseif strcmpi(selection, 'cancel')
                    %        error('Session abort.')
                    %    end
                    %end
                    
                    %if overwrite
                    %    save(f_feat, 'data', '-append')
                    %end
   
                if testOn; time_save = toc; end
                if testOn; disp(['Elapsed time for saving feature ', upper(feature), ' of ONE TRIAL is ', num2str(time_save), ' seconds.']); end
                
                if pbOn 
                    if testOn; tic; end
                    progressbar([], triali/size(eeg.data,3), 0)
                    if testOn; time_pb = toc; end
                    if testOn; disp(['Elapsed time for updating the PROGRESSBAR is ', num2str(time_pb), ' seconds.']); end
                end
                
                time_trial = toc;
                time_trial_arr = [time_trial_arr time_trial];
                if testOn; disp(['Elapsed time for processing feature ', upper(feature), ' of ONE TRIAL is ', num2str(time_trial), ' seconds.']); end
                
            end  % loop: triali 
            toc
            % end % loop: tesi

            % progress update
            if pbOn
                i_current = i_current + 1;           
                progressbar(i_current/n_total, 0, [])
            end

        end  % loop: feati (features)
        
    end  % loop: si (subs_tp)

end  % loop: studi(studies)