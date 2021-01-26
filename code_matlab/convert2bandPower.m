function convert2bandPower(studyId, subs)
% convert2bandPower(studyId, subs)
% Decompose the data into time-frequency domain using wavelet
% transformation and compute the power. 
% studyId = [1,2];
% Author: Christina Jin (christina.mik109@gmail.com)


% path
f_main = fileparts(which('mind_wandering3'));
f_output = [f_main, 'inputs_matfile\\'];
cd(f_main)

% pars
bands = struct();
% pars
load('pars_markers.mat', 'bands')

% band definition
id = find(strcmpi({bands.name}, band));
measure = bands(id).name;
bandrange = bands(id).range;
chans = bands(id).chans;

% subset windows to average
baseline  = [-1000 0];
stimOn    = [0 1000];

% progress
n = length(subs) * length(condcell);
i = 0;
progressbar(['Computing power in ', upper(band), ' band...'])

% loop over conds
for condi = 1:length(condcell)
    cond = condcell{condi};
    
    % loop over subs
    for subi = 1:length(subs)
        sub = subs(subi);
        
        % load
        if studyId == 1
            error('Unfinished Code: studyId == 1')
        elseif studyId == 2
            eeg = loadData2(sub, [baseline(1) stimOn(2)], []);
        else
            error('Unknown study!')                
        end
        
        % get pars
        if subi == 1
            times = eeg.times;
            srate = round(1000*length(eeg.times) / (eeg.times(end) - eeg.times(1)+1));
            baseIdx   = dsearchn(times', baseline');
            stimOnIdx = dsearchn(times', stimOn');
        end
        
        % intialize 
        if studyId == 2
            mat = zeros(length(eeg.urevent),1+length(times)); % colnames: urevent, base, sto
            mat(:,1) = eeg.urevent;
        elseif studyId == 1
            mat = zeros(length(eeg.urevent),2+length(times)); % colnames: urevent, base, sto
            mat(:,1:2) = eeg.urevent;
        end
        
        % wavelet convolution &&&
        dataFilt = waveletTransform(eeg.data, srate);
        if sub == subs(1)
            dataFilt = wavelet_conv(eeg.data, srate, bandrange, checkOn);
        else 
            dataFilt = wavelet_conv(eeg.data, srate, bandrange, 0);
        end

        % power
        powerAllChans = compute_power(dataFilt, [], 0);  % baseline correction: 'z', '%', 0. 
        
        % loop over chans
        for chani = 1:length(chans)
                      
            chan = chans{chani};
            
            % subset
            power = powerAllChans(chani,:,:);
            
            % register
            mat(:,2) = mean(power(1,baseIdx(1):baseIdx(end),:),2);
            mat(:,3) = mean(power(1,stimOnIdx(1):stimOnIdx(end),:),2);
            
            % output
            varName = [measure,'_',lower(cond)];
            file = [f_output, measure, '_', chan, '\\', num2str(sub),'.mat'];
            eval([varName,'= mat;'])
            if exist(file,'file') == 2
                save(file, varName, '-append')
            else
                save(file, varName)
            end
            
        end  % chani
        
        % progress bar
        i = i+1;
        progressbar(i/n)
        
    end   
end  


end  

