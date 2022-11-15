function EEG = preprocessing_Grandchamp(sub, stage, session)
% preprocessing(data, stage, session)
% data can be multiple in some stages
% stage 1: import data, filtering, downsampling, interpolation, epoching
% stage 2: visually inspect epochs before ICA
% stage 3: remove marked epochs and run ICA
% stage 4: visually inspect components
% stage 5: remove marked comps and visually inspect epochs again
% stage 6: remove marked epochs 
% Correspondance: Christina Jin (cyj.sci@gmail.com)

p_eeg = fullfile('data_Grandchamp', ['sub-0', num2str(sub)], 'eeg');
p_prepro = fullfile('data_Grandchamp', ['preprocessing']);

%srate         = 500; 
bandpass      = [0.1 42]; 
epoch         = [-10 8];%in seconds
baseline      = [-0.2 0];
triggers      = {'30', 'condition 50'};
cmps2plot     = 30:-1:1;  

%%        
if stage == 1

    % import raw
    f_eeg = ['sub-0', num2str(sub), '_ses-', num2str(session), '_task-BreathCounting_eeg.bdf'];
    disp(f_eeg);
    EEG = pop_biosig(fullfile(p_eeg, f_eeg), 'channels', 1:64); 
    
    % add channel locations
    EEG=pop_chanedit(EEG, 'lookup','C:\\Programs\\Matlab programs\\eeglab2022.0\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc');
 
    % reref
    EEG = pop_reref( EEG, [16 53] ,'keepref','on');

    % band-pass filtering
    EEG = pop_eegfiltnew(EEG, min(bandpass), max(bandpass));

    % down-sampling
    %EEG = pop_resample(EEG, srate);

    % extract epochs
    EEG = pop_epoch(EEG, triggers, epoch); 

    % baseline correction
    EEG = pop_rmbase(EEG, baseline*1000);
    %if EEG.trials ~= 825
    %    error('Incorrect trigger number!')
    %end

    % save 
    pop_saveset(EEG, fullfile(p_prepro,[num2str(sub), '_', num2str(session), '_epochs.set']));

end

%%
% run ICA direclty since not many trials with these datasets
if stage == 2
    
    disp('Skip the mannual inspection before the ICA.')
            % load data
    EEG = pop_loadset(fullfile(p_prepro,[num2str(sub), '_', num2str(session), '_epochs.set']));
    if 0

        % mannual processing
        disp ('Call the following:')
        disp ('pop_eegplot(EEG);')  % visual inpection

        % save markers
        disp (['pop_saveset(EEG, fullfile(p_prepro, ''', num2str(sub), '', num2str(session), '_epochs.set''))']);
    end

end

%%

if stage == 3

    % load 
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub), '_', num2str(session), '_epochs.set']));

    % reject the marked epoches
    %EEG = pop_rejepoch(EEG, EEG.reject.rejmanual);

    % run ica 
    EEG = pop_runica(EEG, 'extended', 1, 'stop', 1E-7);

    % overwrite 
    pop_saveset(EEG, fullfile(p_prepro, [num2str(sub), '_', num2str(session), '_epochs_ica.set']));

end

%%

if stage == 4

    % load 
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub), '_', num2str(session),  '_epochs_ica.set']));

    % plot cmps
    disp('Call the following:')
    disp(['pop_prop(EEG, 0,[', num2str(cmps2plot),'], NaN,{''freqrange'' [0.1 42]});'])

    % save markers
    disp (['pop_saveset(EEG, fullfile(p_prepro, ''', num2str(sub), '_', num2str(session), '_epochs_ica.set''))']);

end

%%

if stage == 5

    % load 
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub),'_', num2str(session), '_epochs_ica.set']));

    % remove comps
    EEG = pop_subcomp(EEG);

    % baseline correction
    EEG = pop_rmbase(EEG, baseline*1000);

    % save
    EEG = pop_saveset(EEG, fullfile(p_prepro, [num2str(sub),'_', num2str(session), '_epochs_ica_a.set']));

    % visually inspect epochs again
    disp ('Call the following:')
    disp ('pop_eegplot(EEG);')

    % save markers
    disp (['pop_saveset(EEG, fullfile(p_prepro, ''', num2str(sub), '_', num2str(session), '_epochs_ica_a.set''))']);

end

%%

if stage == 6

    % load
    EEG = pop_loadset(fullfile(p_prepro, [num2str(sub),'_', num2str(session), '_epochs_ica_a.set']));

    % reject the marked epoches
    EEG = pop_rejepoch(EEG, EEG.reject.rejmanual);
    
    % save 
    pop_saveset(EEG, fullfile(p_prepro, [num2str(sub), '_', num2str(session), '_epochs_ica_a2.set']));

end

end %func


