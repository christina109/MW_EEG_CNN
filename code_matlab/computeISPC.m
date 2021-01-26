function ISPC = computeISPC(data1, data2, frex, times, winlenRange, byTrialOn, baseline, plotOn, pbOn, parsPrintOn, gpu2use)
% ISPC = computeISPC(data1, data2, frex, times, winlenRange, byTrialOn, baseline, plotOn)
% data1/data2: complex data of the SAME nFreq x nPnt x nTrial
% frex: the frequency range of the first dimention of the input data
% times: a time point vector with length of nPnt; default to 0:nPnt-1
% winlenRange: min and max values of the numbers of the cycles of the
%              wavelet at both ends of the central frequency; 
%              will be linearly spaced to match the frex length.
%              default to [1.5 3.5].
% byTrialOn = 0: turn ON/OFF (default) the ISPC-trials. 
% baseline = {}: baseline time window, default to {} (no correction). 
%           This par can only work when TIMES has been specified
% plotOn = 1: turn ON/OFF plotting(if singleTrialOn plot random 9 trials)
% pbOn = 1: turn ON/OFF showing the progress bar
% parPrintOn = 1: turn ON/OFF printing the parameters
% gpu2use = 0: turn OFF or specify the GPU number
% ISPC: the turn is a ISPC matrix of nFreq x nPnt x nTrial (byTrialOFF)
%       or of nFreq x nPnt (byTrialON)
%       Note at both ends there are zero areas (bcorred) due to the window size
% NOTES. This function will slow down the computation speed on GPU. 

% default & legacy checking

if nargin < 5 || isempty(winlenRange) || ~isnumeric(winlenRange) || any(winlenRange <= 0)
    disp('No/Invalid winlenRange specified. Use default.')
    winlenRange = [1.5, 3.5];
end

if nargin < 6 || isempty(byTrialOn) || ~ismember(byTrialOn,[0 1])
    disp('No/Invalid byTrialOn specified. Use default.')
    disp('Compute ISPC-time.')
    byTrialOn = 0;
end
    
if nargin < 7 || isempty(baseline) || ~isnumeric(baseline) || length(unique(baseline)) < 2
    if parsPrintOn
        disp('No/Invalid BASELINE specified.')
        disp('Baseline Correction is OFF.')
    end
    bcorr = 0;
else
    bcorr = 1;
end

if nargin < 8 || isempty(plotOn) || ~ismember(plotOn, [0 1])
    plotOn = 1;
end

if nargin < 9 || isempty(pbOn) || ~ismember(pbOn, [0,1])
    pbOn = 1;
end

if nargin < 10 || isempty(parsPrintOn) || ~ismember(parsPrintOn, [0 1])
    parsPrintOn = 1;
end

if nargin < 11 || isempty(gpu2use) || ~isnumeric(gpu2use)
    gpu2use = 0;
end
 
% display settings & legacy checking
if isreal(data1) || isreal(data2)
    error('Data must have an IMAGINARY part')
end
if size(data1) ~= size(data2)
    error('Data1 and Data2 must of the same size.')
end

if length(frex) ~= size(data1,1) 
    error('FREX must match the FISRT DIMENSION of the data.')
end

if length(times) ~= size(data1,2)
    error('TIMES must match the second DIMENSION of the data.')
end

winlenRange = [min(winlenRange) max(winlenRange)];
if parsPrintOn, disp(['Window size: ', num2str(winlenRange(1)) ' to ', num2str(winlenRange(2)), ' cycles of the central frequency at both ends.']), end

if bcorr
   baseline = [min(baseline) max(baseline)];
   if parsPrintOn, disp(['Baseline is [', num2str(baseline(1)), ' ', num2str(baseline(2)), '].']), end
end

if parsPrintOn
    if byTrialOn
        disp('Compute ISPC-TRIALS.')
    else
        disp('Compute ISPC-TIME.')
    end
end

% compute pars
srate = 1000/((times(end)-times(1))/(length(times)-1)); 
winPnts = round(srate./frex.*linspace(winlenRange(1), winlenRange(2), length(frex)));

% angle difference
datadiff = data1.*conj(data2); 
% alternative:
% phase_diffs = angle(data1) - angle(data2);

% compute ISPC
if byTrialOn
    ISPC = abs(mean(exp(1i*angle(datadiff)), 3));
    % ISPC = abs(mean(exp(1i*phase_diffs), 3));
else
    if gpu2use > 0
         ISPC = zeros(size(datadiff), 'gpuArrya');
    else
         ISPC = zeros(size(datadiff));
    end

    if pbOn, progressbar('Compute ISPC-time...'), end
    for fi = 1:length(frex)
        winPnt = winPnts(fi);
        for ti = winPnt+1:length(times)-winPnt
            ISPC(fi, ti, :) = abs(mean(exp(1i*angle(datadiff(fi, ti-winPnt:ti+winPnt,:))), 2));
            % ISPC(fi, ti, :) = abs(mean(exp(1i*phase_diffs(fi, ti-winPnt:ti+winPnt, :)), 2));    
        end % loop: ti
        if pbOn, progressbar(fi/length(frex)), end
    end % loop: fi
end

% baseline correction
if bcorr
    baselineidx = dsearchn(times', baseline');
    baseline_mat = ISPC(:, baselineidx(1):baselineidx(2), :);
    ISPC = bsxfun(@minus, ISPC, mean(baseline_mat,2));
end

% plotting
if plotOn
    
    % decide trials to plot
    if size(ISPC,3) < 9
        nPlot = size(ISPC, 3);
        trials2plot = 1:nPlot;
    else
        nPlot = 9;
        trials2plot = randi(size(ISPC,3), [1 9]);
    end
    
    % loop over subplots
    figure
    for pi = 1:nPlot
        trialid = trials2plot(pi);
        if nPlot > 1
            subplot(3,3,pi)
        end
        contourf(times,frex,ISPC(:,:,trialid),20,'linecolor','none'); colorbar;
        yticks = ifelse(length(frex)<6, frex, logspace(log10(frex(1)),log10(frex(end)),6));
        set(gca,'yscale', 'log', 'ytick', yticks, 'yticklabel', round(yticks)) 
   
        % baseline frame
        if bcorr
            hold on
            plot([baseline(1) baseline(1) baseline(2) baseline(2)], [frex(1) frex(end) frex(end), frex(1)], 'color', ones(1,3)*0.2)
            hold off
        end
        
        if pi == 1
            xlabel('Time (ms)')
            ylabel('Frequency (Hz)')
        end
              
    end % loop: pi
    
end


end % func