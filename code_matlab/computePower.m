function power = computePower(data, times, baselineCorr, singleTrialOn, plotting, parsPrintOn)
% power = computePower(data, times, baselineCorr, singleTrialOn, plotting, parsPrintOn)
% data: complex data of nFreq(nChan) x nPnt x nTrial
% times: a time point vector with length of nPnt; default to 0:nPnt-1
% baselineCorr: {'baseline', 'corrType'} cell to specify the baseline
%               corretion, default to {} (no correction). This par can only
%               work when TIMES has been specified
% - baseline: baseline time window
% - corrType: correction method: 'log' (default), '%', 'z'
% singleTrialOn: single-trial power (default) or trial-averaged power
% plotting: {'plotOn', 'frex'} cell to specity the plottiing settings,
%           default to {} (no plot).
% - plotOn: turn ON/OFF plotting(if singleTrialOn plot random 9 trials)
% - frex: the frequency range of the first dimention of the input data
% parsPrintOn = 1: to turn on/off printing the associated settings.
% power: the turn is a power matrix of nFreq(nChan) x nPnt x nTrial (singleTrialOn)
%        or of nFreq(nChan) x nPnt (singleTrialOFF)

% default
if nargin < 2 || isempty(times) || length(times) ~= size(data,2)
    disp('No/Invalid TIME specified.')
    disp(['Default to 0 ~ ', num2str(size(data,2)), ' (SAMPLING POINTS).'])
    pntOn = 1; % sampling point 
else
    pntOn = 0; % time point
end

if nargin < 3 || isempty(baselineCorr) 
    if parsPrintOn, disp('Baseline Correction is OFF.'), end
    bcorr = 0;
elseif ~iscell(baselineCorr) && (isvector(baselineCorr) && isnumeric(baselineCorr))
    tp = baselineCorr;
    baselineCorr = {};
    baselineCorr{1} = tp;
    bcorr = 1;
elseif ~iscell(baselineCorr)
    disp('Invalid BaselineCorr is specified.')
    disp('Baseline Correction is OFF.')
    bcorr = 0;
else
    bcorr = 1;
end

if nargin < 4 || isempty(singleTrialOn) || ~ismember(singleTrialOn, [0 1])
    disp('No/Invalid SingleTrial Mode setting. Use default.')
    singleTrialOn = 1;
end

if nargin < 5 || isempty(plotting) 
    if parsPrintOn, disp('Plotting is OFF.'), end
    plotOn = 0;
elseif ~iscell(plotting) && ismember(plotting, [0 1])
    plotOn = plotting;
    plotting = {};
    plotting{1} = plotOn;
elseif ~iscell(plotting)
    disp('Invalid Plotting setting. Plotting is OFF.')
    plotOn = 0;
else
    plotOn = plotting{1};
end

if nargin < 6 || isempty(parsPrintOn) || ~ismember(parsPrintOn, [0 1])
    parsPrintOn = 1;
end

% legacy checking
if isreal(data)
    error('Data must have an IMAGINARY part')
end

if bcorr == 1
    
    % subset pars
    baseline = baselineCorr{1};
    baseline = [min(baseline) max(baseline)]; % formatting
    if length(baselineCorr) > 1
        corrType = baselineCorr{2};
    else
        disp('No correction type specified. Use default.')
        corrType = 'log';
    end
    
    % legacy check
    if ~isnumeric(baseline) && baseline(2)==baseline(1)
        disp('Invalid baseline specified.')
        disp('Baseline Correction is OFF.')
        bcorr = 0;
    end
    if ~ismember(lower(corrType), {'z','log','%'})
        disp('Invalid correction type specified. Use default.')
        corrType = 'log';
    end
end

if bcorr 
    if pntOn 
        error('Baseline correction CANNOT be performed with No TIMES specified!')
    end
    if parsPrintOn
        disp(['Baseline Correction is ON.'])
        disp(['Baseline is [', num2str(baseline(1)), ' ', num2str(baseline(2)), '].'])
        disp(['Correction Method is ', upper(corrType),'-tranformation.'])
    end
end

if parsPrintOn, disp(ifelse(singleTrialOn, 'Compute Single-Trial power.', 'Compute Trial-Averaged power.')), end

% raw power
power = data.*conj(data);

% baseline correction
if bcorr
    basePnt = dsearchn(times', baseline'); % baselne idx
    baseline_mat = power(:,basePnt(1):basePnt(2),:);
    
    baseline_mean = mean(baseline_mat,2);
    baseline_sd = std(baseline_mat, [], 2);
    if strcmpi(corrType, 'log')
        powerCorr = 10*log10(bsxfun(@rdivide, power, baseline_mean)); 
    elseif strcmpi(corrType, '%')
        powerCorr = 100*bsxfun(@rdivide, bsxfun(@minus, power, baseline_mean), baseline_mean);
    elseif strcmpi(corrType, 'z')
        powerCorr = bsxfun(@rdivide, bsxfun(@minus, power, baseline_mean), baseline_sd);
    end
    power = powerCorr;
end

% average across trials
if ~singleTrialOn
    power = mean(power,3);
end

if plotOn
    % decide frequency range
    if length(plotting) == 1 
        disp('No frequency range has been specified. Use Frequency Point as Y-axis.')
        fpntOn = 1;
    else 
        frex = plotting{2};
        if ~(isvector(frex) && isnumeric(frex)) || length(frex) ~= size(power,1)
            disp('Invaid frequency range has been specified. Use Frequency Point as Y-axis.')
            fpntOn = 1;
        else
            fpntOn = 0;
        end
    end
        
    if fpntOn 
        frex = 1:size(power,1);
    end
    
    % decide trials to plot
    if size(power,3) < 9
        nPlot = size(power, 3);
        trials2plot = 1:nPlot;
    else
        nPlot = 9;
        trials2plot = randi(size(power,3), [1 9]);
    end
    
    % loop over subplots
    figure
    for pi = 1:nPlot
        trialid = trials2plot(pi);
        if nPlot > 1
            subplot(3,3,pi)
        end
        contourf(times,frex,power(:,:,trialid),40,'linecolor','none'); colorbar;
        yticks = ifelse(length(frex)<6, frex, logspace(log10(frex(1)),log10(frex(end)),6));
        set(gca,'yscale', 'log', 'ytick', yticks, 'yticklabel', round(yticks)) 
   
        % baseline frame
        if bcorr
            hold on
            plot([baseline(1) baseline(1) baseline(2) baseline(2)], [frex(1) frex(end) frex(end), frex(1)], 'color', ones(1,3)*0.2)
            hold off
        end
        
        if pi == 1
            xlabel(ifelse(pntOn, 'Time sampling point', 'Time (ms)'))
            ylabel(ifelse(fpntOn, 'Frequency sampling point', 'Frequency (Hz)'))  
        end
               
    end % loop: pi
    
    % graph setting
    if ~bcorr 
    suptitle('Raw power (\muV^2)')
    else
        if strcmpi(corrType, 'log')
            suptitle('Baseline normalized power: db')
        elseif strcmpi(corrType, 'z')
            suptitle('Baseline normalized power: z-score')
        elseif strcmpi(corrType, '%')
            suptitle('Baseline normalized power: %')
        end
    end
                        
end  % end of plotting
    

end % func

