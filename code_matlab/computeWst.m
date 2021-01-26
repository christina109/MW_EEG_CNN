function [Wst, tlag, scale] = computeWst(data, times, plotOn, pbOn, gpu2use)
% [Wst, tlag, scale] = compute_Wst(data, times, plotOn = 1, pbOn = 1)
% Convolving the input single-trial single-channel EEG data with 
% the Mexican Hat function
% data: single trial EEG of lenght of nPnt
% times: time series (in ms) of the input data of length of nPnt
% plotOn: turn ON (defaul)/OFF plotting the convolutional result
% pbOn: turn ON (default)/OFF progress bar
% NOTES. This function will run largely slower when on a GPU.

% default
if nargin < 3 || isempty(plotOn) || ~ismember(plotOn, [0 1])
    plotOn = 1;
end
if nargin < 4 || isempty(pbOn) || ~ismember(pbOn, [0 1])
    pbOn = 1;
end

% legacy
if ~isequal(length(data),length(times))
    error('Data and times must be the same size!')
end

if size(data,2) ~= length(data)
    data = data';
    if size(data,1) ~= 1
        error('Input must be a vector.')
    end
end

% pars
srate = computeSrate(times);
% linear sampling
tlag  = 0:1000/srate:times(end); 
%scale = 1:1000/srate:2500; % using the same srate might not be necessary, but it can save storage
scale = logspace(log10(1),log10(2500),300);

% intialize
if gpu2use == 0
    Wst = zeros(length(scale),length(tlag));
else
    times = gpuArray(times);
    tlag = gpuArray(tlag);
    scale = gpuArray(scale);
    Wst = zeros(length(scale),length(tlag), 'gpuArray');
end

if pbOn
    progressbar('Convolution with the Mexican Hat.')
end

% loop over t/s combo
for ti = 1:length(tlag)
    t = tlag(ti);
    for si = 1:length(scale)
        s = scale(si);
        wavelet = mexican_hat(times,t,s);          
        %Wst(si, ti) = trapz(data.*wavelet).* (1000/srate) ./sqrt(s);
        Wst(si,ti) = sum(data .* wavelet)*(1000/srate)/sqrt(s);
    end
    
    if pbOn
        progressbar(ti/length(tlag))
    end
end

% plot
if plotOn == 1
    figure
    contourf(tlag,scale,Wst, 40, 'linecolor', 'none')
    set(gca,'YDir','normal')
    ylabel('Scale (ms)')
    xlabel('Time lag (ms)')
    caxis([-1 1]*max(abs(Wst(:))))
    colorbar
end
        

end % func


function ps = mexican_hat(times, lag, scale)

    t = (times-lag)./scale;
    ps = (1- 16*t.^2).*exp(-8*t.^2);

end % func

function srate = computeSrate(times)
% To computes sampling rate based on the given time series
% times: ms

    step = (times(end)-times(1))/(length(times)-1);
    srate = round(1000/step);

end