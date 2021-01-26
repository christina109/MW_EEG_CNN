function convres = waveletConv(data, srate, fRange, nFreq, nCycleRange, scaling, waveletCheckOn, filteringCheckOn, recoverCheckOn, parsPrintOn, gpu2use)
% convres = waveletConv(data, srate, fRange, nFreq, nCycleRange, scaling, waveletCheckOn, filteringCheckOn, recoverCheckOn, parsPrintOn)
% Wavelet convolution with the input EEG data.
% data: SINGLE CHANNEL EEG data formatting of nPnt x nTrial
% srate: the srate of the input EEG
% fRange = [1 80]: frequency range of interest
% nFreq = fRange(2)-fRange(1)+1: the number of frequencies
% nCycleRange = [3 10]: number range of the cycles of the guassian typer 
% scaling = 'log': 'linear' or 'log'/'logrithmic' increase the frequency/cycle
%          number 
% waveletCheckOn = 0: turn on/off the visual inspecting of the wavelets
% filteringCheckOn = 0: turn on/off the visual inspecting of the filtered
%                       signal at each frequency
% recoverCheckOn = 0: turn on/off the recovery of the signal
% parsPringOn = 1: turn on/off printing the wavelet parameters
% convres: 
% - .data: complex mat of nFreq x nPnt x nTrial
% - .frex: frequency vector length of nFreq

% defaults
if nargin < 3 || ~isnumeric(fRange) || isempty(fRange)
    fRange = [1 80];
end
fRange = [min(fRange) max(fRange)]; % format fRange in case of wierd input

if nargin < 4 || mod(nFreq,1)~=0 || isempty(nFreq) || nFreq < 2
    nFreq = fRange(2)-fRange(1)+1;  % frequency size
    
end

if nargin < 5 || ~isnumeric(nCycleRange) || isempty(nCycleRange)
    nCycleRange = [3 10];
end
nCycleRange = [min(nCycleRange), max(nCycleRange)];

if nargin < 6 || ~(ischar(scaling) && ismember(lower(scaling), {'log', 'logrithmic', 'linear'}))
    scaling = 'log';
end
scaling = lower(scaling); % formatting

if nargin < 7 || isempty(waveletCheckOn) || ~ismember(waveletCheckOn, [0 1])
    waveletCheckOn = 0;
end

if nargin < 8 || isempty(filteringCheckOn)|| ~ismember(filteringCheckOn, [0 1])
    filteringCheckOn = 0;
end

if nargin < 9 || isempty(recoverCheckOn) || ~ismember(recoverCheckOn, [0 1])
    recoverCheckOn = 0;
end

if nargin < 10 || isempty(parsPrintOn) || ~ismember(parsPrintOn, [0 1])
    parsPrintOn = 1;
end

if nargin < 11 || isempty(gpu2use) || ~isnumeric(gpu2use)
    gpu2use = 0;
end


% pars
time = [-3 3]; % in seconds, for wavelet
nCheck = 11;   % both ends included

% print wavelet parameters
if parsPrintOn
    disp(['Frequency Range: ', num2str(fRange(1)), ' to ', num2str(fRange(2)), ' Hz'])
    disp(['Number of Frequencies: ', num2str(nFreq)])
    disp(['Number of Cycles of the Gaussian Taper: ', num2str(nCycleRange(1)), ' to ', num2str(nCycleRange(2))])
    disp(['Scaling Method: ', upper(scaling)])
end

% check input legacy
if length(size(data)) == 3 && size(data,1) == 1
    if parsPrintOn
        disp('Detect the input data is a possbile single channel signal.')
        disp('Convert to the legal format.')
    end
    data = squeeze(data);
elseif length(size(data)) ~= 2
    error('Input data should be from SINGLE CHANNEL and of nPnt x nTrial formmatting!')
elseif length(size(data)) == 2 && size(data,1) == 1 && length(data) > 1
    if parsPrintOn
        disp('Detect the input data is a possbile single channel signal.')
        disp('Convert to the legal format.')
    end
    data = data';
end



% data dimension
[nPnt, nTrial] = size(data);

% define wavelets
% log increase
if strcmpi(scaling, 'log') || strcmpi(scaling, 'logarithm')
    frex  = logspace(log10(fRange(1)), log10(fRange(2)), nFreq);
    s     = logspace(log10(nCycleRange(1)),log10(nCycleRange(2)),nFreq)./(2*pi*frex);
% linear increase
elseif strcmpi(scaling, 'linear') 
    frex  = linspace(fRange(1), fRange(2), nFreq);
    s     = linspace(nCycleRange(1), nCycleRange(2), nFreq)./(2*pi*frex);    
end

    
% decide checking pars
if waveletCheckOn || filteringCheckOn 
    if nFreq >= nCheck
        checkNow = zeros(1,nFreq);
        checkFrexPnt = linspace(fRange(1), fRange(2), nCheck);
        checkNow(dsearchn(frex', checkFrexPnt')) = 1;
    else
        checkNow = ones(1,nFreq);
    end
end   
    
% define convolution pars
timeline = time(1):1/srate:time(2); if gpu2use > 0; timeline = gpuArray(timeline); end
n_wavelet = length(timeline);
n_data = nPnt * nTrial;
n_convolution = n_wavelet + n_data-1;
n_conv_pow2 = pow2(nextpow2(n_convolution));
half_of_wavelet_size = (n_wavelet-1)/2;

% get FFT of data
if gpu2use == 0
    data2fft = reshape(data, 1, n_data);
else
    % note reshape() does not suppor gpu array
    data2fft = zeros([1,n_data], 'gpuArray');
    for triali = 1:nTrial
        data2fft((1:nPnt)+(triali-1)*nPnt) = data(:,triali);
    end
end
eegfft = fft(data2fft,n_conv_pow2); 
            
% initialize
if gpu2use == 0
    dataConv = zeros([nFreq size(data)]); 
else 
    dataConv = zeros([nFreq size(data)], 'gpuArray');
end
        
% loop over frequencies
for fi = 1:nFreq
        
    % build wavelet
    sine_wave = exp(2*1i*pi*frex(fi).*timeline);
    gaussian_win = exp(-timeline.^2./(2*(s(fi)^2)));
    wavelet = sine_wave .* gaussian_win;
    % wavelet = sine_wave .* gaussian_win .* sqrt(1/(s(fi)*sqrt(pi))) ;  % scaling factor behind? based on Figure 13.11
    
    % check the taper
    if waveletCheckOn && checkNow(fi)
        figure
        plot(timeline, real(wavelet),'k', 'linewidth', 2)
        hold on
        plot(timeline, imag(wavelet), '--r')
        title([num2str(frex(fi)), ' Hz'])
    end
    
    % get FFT of wavelet
    wfft = fft(wavelet, n_conv_pow2);  %ref to eq 13.9, 13.10
                
    % convolution
    eegconv = ifft(wfft.*eegfft);
    % eegconv = ifft(wfft.*eegfft)*sqrt(s(fi))/10;  % temporay scaling factor for comparison with Figure 12.5
    eegconv = eegconv(1:n_convolution);
    eegconv = eegconv(half_of_wavelet_size+1:end-half_of_wavelet_size);
                
    % register
    if nTrial > 1
        dataConv(fi,:,:) = reshape(eegconv,nPnt,nTrial);
    else
        dataConv(fi,:,:) = eegconv;
    end
    
    
    % randomly chose 10 epochs to show the effect
    if (filteringCheckOn && checkNow(fi)) || recoverCheckOn         
        if nTrial > 10 
            ts2show = randi(nTrial, [10 1]);
        else
            ts2show = 1:nTrial;
        end
    end
    
    % check filtering result 1/2
    if filteringCheckOn && checkNow(fi)
        figure 
        for ti = 1:length(ts2show)
            subplot(5,2,ti)
            tid = ts2show(ti);
            plot(normalizeK(data(:,tid), 'range', 0, 0))   % normalize for the ease of visual comparison 
            hold on
            dataFilt = squeeze(real(dataConv(fi, :, tid)));
            plot(normalizeK(dataFilt, 'range', 1, 0), 'r--')
        end  
        suptitle('X-axis is time point')
    end      
                
end % fi

% check filtering result 2/2
if recoverCheckOn
    figure 
    for ti = 1:length(ts2show)
        subplot(5,2,ti)
        tid = ts2show(ti);
        plot(normalizeK(data(:,tid), 'range', 0, 0))   % normalize for the ease of visual comparison
        hold on
        fStep = (fRange(2) - fRange(1))/(nFreq - 1);
        dataFilt = squeeze(sum(real(dataConv(:, :, tid)),1))*fStep;
        plot(normalizeK(dataFilt, 'range', 1, 0), 'r--')
    end  
    suptitle('Signal recover!')
end

convres.data = dataConv;
convres.frex = frex;

end