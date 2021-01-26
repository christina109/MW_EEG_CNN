function [flocs, frex_new] = matchFrequencies(fRange, nFreq_old, logSpcOn, nFreq_new)

if logSpcOn
    frex  = logspace(log10(fRange(1)), log10(fRange(2)), nFreq_old);    
else
    frex  = linspace(fRange(1), fRange(2), nFreq_old);
end

flocs = dsearchn([1:nFreq_old]',linspace(1,nFreq_old,nFreq_new)');
frex_new = frex(flocs);


end