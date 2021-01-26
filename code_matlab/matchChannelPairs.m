%matchChannelpairs

function plocs = matchChannelPairs(chans, chanlist)

    pairs = generatePairs(chanlist);
    
    nChan = length(chans);
    plocs = zeros(nChan*(nChan-1)/2, 1);
    for chani = 1:nChan-1
        for chanj = chani+1:nChan
            matcharr = strcmpi(chans{chani}, {pairs(:).left}) & strcmpi(chans{chanj}, {pairs(:).right});
            if sum(matcharr) == 0  % if not match, looking for the reversed order
                matcharr = strcmpi(chans{chanj}, {pairs(:).left}) & strcmpi(chans{chani}, {pairs(:).right});
            end
            plocs(sum(nChan-1:-1:nChan-chani+1)+chanj-chani) = find(matcharr);
        end
    end
    
    if any(plocs==0)
        warning('Unmatched channels found')
    end
end




% generate chan_pair struct using the specified 'chans' (cell)
function pairs = generatePairs(chans)

    pairs = struct();
    nChan = length(chans);
    for chani = 1:nChan-1
        for chanj = chani+1:nChan
            pairs(sum(nChan-1:-1:nChan-chani+1)+chanj-chani).left = chans{chani};
            pairs(sum(nChan-1:-1:nChan-chani+1)+chanj-chani).right = chans{chanj};
        end
    end
end
