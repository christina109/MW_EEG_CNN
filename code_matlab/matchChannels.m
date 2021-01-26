function locs = matchChannels(chans, chanlist)
% return the location id of each channel from the 'chans' in the 'chanlist'

locs = zeros(1,length(chans));

for ci = 1:length(chans)
    
    [~, locs(ci)] = find(strcmpi(chans{ci}, chanlist));
    
end  % loop: ci (chans)

if any(locs == 0)
    msg2print = 'Unmatched channels: ';
    umchans = chans(locs == 0);
    for umi = 1:length(umchans)
        msg2print = [msg2print, umchans{umi}, ' '];
    end
    
    warning(msg2print)

end


end