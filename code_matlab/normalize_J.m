function data_new = normalize_J(data, unit)
% normalize the input matrix in the specified unit
% frequency must be the first dimension
% channel must be the last dimension

    if strcmpi(unit, 'dataset')
        min_val = min(data(:));
        max_val = max(data(:));
        
        data_new = (data-min_val)/(max_val - min_val);
        
    elseif strcmpi(unit, 'chan')
        data_size = size(data);
        data_new = zeros(data_size);
        
        if length(data_size) == 3
            for chani = 1:data_size(3)
                tp = data(:,:,chani); 
                min_val = min(tp(:));
                max_val = max(tp(:));
                tp_new = (tp-min_val)/(max_val-min_val);
                data_new(:,:,chani) = tp_new;
            end
        
        else 
            for chani = 1:data_size(2)
                tp = data(:,chani); 
                min_val = min(tp(:));
                max_val = max(tp(:));
                tp_new = (tp-min_val)/(max_val-min_val);
                data_new(:,chani) = tp_new;
            end
        end
        
    end
        

end