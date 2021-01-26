function overwrite = parSave(f_feat, data, overwrite)

    if exist(f_feat, 'file') ~= 2
        save(f_feat, 'data')
    else
        if ~overwrite  % if overwrite is OFF, ask for permission
            msg = 'Feature has been computed before. Saving the new feature will overwrite previous file.';
            title = 'Confirm Save';
            selection = questdlg(msg,title, 'overwrite', 'cancel', 'cancel');            
            if strcmpi(selection, 'overwrite')
                disp('Feature files will be OVERWRITTEN.')
                overwrite = 1;  % overwrite ON 
            elseif strcmpi(selection, 'cancel')
                error('Session abort.')
            end
        end

        if overwrite
            save(f_feat, 'data', '-append')
        end
    end


end 