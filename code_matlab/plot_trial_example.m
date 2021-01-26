function plot_trial_example(study, sub, trial, coverchan)
% plot the transformations of specified trial 
% study (int), sub(int), trial(str)
% specify the channel shown to be in front by 'coverchan' (doesn't influece ISPC)

    p_main = fullfile('H:', 'Topic_mind wandering', '3data3', 'feats_matfile');
    p_save = fullfile('c:', 'Topic_mind wandering', '4result3','trial_examples');

    feats = {'raw', 'power', 'ispc', 'wst'};
    norms = {'dataset', 'chan', 'off', 'chan'};
    
    figure 
    for fi = 1:length(feats)
        feat = feats{fi};
        norm_unit = norms{fi};

        % load data
        try
            load(fullfile(p_main, num2str(study), pad(num2str(sub),3,'left','0'), trial, [feat,'.mat']))
        catch
            error('Dataset does not exist.')
        end
        
        % load axis values
        load('pars_stFeats.mat', 'chans', 'frex','plocs16_32','chan_pairs', 'tlag', 'scale')  % load within the loop in case later being modified for plotting
        
        % make sure the last dimension is channel (for the following preprocessing)
        if strcmpi(feat,'raw')
            data = permute(data,[2,1]);
        end
        
        % take the specified channel in front (no influence on ISPC)
        ordering = [coverchan setdiff(1:32, coverchan)];
        if ~strcmpi(feat, 'ispc')
            if strcmpi(feat,'raw')
                data = data(:,ordering);
            else
                data = data(:,:,ordering);
            end
            chans = {chans{ordering}};
        end
        
        % normalize
        if ~strcmpi(norm_unit, 'off')
            data = normalize_J(data, norm_unit);
        end

        % swap axes
        if length(size(data)) == 3
            data2plot = permute(data, [3,2,1]);  % freq x time x chan  --> chan x time x freq (horizontal view as imagesc())

            % subset by channels (ispc only)
            if strcmpi(feat,'ispc')
                data2plot = data2plot(plocs16_32,:,:);
            end

            % plot
            subplot(2,2,fi)
            if strcmpi(feat, 'wst')
                times = tlag;
            else
                times = linspace(-400, 1000, size(data2plot,2));
            end
            h = slice(times, 1:size(data2plot,1), ifelse(strcmpi(feat,'wst'),scale,frex), data2plot,[], 1:size(data2plot,1),1);
            set(h,'edgecolor','none')
            colormap(parula)  %[parula, jet]

            % set axis texts
            xlabel(ifelse(strcmpi(feat,'wst'),'Time lag (ms)','Time (ms)'))
            xlim([ifelse(strcmpi(feat,'wst'),0,-400) 1000])

            ylabel(ifelse(strcmpi(feat,'ispc'),'Channel pair','Channel'))
            ylim([1 size(data2plot,1)])
            if strcmpi(feat,'ispc')
                cpairs = {};
                for cpi = 1:120
                    ploc = plocs16_32(cpi);
                    pair = [chan_pairs{ploc,1},'-',chan_pairs{ploc,2}];
                    cpairs{cpi} = pair;
                end
                yticks(1:20:120)
                yticklabels({cpairs{1:20:120}});
            else

                yticks(1:5:32)
                yticklabels({chans{1:5:32}})
            end

            set(gca, 'zScale', 'log')
            if strcmpi(feat,'wst')
                zlim([scale(1) scale(end)])
                z_ticks = round(exp(linspace(log(scale(1)),log(scale(end)),6)));
                zticks(z_ticks)
                zlabel('Scale (ms)')
            else
                zlim([frex(1) frex(end)])
                z_ticks = round(exp(linspace(log(frex(1)),log(frex(end)),6)));
                zticks(z_ticks)
                zlabel('Frequency (Hz)')
            end

        else

            data2plot = permute(data, [2,1]);

            % plot
            times = linspace(-400, 1000, size(data2plot,2));
            subplot(2,2,fi)
            [~,h] = contourf(times, 1:length(chans),data2plot);
            set(h,'LineColor','none')

            % set axis texts
            xlabel('Time (ms)')
            xlim([-400 1000])

            ylabel('Channel')
            ylim([1 32])
            yticks(1:5:32)
            yticklabels({chans{1:5:32}})
            set(gca,'ygrid', 'on', 'yminorgrid', 'on')

        end
    end    
    
    % output
    f_save = [num2str(study),'_', num2str(sub), '_', trial,'.tif'];
    set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 30 18])
    print(gcf, '-dtiff','-r300',fullfile(p_save, f_save))

end



