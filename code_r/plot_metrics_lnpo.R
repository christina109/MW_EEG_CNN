library(stringr)
library(tidyr)
library(ggplot2)
library(plotly)
library(RColorBrewer)
library(ez)
library(rstatix)
library(PerformanceAnalytics)
library(corrplot)

source(file.path('code_r', 'plot_data.r'))


# data to plot
nn_name <- 'simplecnn_5'
randseeds <- c(36,54,87) # c(36, 54, 87) (real experiment) or 27(testing)
#feats <- c('raw','wst', 'power', 'ispc')
feats <- c('raw_M1', 'raw_M2', 'raw_M3',
           'power_M1', 'power_M2', 'power_M3',
           'ispc_M1', 'ispc_M2', 'ispc_M3',
           'wst_M1', 'wst_M2', 'wst_M3')
# feats <- c('raw_sart', 'raw_vs', 'raw_sart_vs', 'raw_vs_sart')

metrics <- c('specificity', 'sensitivity', 'precision', 'accuracy')
metrics2plot <- metrics #c('accuracy')
lnpos <- 0:4
beta <- 1 # f1:beta =1 
err_type <- 'ci'

#hpar <- 'classweights'
#hvals <- c(1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6)
#hvals <- c(1,1.2,1.4,1.6,1.8,2)

#hpar <- 'normalization'
#hvals <- c('off', 'dataset', 'chan', 'freq', 'chanfreq', 'trial', 'signal')

#hpar <- 'window_length'
#hvals <- c(10,15,20,25,30,35,40)

#hpar <- 'neurons_2convs'
#hvals <- c('[256, 256]', '[256, 128]', '[256, 64]', '[256, 32]', 
#           '[128, 128]', '[128, 64]', '[128, 32]', '[128, 16]',
#           '[64, 64]', '[64, 32]', '[64, 16]', 
#           '[32, 32]','[32, 16]',
#           '[16, 16]')
#hvals <- c('[32, 64]', '[32, 64, 128]', '[32, 64, 128, 256]',
#           '[64, 128]', '[64, 128, 256]',
#           '[128, 256]')
#hvals <- c('[64, 64]')
#hpar <- 'leaveout_participant'
#hvals <- 1:30
#hpar <- 'subset_b'
#hvals <- c('N=30', 'N=26','N=18')

#hpar <- 'learningrate'
#hvals <-c('0.001', '0.0005', '0.0001')

#hpar <- 'reg_l1'
#hvals <- c('none','0.1', '0.01', '0.001', '0.0001', '1e-05', '1e-06')

hpar <- 'idv_models'
hvals <- 1:30

#hpar <- 'dropout_rate'
#hvals <- c(0.2,0.5)

#hpar <- 'channel'
#hvals <- 0:119



#full_mean <- 0.5997383 # wst [64,64] kernel33
#full_se <-  0.01891318 # wst [64,64] kernel33
#full_sd <- 0.03782635 # wst [64,64] kernel33

# path setting
f_main <- file.path(getwd(), 'history')
f_res <- file.path('c:', 'topic_mind wandering', '4result3')


# plotting parameters
#colors <- c('#FF8C00', '#3CB371', '#7B68EE')
#colors <- c('#bce784','#5dd39e','#348aa7', '#525174','#513b56')
#colors <- c('#faa275', '#ff8c61', '#ce6a85', '#985277', '#5c374c')
#colors <- c('#f9940e','#d23b2d', '#7f1454', '#320055', '#000005')
colors <- c('#fce51e','#f9952a', '#d74d50', '#9f0f7d', '#550097', '#0b0074')
posn.d <- position_dodge(0.5)


feature <- c()
hyperparameter_value <- c()
lnpo <- c()
specificity <- c()
sensitivity <- c()
precision <- c()
accuracy <- c()
train_acc <- c()
randseed <- c()

# load 
for (feat in feats) {
  
  for (hval in hvals) {
    
    for (seed in randseeds) {
      
      for (li in lnpos){
        
        tp <- try(read.csv(file.path(f_main, paste0(nn_name, '_', feat), paste0(hpar,'_',seed), hval, paste0('lnpo',li,'.csv'))), TRUE)
        if(isTRUE(class(tp)=="try-error")) { next }
        feature <- c(feature, feat)
        hyperparameter_value <- c(hyperparameter_value, hval)
        randseed <- c(randseed, seed)
        lnpo <- c(lnpo,li)
        specificity <- c(specificity, tp[nrow(tp), 'specificity'])
        sensitivity <- c(sensitivity, tp[nrow(tp), 'sensitivity'])
        precision <- c(precision, tp[nrow(tp), 'precision'])
        accuracy <- c(accuracy, tp[nrow(tp), 'val_accuracy'])
        train_acc <- c(train_acc, tp[nrow(tp), 'accuracy'])
      
      }
    }
  }
}


# prepare data
df <- data.frame(feature, hyperparameter_value, randseed, lnpo, 
                 specificity, sensitivity, precision, accuracy, train_acc)
df$fscore_1 <- (1+beta^2)*(df$sensitivity*df$precision)/(df$sensitivity+beta^2*df$precision)
df$fscore_bi <- (1+beta^2)*(df$sensitivity*df$specificity)/(df$sensitivity+beta^2*df$specificity)

df_g <- gather(df, key = 'metric', value = 'val', train_acc, accuracy, specificity, sensitivity, precision, fscore_1, fscore_bi)
# filter ill-defined precisions (no positive predictions)
df_g <- df_g[!c(df_g$metric == 'precision' & df_g$val < 0), ]

df_g$dataset <- ifelse(df_g$metric != 'train_acc', 'val', 'train')
df_g[df_g$metric == 'train_acc', 'metric'] <- 'accuracy'

df_g$hyperparameter_value <- factor(df_g$hyperparameter_value, levels = hvals)
df_g$feature <- factor(df_g$feature, levels = feats)
df_g$metric <- factor(df_g$metric, levels = c(metrics,'fscore_1', 'fscore_bi'))
df_g$dataset <- factor(df_g$dataset, levels = c('val', 'train'))

df_s <- group_by(df_g, feature, metric, hyperparameter_value, dataset)
df_s <- summarise(df_s, mean = mean(val), sd = sd(val), 
                  n = n(), se = sd/sqrt(n-1), ci = se*1.96)

# prepare graph
df2plot <- subset(df_s,metric %in% metrics2plot)
p <- ggplot(df2plot, aes(x = hyperparameter_value, y = mean, group = dataset, 
                      color = metric, fill = metric, shape = dataset, linetype = dataset)) 
p <- p + geom_point(size = 3) 
#p <- p + geom_line(size = 1, alpha = 0.8)

if (err_type == 'se') {
  p <- p + geom_errorbar(aes(ymin = mean -se, ymax = mean+se), width = 0.15, size = 1, linetype = 1)
} else {
  p <- p + geom_errorbar(aes(ymin = mean -ci, ymax = mean+ci), width = 0.15, size = 1, linetype =1)
}

p <- p + geom_hline(aes(yintercept = 0.5), linetype = 2)
#p <- p + geom_hline(aes(yintercept = full_mean), linetype = 2)
#p <- p + geom_hline(aes(yintercept = full_mean+full_se))
#p <- p + geom_hline(aes(yintercept = full_mean-full_se))
if (length(feats) == 1) {
  p <- p + facet_grid(metric~.)
} else if (length(metrics2plot)==1){
  p <- p + facet_grid(feature~.)
} else {
  p <- p + facet_grid(feature~metric)
}
#p <- p + scale_color_manual(values = ifelse(length(metrics2plot)==1, 'blue', colors))
p <- p + scale_y_continuous(breaks = c(0.2, 0.4, 0.6, 0.8,1))
p <- p + labs(title = paste('hyperparameter:', hpar))
p <- p + ylab(ifelse(hpar == 'idv_models', '5fold-cv','lnpocv'))
p <- p + get.mytheme(0)


# prepare saving 
f_save <- file.path(f_res, paste0('lnpo_', nn_name, '_', hpar, '_', err_type, '.tif'))

if (length(feats) == 1) {
  w = 54
  h = (length(metrics)+2)*8
} else {
  w = 54
  h = 8*length(feats)
}

if (file.exists(f_save)) {
  res <- readline(prompt = 'Overwrite?[y/n]')
  if (res == 'y'){
    ggsave(f_save, device = 'tiff', plot = p, dpi = 300, width = w, height = h, units = 'cm')
  }
  else {
    p
  }
} else {
  ggsave(f_save, device = 'tiff', plot = p, dpi = 300, width =w, height = h, units = 'cm')
}



# report datasets with lower performance
if (FALSE){
  df_acc <- subset(df_s, metric == 'accuracy' & feature == 'powerispc')
  
  threshold <- 0.6
  which((df_acc$mean+df_acc$ci >= threshold & df_acc$mean-df_acc$ci <= threshold) 
        | df_acc$mean+df_acc$ci < threshold) 
}

# report the "best" hval (per feat)
if (FALSE) {
  df_acc <- subset(df_s, metric == 'accuracy' & dataset == 'val')
  print(subset(df_acc, mean-ci <= 0.5, select = 'hyperparameter_value'))
  df_acc <- subset(df_acc, mean-ci > 0.5)  # above-cl(=0.5)
  
  n1 <- df_acc$n
  n2 <- df_acc$n
  sd1 <- df_acc$sd
  sd2 <- df_acc$sd
  m1 <- df_acc$mean
  m2 <- 0.5
  s <- sqrt(((n1-1)*sd1^2+(n2-1)*sd2^2)/(n1+n2-2))
  df_acc$d <- (m1 - m2)/s # cohen's d
  
  df_acc <- group_by(df_acc, feature)
  print('Ordered by effect size:')
  print(df_acc %>% top_n(3,d) %>% arrange(desc(d)))
  print('Ordered by mean accuracy:')
  print(df_acc %>% top_n(3,d) %>% arrange(desc(mean)))
  #dsorted <- order(d, decreasing = TRUE)  # sort() returns the value
  #print(dsorted[sort(d, decreasing = TRUE)>-0.5])  # threshold d = 0.5(medium)
}

# summarise accuries across feature/dataset
if (FALSE) {
  err_type <- 'se'
  
  df_acc <- subset(df_s, metric == 'accuracy', select = c(hyperparameter_value,feature, dataset, mean))
  colnames(df_acc)[4] <- 'val' # prevent confusions from the function mean()
  
  df_acc_g <- group_by(df_acc, feature, dataset)
  df_acc_s <- summarise(df_acc_g, mean = mean(val), sd = sd(val),
                        n = n(), se = sd/sqrt(n-1), ci = se*1.96,
                        min = min(val), max = max(val))
  
  df_acc_s <- separate(df_acc_s, col = 'feature', into = c('feature', 'model'), sep = '_')
  df_acc_s$feature <- factor(df_acc_s$feature, levels = c('raw', 'power', 'ispc', 'wst'),
                             labels = c('raw', 'power', 'ISPC', 'stERP'))
  
  p <- ggplot(df_acc_s, aes(x = model, y = mean, 
                            fill = dataset, label = round(mean,2)))
  p <- p + geom_bar(stat = 'identity', position = posn.d, width = 0.4)
  p <- p + geom_text(position = posn.d, vjust = 3, size = 5)
  
  if (err_type == 'se') {
    p <- p + geom_errorbar(aes(ymin = mean -se, ymax = mean+se), width = 0.15, size = 1, linetype = 1, position = posn.d)
  } else {
    p <- p + geom_errorbar(aes(ymin = mean -ci, ymax = mean+ci), width = 0.15, size = 1, linetype =1, position = posn.d)
  }
  
  p <- p + geom_hline(aes(yintercept = 0.5), linetype = 2)
  p <- p + scale_fill_manual(values = c('orange', 'skyblue'))
  p <- p + scale_y_continuous(limits = c(0,1),breaks = c(0.2, 0.4, 0.6, 0.8,1))
  p <- p + ylab('')
  p <- p + xlab('model')
  p <- p + get.mytheme(0)
  p <- p + facet_grid(feature~.)
  p <- p + theme(legend.position="top")
  p
  
  # stats

  val_acc <- subset(df_acc, dataset == 'val')
  val_acc <- separate(val_acc, col = 'feature', into= c('feature','model'), sep = '_')
  
  mano <- ezANOVA(val_acc,dv=val,wid=hyperparameter_value, within=.(feature,model), return_aov = TRUE)
  
  # post-hoc
  res <- val_acc %>% #group_by(model) %>%
    pairwise_t_test(val ~ feature, paired = TRUE, p.adjust.method = "bonferroni")
  
  # report mean
  val_acc %>% group_by(feature) %>% summarise(mean(val))
  
}


# plot train-val scatter dots
if (FALSE) {
  clusteringOn <- TRUE
  
  df_acc <- subset(df_s, metric == 'accuracy', select = c(feature, hyperparameter_value,dataset, mean))
  df2plot <- spread(df_acc, dataset, mean)
  
  if (FALSE) { # for other metric
    df_tp <- subset(df_s, metric == 'fscore_bi', select = c(feature, hyperparameter_value,dataset, mean))
    df2plot <- spread(df_tp, dataset, mean)
  }
  
  df2plot <- separate(df2plot, feature, into = c('feature', 'model'))
  df2plot$feature <- factor(df2plot$feature, levels = c('raw','power','ispc','wst'),
                            labels= c('raw','power','ISPC','stERP'))
  #df2plot <- spread(subset(df_acc, feature %in% models), feature, mean)
  
  if (clusteringOn) {

    #tp <- subset(df2plot, feature == model, select = c(train, val)) 
    tp <- subset(df2plot, select = c(train, val)) 
    #mk <- kmeans(tp, 3)
    #save(mk,file = 'clusters.rdata')
    load('clusters.rdata')
    
    
    #print(model)
    print(round(mk$centers,2))
    #df2plot[df2plot$feature == model, 'cluster'] <- mk$cluster
    df2plot$cluster <- mk$cluster
    df2plot$cluster <- factor(df2plot$cluster)
    
    if (FALSE) { # t.test between clusters
      x1 <- df2plot[df2plot$cluster == 1, 'val']
      x2 <- df2plot[df2plot$cluster == 2, 'val']
      t.test(x1,x2)
      
    }
    
    df2plot %>% group_by(feature, model, cluster) %>% summarise(count = n())
    
    
    if (TRUE){
      
      df2plot$ratio <- rep(c(9.42,  1.57,  2.14,  0.2 ,  2.36,  2.18,  0.81,  0.82,  0.19,
                         0.75,  3.33,  0.08,  0.43,  4.24,  1.85, 23.  , 14.24,  2.35,
                         20.3 ,  0.75,  0.83,  3.09,  0.75,  6.55,  5.48,  2.78,  0.75,
                         21.56,  0.78,  4.4), nrow(df2plot)/30)
      
      p <- ggplot(df2plot, aes(x=cluster, y=log(ratio), color = cluster)) + geom_point()
      p <- p + geom_hline(aes(yintercept = 0), linetype = 2)
      p <- p + facet_grid(feature~model)
      p <- p + get.mytheme(0)
      p
      
      beh <- read.csv('beh_report_1.csv')
      df2plot$sart.rt <- rep(beh[beh$task == 'sart', 'rt'],nrow(df2plot)/30)
      df2plot$sart.acc <- rep(beh[beh$task == 'sart', 'acc'],nrow(df2plot)/30)
      df2plot$vs.rt <- rep(beh[beh$task == 'vs', 'rt'],nrow(df2plot)/30)
      df2plot$vs.acc <- rep(beh[beh$task == 'vs', 'acc'],nrow(df2plot)/30)
      
      p <- ggplot(df2plot, aes(x=cluster, y=log(vs.rt), color = cluster)) + geom_point()
      p <- p + geom_hline(aes(yintercept = 0), linetype = 2)
      p <- p + facet_grid(feature~model)
      p <- p + get.mytheme(0)
      p
      
      
    }
    
  }
    
  
  p <- ggscatter(df2plot, x = 'val', y = 'train', color = ifelse(clusteringOn, 'cluster', 'black'),
  #p <- ggscatter(df2plot, x = models[1], y = models[2], 
                 add = 'reg.line', 
                 add.params = list(color = "blue", fill = "lightgray"),
                 conf.int = TRUE)
  p <- p + stat_cor(method = "spearman", vjust = 7)
  #p <- p + facet_grid(feature~model)
  #p <- p + facet_grid(.~dataset)
  p <- p + scale_y_continuous(limits = c(0.2,1),breaks = c(0.2, 0.4, 0.6, 0.8,1))
  p <- p + scale_x_continuous(limits = c(0.2,1),breaks = c(0.2, 0.4, 0.6, 0.8,1))
  p <- p + get.mytheme(0)
  p
  
  # correlation matrix
  df2cor <- unite(df2plot, col = Model, c('feature','model'), sep = '_')
  #$Model <- factor(df2cor$Model, levels = feats, 
  #                       labels = paste0(rep(c('raw', 'power', 'ISPC', 'stERP'), each = 3), '_', c('M1', 'M2', 'M3')))
  df2cor <- pivot_wider(df2cor, hyperparameter_value, Model, values_from = val)
  
  library(Hmisc)
  res2<-rcorr(as.matrix(df2cor[,2:13]), type = 'spearman')
  corrplot(res2$r, type="upper", tl.col = "black",
           p.mat = res2$P, sig.level = 0.01, insig = "blank",
           addCoef.col = 'black', addCoefasPercent = TRUE)
  #chart.Correlation(df2cor, histogram=TRUE, pch=19, method = 'spearman')
  

}


### compare to the published results ###
# set feats <- c('raw_sart', 'raw_vs'), re-run the script

if (FALSE){ 
  load('subs_study1.rdata') # 'subs'
  tp <- subset(df_s, metric == 'accuracy' & dataset == 'val')
  tp <- tp[tp$hyperparameter_value %in% subs,]
  
  tasks <- c('raw_sart', 'raw_vs', 'raw_sart_vs', 'raw_vs_sart')
  for (task in tasks) {
    tp2 <- subset(tp, feature == task)
    res <- mean(tp2$mean)
    print(paste('task=', task, ' mean=', round(res,3)))
  }
  
}


### summarize stacking models ###

if (FALSE) {  
  smodel_name <- '20201030_173814'  # '20201022_022901'  '20201030_173814'
  folder <- file.path('history', 'stacking_ensemble', smodel_name)
  df <- read.csv(file.path(folder, 'report.csv'))
  
  df_g <- gather(df, key = 'metric', value = 'val', accuracy, specificity, sensitivity, precision)
  # filter ill-defined precisions (no positive predictions)
  df_g <- df_g[!c(df_g$metric == 'precision' & df_g$val < 0), ]
  
  df_g$dataset <- factor(df_g$dataset, levels = c('val', 'train'))
  
  df_s <- group_by(df_g, model, dataset, metric)
  df_s <- summarise(df_s, mean = mean(val), sd = sd(val), 
                    n = n(), se = sd/sqrt(n-1), ci = se*1.96)
  
  # add asps
  if (FALSE) {
    folder <- file.path('history', 'asp',smodel_name)
    df2 <- read.csv(file.path(folder, 'report.csv'))
    
    df_g2 <- gather(df2, key = 'metric', value = 'val', accuracy, specificity, sensitivity, precision)
    # filter ill-defined precisions (no positive predictions)
    df_g2 <- df_g2[!c(df_g2$metric == 'precision' & df_g2$val < 0), ]
    df_g2$model <- factor(df_g2$model, levels = c('lr', 'nn'), labels = c('stacked_lr', 'stacked_nn'))
    df_g2$dataset <- 'asp'
    
    
    df_s2 <- group_by(df_g2, model, dataset, metric)
    df_s2 <- summarise(df_s2, mean = mean(val), sd = sd(val), 
                      n = n(), se = sd/sqrt(n-1), ci = se*1.96)
    

    df_s <- rbind(df_s, df_s2)
    df_s$dataset <- factor(df_s$dataset, levels = c('train', 'val', 'asp'))
  }
  
  # plot
  posn.d <- position_dodge(0.5)
  err_type <- 'se'
  
  p <- ggplot(df_s, aes(x = metric, y = mean, fill = model,label = round(mean,2)))
  p <- p + geom_bar(stat = 'identity', position = posn.d, width = 0.4)
  p <- p + geom_text(position = posn.d, vjust =2, size = 4)
  
  if (err_type == 'se') {
    p <- p + geom_errorbar(aes(ymin = mean -se, ymax = mean+se), width = 0.15, size = 1, linetype = 1, position = posn.d)
  } else {
    p <- p + geom_errorbar(aes(ymin = mean -ci, ymax = mean+ci), width = 0.15, size = 1, linetype =1, position = posn.d)
  }
  
  p <- p + geom_hline(aes(yintercept = 0.5), linetype = 2)
  p <- p + scale_fill_manual(values = c('orange', 'skyblue'))
  p <- p + scale_y_continuous(limits = c(0,1),breaks = c(0.2, 0.4, 0.6, 0.8,1))
  p <- p + ylab('')
  p <- p + xlab('model')
  p <- p + get.mytheme(0)
  p <- p + facet_grid(dataset~.)
  p <- p + theme(legend.position="right")
  p
  
  # stats (model comparisons)
  for (metric in c('accuracy', 'precision', 'specificity', 'sensitivity')) {
    tp <- subset(df_g2, dataset == 'asp')
    tp <- tp[tp$metric == metric,]
    x1 <- tp[tp$model == 'stacked_lr', 'val']
    x2 <- tp[tp$model == 'stacked_nn', 'val']
    res <- t.test(x1,x2, paired = TRUE)
    print(metric)
    print(res)
  }
  
  # stats (to chancel level)
  df_g3 <- rbind(df_g, df_g2)
  cl <- .5
  for (dataset in c('val', 'asp')) {
    for (model in c('stacked_lr', 'stacked_nn')) {
      tp <- df_g3[df_g3$dataset == dataset,]
      tp <- tp[tp$model == model,]
      tp <- tp[tp$metric == 'accuracy',]
      res <- t.test(tp$val, mu = cl)
      print(paste(dataset, model))
      print(res)
    }
  }

  
}




### stacking model comparisons ###

if (FALSE) {  
  smodel_names <- c('20201030_173814',  '20201022_022901')
  metric <- 'sensitivity'
  
  for (v_dataset in c('val', 'asp')){
    if (v_dataset == 'val') {
      folder <- file.path('history', 'stacking_ensemble', smodel_names[1])
      df <- read.csv(file.path(folder, 'report.csv'))
      df <- df[df$dataset == 'val',]
      df <- arrange(df, model, seed, lnpoi)
      eval(parse(text = paste0('x1<-df$',metric)))
      
      folder <- file.path('history', 'stacking_ensemble', smodel_names[2])
      df <- read.csv(file.path(folder, 'report.csv'))
      df <- df[df$dataset == 'val',]
      df <- arrange(df, model, seed, lnpoi)
      eval(parse(text = paste0('x2<-df$',metric)))
      
    } else {
      folder <- file.path('history', 'asp', smodel_names[1])
      df <- read.csv(file.path(folder, 'report.csv'))
      df <- arrange(df, model, seed, lnpoi)
      eval(parse(text = paste0('x1<-df$',metric)))
      
      folder <- file.path('history', 'asp', smodel_names[2])
      df <- read.csv(file.path(folder, 'report.csv'))
      df <- arrange(df, model, seed, lnpoi)
      eval(parse(text = paste0('x2<-df$',metric)))
      
    }
    
    res <- t.test(x1, x2, paired = TRUE)
    print(v_dataset)
    print(res)
  }


  
}



### summarise trial count ###

if (FALSE) {
  
  f_load <- 'trial_count_2.csv'
  
  df <- read.csv(f_load)
  
  df <- df[, c('study', 'task', 'sub', 'n_ot', 'n_mw')]
  df_sart <- df[df$task == 'sart',]
  df_vs <- df[df$task == 'vs',]
  df_sart <- arrange(df_sart, sub)
  df_vs <- arrange(df_vs, sub)
  
  if (!all.equal(df_sart$sub,df_vs$sub)) {print('ERROR: subjects unmatch')}
  
  df_all <- data.frame(sub = df_sart$sub,
                       n_mw = df_sart$n_mw + df_vs$n_mw,  
                       n_ot = df_sart$n_ot + df_vs$n_ot)

  df_all$ntrial <- df_all$n_ot +df_all$n_mw
  df_all$perc_mw <- df_all$n_mw/df_all$ntrial
  
  df_g <- group_by(df_all)
  summarise(df_g, mean_ntrial = mean(ntrial), sd_ntrial = sd(ntrial),
            min_ntrial = min(ntrial), max_ntrial = max(ntrial),
            min_perc = min(perc_mw), max_perc = max(perc_mw),
            median_perc = median(perc_mw))
  
}




