import os
import argparse
from datetime import datetime
import numpy as np 
import pandas as pd
import torch 
import torchvision
import time 
import timm 
import torch.optim as optim 
import torch.nn as nn 
import sys
from tqdm import tqdm 
sys.path.append('/Code/DataCode/')
path_to_result = "/Results/"

from Utils import utils, eval_metrics
from DataCode.data_load import dataloader, set_data_path

if __name__ == '__main__':
    __file__ = os.path.abspath('')

    # Path to the dataset directory
    set_data_path(f'{__file__}/Data')

    parser = argparse.ArgumentParser()
    
    # Graphical Representation Variables
    parser.add_argument('--plottype', default="bar_z_score", type=str,
                        choices = ["pie_min_max", "bar_z_score", "bar_nan_mark_z_score", "bar_min_max"], 
                        # pie_min_max - pie chart with min-max normalization
                        # bar_z_score - bar plot without nan marking with z-score normalization
                        # bar_nan_mark_z_score - bar plot with nan marking with z-score normalization (with matplotlib plotting)
                        # bar_min_max - bar plot without nan marking with min-max normalization
                        help='The type of graphical representation.')
    parser.add_argument('--spacing', default=0.3, type=float, help='fraction of spacing for bar faster')
    parser.add_argument('--vert',default=True, type=bool, help='whether plot orientation is vertical(True) or horizontal(False)') # Always True
    
    # Data Variables
    parser.add_argument('--dataname', default = "dry_bean", type = str,
                        choices = ["dry_bean", "gas"],
                        help='The name of the data')
    parser.add_argument('--p', default = 0.5, type = float, help = "fraction of data available for the model")
    
    # Model Variables
    parser.add_argument('--modelname', default = "res34", type = str,
                        choices = ["res34", "vit_small"],
                        help = "The name of the model used")
    
    # Training Variables
    parser.add_argument('--seed', default=2024, type=int, help='Seeding Number')
    parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate')
    parser.add_argument('--save',default=False, type=bool, help='whether to save model state(True) or not(False)')
    parser.add_argument('--load',default=False, type=bool, help='whether to load model state(True) or not(False)')
    parser.add_argument('--nruns', default=2, type=int, help='Number of runs')
    
    args = parser.parse_args()
    
    seed = args.seed
    plot_type = args.plottype
    data_name = args.dataname
    p_available = args.p
    model_name = args.modelname
    lr = args.lr
    vert = args.vert
    save = args.save
    load = args.load
    nruns = args.nruns
    spacing = args.spacing
    utils.seed_everything(seed)
    
    # load data, colors corresponding to features and reverse mask (for nan-marking style plotting)
    # drop_df: dataframe with missing values dropped, labels: target labels, mat_rev_mask: reverse mask for plotting, colors: colors for features
    # data_folder: name of the dataset, p_available: fraction of data available for training, seed: seed for reproducibility and consistency of colors
    drop_df, labels, num_class, mat_rev_mask, colors = dataloader(data_folder = data_name, p_available = p_available, seed = 42)
    
    num_inst=drop_df.shape[0]
    num_feats=drop_df.shape[1]
    '''
    
    For ease of coding, we determine the number of features from the data. However, note that this information is not needed.
    This code can be very easily adapted to work with any number of features. The paper HI2 does not require the number of features to be known.
    
    '''
    results = []
    for n in range(nruns):
        utils.seed_everything(n)
        if model_name=='res34':
            model=torchvision.models.resnet34(weights='IMAGENET1K_V1')
            model.fc=nn.Linear(model.fc.in_features, num_class)
        elif model_name=='vit_small':
            model=timm.create_model('vit_small_patch16_224',pretrained=True)
            model.head=nn.Linear(model.head.in_features, num_class)

        # criterion=nn.BCEWithLogitsLoss() # binary cross entropy loss with logits
        criterion=nn.CrossEntropyLoss() # cross entropy loss with logits
        optimizer=optim.Adam(model.parameters(),lr=lr)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        model=model.to(device) 

        # create lists for storing various data for evaluation and analysis
        # loss_history=[]
        # loss=0.0
        preds=[]
        pred_logits=[]
        true=[]
        # acc_history=[]
        # f1_history=[]

        feat=np.arange(num_feats)
        min_arr=[np.nan]*num_feats
        min_arr=np.array(min_arr)
        max_arr=np.copy(min_arr)
        
        minmax_time=0
        plot_time=0
        model_time=0
        predict_time=0
        model.train()
        run_sum=np.zeros(num_feats)
        sum_sq=np.zeros(num_feats)
        count=np.zeros(num_feats)

        tic = time.time()
        for k in tqdm(range(num_inst)):

            row = drop_df[k]
            rev = mat_rev_mask[k]                         # fetch data to be plotted
            label = labels[k]
            label = torch.tensor(label)
            
            start1 = datetime.now()

            if(plot_type == 'bar_z_score' or plot_type == 'bar_nan_mark_z_score'): # bar_z_score and bar_nan_mark_z_score is meant for z-score normalization based bar plots
                norm_row , run_sum, sum_sq, count = utils.zscore(row, run_sum, sum_sq, count)
            else:
                norm_row, min_arr, max_arr = utils.minmaxnorm(row,min_arr,max_arr,epsilon=1e-15)    # normalize and update min, max
            end1 = datetime.now()
            diff1 = (end1 - start1).total_seconds()
            minmax_time += diff1

            start2 = datetime.now()
            if plot_type=='bar_nan_mark_z_score':
                img=utils.bar_nan_mark_z_score_plot(norm_row,rev,colors,feat,vert,dpi=56)        #obtain bar plot tensor 
            elif plot_type== 'bar_min_max':
                img=utils.bar_min_max_plot(norm_row, colors, spacing)
            elif plot_type== 'pie_min_max':
                img=utils.pie_min_max_plot(norm_row, colors)
            elif plot_type== 'bar_z_score':
                img=utils.bar_z_score_plot(norm_row, colors)    
                
            
            end2=datetime.now()
            diff2 = (end2 - start2).total_seconds()
            plot_time += diff2


            img, label = img.to(device), label.to(device)      #transfer to GPU
            img = torch.reshape(img,(-1,3,224,224))      # add extra dimension corresponding to batch, as required by model
            with torch.no_grad():
                start3 = datetime.now()
            optimizer.zero_grad()
            
            outputs = model(img)
            
                
            outputs = torch.reshape(outputs,(-1,num_class))
        
            
            loss=criterion(outputs,label.long())         # compute loss
        
            loss.backward()                               
            # loss_history += [loss.item()]                 # record loss
            
            optimizer.step()
                
        
            with torch.no_grad():

                end3 = datetime.now()
                diff3 = (end3 - start3).total_seconds()
                model_time += diff3


                start4 = datetime.now()

                predicted = torch.softmax(outputs, dim=-1)   # since we use CrossEntropyLoss, softmax is required for obtaining probability
                predicted = torch.argmax(predicted)          # get binary prediction
                
                predicted = predicted.to('cpu')
                label = label.to('cpu')
                outputs = outputs.to('cpu')                 # transfer these to cpu to evaluate using sklearn based metrics
                
                
                pred_logits += [outputs.tolist()]                    # update lists
                preds.append(predicted.item())
                true.append(label.item())
                end4 = datetime.now()
                diff4 = (end4 - start4).total_seconds()
                predict_time += diff4
        toc = time.time()
        
        # b_acc = eval_metrics.BalancedAccuracy(true,preds)
        # auroc = eval_metrics.AUROC(true,pred_logits)
        # auprc = eval_metrics.AUPRC(true,pred_logits)

        metrics = eval_metrics.get_all_metrics(true, preds, pred_logits, time_taken=toc-tic)

        # if save:
        #     torch.save(model, path_to_result+model_name+f'/{data_name}-{p_available}.pth')

        res_dict = {
            'seed'  : seed,
            'plot'  : plot_type,
            'data'  : data_name,
            'p'     : p_available,
            'model' : model_name,
            'lr'    : lr,

            'norm_time'     : minmax_time,
            'plot_time'     :  plot_time,
            'model_time'    :  model_time,
            'predict_time'  :  predict_time,
        }
        del model, criterion, optimizer
        res_dict.update(metrics)

        results.append(res_dict)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{path_to_result}/{model_name}/{data_name}-{p_available}.csv")
