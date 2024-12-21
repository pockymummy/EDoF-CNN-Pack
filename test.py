# =============================================================================
# Create folders with images per stack 
# =============================================================================

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices=['cervix93'], default='fraunhofer_elastic_only')
parser.add_argument('--image_size', choices=[640], default=512)
parser.add_argument('--method', choices=['EDOF_CNN_pack','EDOF_CNN_pack_rgb'], default='EDOF_CNN_pack_rgb')
parser.add_argument('--Z', choices=[3,5,7,9], type=int, default=5)
parser.add_argument('--fold', type=int, choices=range(5),default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cudan', type=int, default=0)
parser.add_argument('--image_channels', choices=['rgb','grayscale'], default='rgb')
parser.add_argument('--post_processing', type=bool, default=False)
args = parser.parse_args()


import numpy as np
from time import time
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import dataset, models
from tqdm import tqdm
from PIL import Image
from matplotlib import cm
import os 
from skimage.exposure import match_histograms

device = torch.device('cuda:'+str(args.cudan) if torch.cuda.is_available() else 'cpu')

#define transforms if rgb or not
if args.image_channels=='rgb':
    train_transform=dataset.aug_transforms_rgb
    test_transform=dataset.val_transforms_rgb
else:
    train_transform=dataset.aug_transforms
    test_transform=dataset.val_transforms

tr_ds = dataset.Dataset('train', train_transform, args.dataset, args.Z, args.fold)
tr = DataLoader(tr_ds, args.batchsize, True,  pin_memory=True)
ts_ds = dataset.Dataset('test', test_transform, args.dataset, args.Z, args.fold)
ts = DataLoader(ts_ds, args.batchsize,False,  pin_memory=True)

#to view images
tst = DataLoader(ts_ds, 1,False,  pin_memory=True)

prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items())
weight_prefix = '-'.join(f'{k}-{v}' for k, v in vars(args).items() if k != "post_processing")
weight_prefix_red = '-'.join(f'{k}-{"red" if k == "image_channels" else "EDOF_CNN_pack" if k == "method" else v}' for k, v in vars(args).items() if k != "post_processing")
weight_prefix_green = '-'.join(f'{k}-{"green" if k == "image_channels" else "EDOF_CNN_pack" if k == "method" else v}' for k, v in vars(args).items() if k != "post_processing")
weight_prefix_blue = '-'.join(f'{k}-{"blue" if k == "image_channels" else "EDOF_CNN_pack" if k == "method" else v}' for k, v in vars(args).items() if k != "post_processing")

if args.method=='EDOF_CNN_pack':
    model = models.EDOF_CNN_pack()
    model.load_state_dict(torch.load('weight/'+str(weight_prefix)+'.pth'))

elif args.method=='EDOF_CNN_pack_rgb':
    model_red = models.EDOF_CNN_pack()
    model_red.load_state_dict(torch.load('weight/'+str(weight_prefix_red)+'.pth'))
    model_green = models.EDOF_CNN_pack()
    model_green.load_state_dict(torch.load('weight/'+str(weight_prefix_green)+'.pth'))
    model_blue = models.EDOF_CNN_pack()
    model_blue.load_state_dict(torch.load('weight/'+str(weight_prefix_blue)+'.pth'))
    model = models.EDOF_CNN_pack_rgb(model_red,model_green,model_blue)
    
else: 
    model = models.EDOF_CNN_concat()


model = model.to(device)


def save_image_stacks():
    Yhats=[]
    Ytrues=[]
    stacks=[]
    model.eval()
    with torch.no_grad():
        for XX, Y in tst:
              XX = [X.to(device) for X in XX]
              Y = Y.to(device, torch.float)
              Yhat = model(XX)
              Yhats.append(Yhat[0].cpu().numpy())
              Ytrues.append(Y[0].cpu().numpy())
              stacks.append([z.cpu().numpy() for z in XX])
              
    for i in range(len(tst)):
        path = 'test_images_for_EDOF\\image_'+str(i)
        os.makedirs(path, exist_ok=True)
        stack = stacks[i]
        for s in range(args.Z):
            stack0 = Image.fromarray(stack[s][0,0,:,:]* 255)
            if stack0.mode != 'RGB':
                stack0 = stack0.convert('RGB')
            stack0.save('test_images_for_EDOF\\image_'+str(i)+'\\image_'+str(i)+'_stack_'+str(s)+'.png')
    
        x = np.moveaxis(Yhats[i], 0,2 )
        xt = np.moveaxis(Ytrues[i], 0,2 )
        x = x[:, :, 0]
        xt = xt[:, :, 0]
        # img = Image.fromarray(np.uint8(x*255), 'RGB')
        img = Image.fromarray(x* 255)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # img.save('image/teste'+str(i)+'_epoch_'+str(epochv)+'.png')
        img.save('test_images_for_EDOF\\image_pred_edof_'+str(i)+'.png')
        # imgt = Image.fromarray(np.uint8(xt*255), 'RGB')
        imgt = Image.fromarray(xt* 255)
        if imgt.mode != 'RGB':
            imgt = imgt.convert('RGB')
        imgt.save('test_images_for_EDOF\\image_full_edof_'+str(i)+'.png')


# save_image_stacks()

# print some metrics 
def predict_metrics(data, model):
    model.eval()
    Phat = []
    Y_true=[]
    input_list =[]
    with torch.no_grad():
        for XX, Y in data:
            XX = [X.to(device, torch.float) for X in XX]
            Y = Y.to(device, torch.float)
            Yhat = model(XX)
            Phat += list(Yhat.cpu().numpy())
            Y_true += list(Y.cpu().numpy())
            numpy_list= []
            for x in XX:
                numpy_list+= list(x.cpu().numpy())
            input_list.append(numpy_list)
    return Y_true, Phat, input_list

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, normalized_root_mse 


data_test = DataLoader(ts_ds, 1,False,  pin_memory=True)
Y_true, Phat, X_list = predict_metrics(data_test, model)

if (args.post_processing):
    print("Post processing")
    color_transfered = []
    temp_list = []
    for x in X_list:
        temp_list.append(x[0])
    ref_list = temp_list
    for pred, ref in zip(Phat, ref_list):
        pred_reshape = np.moveaxis(pred,0,2)
        ref_reshape = np.moveaxis(ref,0,2)
        matched_histogram = match_histograms(pred_reshape, ref_reshape,channel_axis=-1)
        resume_shape = np.moveaxis(matched_histogram,2,0)
        color_transfered.append(resume_shape)
    Phat = color_transfered

mse = np.mean([mean_squared_error(Y_true[i], Phat[i]) for i in range(len(Y_true))])
rmse = np.mean([normalized_root_mse(Y_true[i], Phat[i]) for i in range(len(Y_true))])
ssim =np.mean([ssim(Y_true[i], Phat[i],channel_axis=0, data_range = 1) for i in range(len(Y_true))]) 
psnr =np.mean([peak_signal_noise_ratio(Y_true[i], Phat[i]) for i in range(len(Y_true))]) 


f = open('results/'+ str(prefix)+'.txt', 'a+')
f.write('\n\nModel:'+str(prefix)+
    ' \nMSE:'+ str(mse)+
    ' \nRMSE:'+ str(rmse)+
    ' \nSSIM:'+str(ssim)+
    ' \nPSNR:'+ str(psnr))
f.close()
print("Write to " + 'results/'+ str(prefix)+'.txt')
