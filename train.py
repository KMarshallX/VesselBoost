#!/usr/bin/env python3

"""
Training the chosen model (new pipeline)

Editor: Marshall Xu
Last Edited: 06/19/2023
"""
import train_config
import torch
from tqdm import tqdm
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath("./train/train.py/"))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.module_utils import * 
from utils.new_data_loader import single_channel_loader
from models.unet_3d import Unet
from models.asppcnn import ASPPCNN
from models.siyu import CustomSegmentationNetwork
from models.ra_unet import MainArchitecture
from utils.module_utils import preprocess


def model_chosen(model_name, in_chan, out_chan, filter_num):
    if model_name == "unet3d":
        return Unet(in_chan, out_chan, filter_num)
    elif model_name == "aspp":
        return ASPPCNN(in_chan, out_chan, [1,2,3,5,7])
    elif model_name == "test": # another aspp
        return CustomSegmentationNetwork()
    elif model_name == "atrous":
        return MainArchitecture()
    else:
        print("Insert a valid model name.")

def optim_chosen(optim_name, model_params, lr):
    if optim_name == 'sgd':
        return torch.optim.SGD(model_params, lr)
    elif optim_name == 'adam':
        return torch.optim.Adam(model_params, lr)
    else:
        print("Insert a valid optimizer name.")

def loss_metric(metric_name):
    """
    :params metric_name: string, choose from the following: bce->binary cross entropy, dice->dice score 
    """
    # loss metric could be updated later -> split into 2 parts
    if metric_name == "bce":
        # binary cross entropy
        return BCELoss()
    elif metric_name == "dice":
        # dice loss
        return DiceLoss()
    elif metric_name == "tver":
        # tversky loss
        return TverskyLoss()
    else:
        print("Enter a valid loss metric.")

args = train_config.args
# input images & labels
raw_img = args.inimg
processed_img = args.ps_path
seg_img = args.inlab
prep_mode = args.prep_mode
# when the preprocess is skipped, 
# directly take the raw data for inference
if prep_mode == 4:
    processed_img = raw_img
# loss
loss_name = args.loss_m
metric = loss_metric(loss_name)
#epoch number
epoch_num = args.ep
# hardware config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model configuration
model = model_chosen(args.mo, args.ic, args.oc, args.fil).to(device)
# optimizer
op_name = args.op
optimizer = optim_chosen(op_name, model.parameters(), args.lr)
# set optim scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.optim_gamma, patience=args.optim_patience)
# initialize the augmentation method
aug_item = aug_utils(args.osz, args.aug_mode)


if __name__ == "__main__":
    print("Training session will start shortly..")

    # initialize the preprocessing method with input/output paths
    preprocessing = preprocess(raw_img, processed_img)
    # start or abort preprocessing 
    preprocessing(prep_mode)

    raw_file_list = os.listdir(processed_img)
    seg_file_list = os.listdir(seg_img)
    assert (len(raw_file_list) == len(seg_file_list)), "Number of images and correspinding segs not matched!"
    file_num = len(raw_file_list)
    
    # traning loop (this could be separate out )
    for idx in tqdm(range(file_num)):

        raw_arr_name = processed_img + raw_file_list[idx]
        for i in range(file_num):
            if seg_file_list[i].find(raw_file_list[idx].split('.')[0]) != -1:
                seg_arr_name = seg_img + seg_file_list[i]
                break
        assert (seg_arr_name != None), f"There is no corresponding label to {raw_file_list[idx]}!"
        print(f"Current training image: {raw_arr_name}, current training label: {seg_arr_name}")      
        # initialize single channel data loader
        single_chan_loader = single_channel_loader(raw_arr_name, seg_arr_name, args.osz, args.ep)

        for epoch in tqdm(range(epoch_num)):
            
            image, label = next(iter(single_chan_loader))
            image_batch, label_batch = aug_item(image, label)
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output = model(image_batch)
            loss = metric(output, label_batch)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Learning rate shceduler
            scheduler.step(loss)

            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f'Epoch: [{epoch+1}/{epoch_num}], Loss: {loss.item(): .4f}, Current learning rate: {current_lr: .8f}')

        tqdm.write(f'File number [{idx+1}/{file_num}]')
    print("Training finished! Please wait for the model to be saved!")

    # save the model
    saved_model_path = args.outmo
    torch.save(model.state_dict(), saved_model_path)
    print(f"Model successfully saved! The location of the saved model is: {saved_model_path}")

    
















