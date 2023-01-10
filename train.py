"""
Training the chosen model 

Editor: Marshall Xu
Last Edited: 01/10/2023
"""

import config
import torch
from tqdm import tqdm

from utils.unet_utils import *
from utils.data_loader import data_loader
from torch.utils.data import DataLoader
from models.unet_3d import Unet
from models.test_model import test_mo
from models.asppcnn import ASPPCNN


def model_chosen(model_name, in_chan, out_chan, filter_num):
    if model_name == "unet3d":
        return Unet(in_chan, out_chan, filter_num)
    elif model_name == "test_mo":
        return test_mo(in_chan, out_chan, filter_num)
    elif model_name == "aspp":
        return ASPPCNN(in_chan, out_chan, [1,2,3,5,7])
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
        return BCELoss()
    elif metric_name == "dice":
        return DiceLoss()
    else:
        print("Enter a valid loss metric.")


if __name__ == "__main__":
    args = config.args

    # hardware config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input images & labels
    raw_img = args.inimg
    seg_img = args.inlab
    
    # model configuration
    model = model_chosen(args.mo, args.ic, args.oc, args.fil).to(device)

    # training configuration
    batch_size = args.bsz
    d_loader = data_loader(raw_img, seg_img, args.psz, args.osz, args.pst)

    # loss
    loss_name = args.loss_m
    metric = loss_metric(loss_name)

    # optimizer
    op_name = args.op
    optimizer = optim_chosen(op_name, model.parameters(), args.lr)
    # set optim scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    #epoch number
    epoch_num = args.ep

    # step number
    step_num = len(d_loader) // batch_size


    # traning loop (this could be separate out )
    for epoch in tqdm(range(epoch_num)):
        loss_mean = 0
        tic = 0

        for step in tqdm(range(step_num)): 

            image_batch, label_batch = d_loader[step * batch_size: (step + 1) * batch_size]

            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output = model(image_batch)
            # loss = criterion(output, label)
            # score = metric(output, label)
            # loss += 1 - score
            loss = metric(output, label_batch)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            loss_mean = loss_mean + loss.item()
            tic = tic + 1

            print(f'Step: [{step+1}/{step_num}], Loss: {loss.item(): .4f}\n')
        
        # Learning rate shceduler
        scheduler.step()

        print(f' Epoch [{epoch+1}/{epoch_num}], Average Loss of this iteration: {loss_mean/tic:.4f}')

    print("Finished Training, Loss: {loss_mean/tic:.4f}")

    # save the model
    saved_model_path = args.outmo
    torch.save(model.state_dict(), saved_model_path)
    print("Model successfully saved!")

    # # Make prediction
    # raw_path = args.tinimg
    # seg_path = args.tinlab
    # out_img_name = args.outim

    # out_img_path = "./saved_image/" + out_img_name + ".nii.gz"

    # verification(raw_path, seg_path, 0, model, out_img_path, mode='sigmoid')
    
















