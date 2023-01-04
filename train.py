"""
Training the chosen model 

Editor: Marshall Xu
Last Edited: 12/21/2022
"""

import config
import torch
from tqdm import tqdm

from utils.unet_utils import *
from utils.data_loader import data_loader
from models.unet_3d import Unet
from models.test_model import test_mo


def model_chosen(model_name, in_chan, out_chan, filter_num):
    if model_name == "unet3d":
        return Unet(in_chan, out_chan, filter_num)
    elif model_name == "test_mo":
        return test_mo(in_chan, out_chan, filter_num)
    else:
        print("Insert a valid model name.")

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
    d_loader = data_loader(raw_img, seg_img, args.psz, args.osz,args.pst)

    # loss
    loss_name = args.loss_m
    metric = loss_metric(loss_name)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    #epoch number
    epoch_num = args.ep

    # traning loop (this could be separate out )
    for epoch in tqdm(range(epoch_num)):
        loss_mean = 0
        tic = 0

        for image, label in d_loader: 
            
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output = model(image.unsqueeze(0))
            # loss = criterion(output, label)
            # score = metric(output, label)
            # loss += 1 - score
            loss = metric(output, label)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            loss_mean = loss_mean + loss.item()
            tic = tic + 1

            print(f'Step: [{tic}], Loss: {loss.item(): .4f}\n')

        print(f' Epoch [{epoch+1}/{epoch_num}], Average Loss of this iteration: {loss_mean/tic:.4f}')

    print("Finished Training, Loss: {loss_mean/tic:.4f}")

    # save the model
    saved_model_path = args.outmo
    torch.save(model.state_dict(), saved_model_path)
    print("Model successfully saved!")
    
















