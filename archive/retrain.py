"""
Retrain a pre-trained model (finetuning)

Editor: Marshall Xu
Last Edited: 03/14/2023
"""

import config
import torch
from tqdm import tqdm
import os

from utils.unet_utils import *
from utils.new_data_loader import single_channel_loader
from models.unet_3d import Unet
from models.test_model import test_mo
from models.asppcnn import ASPPCNN
from models.siyu import CustomSegmentationNetwork
from models.ra_unet import MainArchitecture


def model_chosen(model_name, in_chan, out_chan, filter_num):
    if model_name == "unet3d":
        return Unet(in_chan, out_chan, filter_num)
    elif model_name == "test_mo":
        return test_mo(in_chan, out_chan, filter_num)
    elif model_name == "aspp":
        return ASPPCNN(in_chan, out_chan, [1,2,3,5,7])
    elif model_name == "test":
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


if __name__ == "__main__":
    args = config.args

    # hardware config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # input images & labels
    raw_img = args.inimg
    seg_img = args.inlab

    # output model path
    saved_model_path = "./saved_models/"
    saved_model_name = args.retrain_name
    
    # model configuration
    load_model = model_chosen(args.mo, args.ic, args.oc, args.fil).to(device)
    trained_model_path = args.tm # this one has to be a full path
    # trained_model_path = "./saved_models/Unet_ep10_lr1e4_1slab_raw"
    load_model.load_state_dict(torch.load(trained_model_path))
    load_model.eval()

    # loss
    loss_name = args.loss_m
    metric = loss_metric(loss_name)

    # optimizer
    op_name = args.op
    optimizer = optim_chosen(op_name, load_model.parameters(), args.lr)
    # set optim scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.optim_step, gamma=args.optim_gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.optim_gamma, patience=args.optim_patience)

    #epoch number
    epoch_num = args.ep

    # initialize the augmentation method
    aug_item = aug_utils(args.osz, args.aug_mode)

    raw_file_list = os.listdir(raw_img)
    seg_file_list = os.listdir(seg_img)
    assert (len(raw_file_list) == len(seg_file_list)), "Number of images and correspinding segs not matched!"
    
    # traning loop (this could be separate out )
    loss_mean = 0

    raw_arr_name = raw_img + raw_file_list[0]
    seg_arr_name = seg_img + seg_file_list[0]      
    # initialize single channel data loader
    single_chan_loader = single_channel_loader(raw_arr_name, seg_arr_name, args.osz, args.ep)

    for epoch in tqdm(range(epoch_num)):
        
        image, label = next(iter(single_chan_loader))
        
        image_batch, label_batch = aug_item(image, label)
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        output = load_model(image_batch)
        loss = metric(output, label_batch)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        loss_mean = loss_mean + loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        # Learning rate shceduler
        scheduler.step(loss)

        print(f'Epoch: [{epoch+1}/{epoch_num}], Loss: {loss.item(): .4f}, Current learning rate: {current_lr: .8f}\n')
        if (epoch+1)%1000 == 0:
            checkpoint_name = saved_model_path + saved_model_name + str(epoch+1)
            torch.save(load_model.state_dict(), checkpoint_name)
            print(f"Retrained model checkpoint {epoch+1} saved!")

    print(f"Finished Training! Average loss value is: {loss_mean/epoch_num: .4f}")

    # save the model
    model_name = saved_model_path + saved_model_name + "_endpoint"
    torch.save(load_model.state_dict(), model_name)
    print("Retrained Model successfully saved!")

    
















