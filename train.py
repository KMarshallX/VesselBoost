#!/usr/bin/env python3

"""
Training the chosen model (new pipeline)

Editor: Marshall Xu
Last Edited: 10/07/2023
"""
import torch
import config.train_config as train_config
from tqdm import tqdm
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
raw_img = args.ds_path
processed_img = args.ps_path
seg_img = args.lb_path
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
optim_gamma = args.optim_gamma
optim_patience = np.int64(np.ceil(epoch_num * 0.2)) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = optim_gamma, patience = optim_patience)
# initialize the augmentation method
aug_item = aug_utils(args.osz, args.aug_mode)


if __name__ == "__main__":
    print("Training session will start shortly..")

    # initialize the preprocessing method with input/output paths
    preprocessing = preprocess(raw_img, processed_img)
    # start or abort preprocessing 
    preprocessing(prep_mode)

    # make sure the image path and seg path contains equal number of files
    raw_file_list = os.listdir(processed_img)
    seg_file_list = os.listdir(seg_img)
    assert (len(raw_file_list) == len(seg_file_list)), "Number of images and correspinding segs not matched!"
    file_num = len(raw_file_list)

    # initialize single_channel_loaders for each image
    # and store the initialized loaders in a linked hashmaps
    loaders_dict = dict()
    for i in range(file_num):
        # joined path to the current image file 
        raw_img_name = os.path.join(processed_img, raw_file_list[i])
        # find the corresponding seg file in the seg_folder
        seg_img_name = None
        for j in range(file_num):
            if seg_file_list[j].find(raw_file_list[i].split('.')[0]) != -1:
                seg_img_name = os.path.join(seg_img, seg_file_list[j])
                break
        assert (seg_img_name != None), f"There is no corresponding label to {raw_file_list[i]}!"
        # a linked hashmap to store the provoked data loaders
        loaders_dict.__setitem__(i, single_channel_loader(raw_img_name, seg_img_name, args.osz, args.ep))

    # traning loop (this could be separate out )
    for epoch in tqdm(range(epoch_num)):
        #traverse every image, load a chunk with its augmented chunks to the model
        for file_idx in range(len(loaders_dict)):
            image, label = next(iter(loaders_dict[file_idx]))
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


    print("Training finished! Please wait for the model to be saved!\n")

    # save the model
    saved_model_path = args.outmo
    torch.save(model.state_dict(), saved_model_path)
    print(f"Model successfully saved! The location of the saved model is: {saved_model_path}\n")

    
















