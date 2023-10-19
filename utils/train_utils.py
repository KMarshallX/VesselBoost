"""
Provides all the utilities used for training process

Last edited: 19/10/2023
"""

import torch
from tqdm import tqdm
from utils.unet_utils import *
from utils.module_utils import prediction_and_postprocess
from models.unet_3d import Unet
from models.asppcnn import ASPPCNN
from models.aspp import CustomSegmentationNetwork
from models.ra_unet import MainArchitecture

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
    elif metric_name == "combo":
        # combo loss
        return ComboLoss()
    else:
        print("Enter a valid loss metric.")

class Training:
    """
    A class for training a neural network model.

    Args:
    loss_name (str): Name of the loss function to be used.
    model_name (str): Name of the neural network model to be used.
    in_chan (int): Number of input channels.
    out_chan (int): Number of output channels.
    filter_num (int): Number of filters to be used.
    optimizer_name (str): Name of the optimizer to be used.
    learning_rate (float): Learning rate for the optimizer.
    optim_gamma (float): Gamma value for the optimizer.
    epoch_num (int): Number of epochs for training.
    batch_mul (int): Batch size multiplier.
    patch_size (tuple): Size of the patches to be used.
    augmentation_mode (str): Name of the augmentation method to be used.

    Attributes:
    metric (function): Loss function to be used.
    device (torch.device): cpu or gpu.
    model (torch.nn.Module): Neural network model to be used.
    optimizer (torch.optim.Optimizer): Optimizer to be used.
    epoch_num (int): Number of epochs for training.
    scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler to be used.
    aug_item (function): Augmentation method to be used.
    batch_mul (int): Batch size multiplier.

    Methods:
    __call__(self, data_loader, save_path): Trains the model using the given data loader and saves the model to the given path.
    """
class Training:
    def __init__(self, loss_name, model_name, 
                in_chan, out_chan, filter_num,
                optimizer_name, learning_rate,
                optim_gamma, epoch_num,
                batch_mul, patch_size, 
                augmentation_mode):
        # loss metric
        self.metric = loss_metric(loss_name)
        # hardware config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model configuration
        self.model = model_chosen(model_name, in_chan, out_chan, filter_num).to(self.device)
        # optimizer
        self.optimizer = optim_chosen(optimizer_name, self.model.parameters(), learning_rate)
        # epoch number
        self.epoch_num = epoch_num
        # set optim scheduler
        optim_patience = np.int64(np.ceil(self.epoch_num * 0.2))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = optim_gamma, patience = optim_patience)
        # initialize the augmentation method
        self.aug_item = aug_utils(patch_size, augmentation_mode)
        # batch size multiplier
        self.batch_mul = batch_mul

    def __call__(self, data_loader, save_path):
        # traning loop (this could be separate out )
        for epoch in tqdm(range(self.epoch_num)):
            #traverse every image, load a chunk with its augmented chunks to the model
            sum_lr = 0
            for file_idx in range(len(data_loader)):
                image, label = next(iter(data_loader[file_idx]))
                image_batch, label_batch = self.aug_item(image, label)
                for i in range(1, self.batch_mul):
                    image, label = next(iter(data_loader[file_idx]))
                    image_batch_temp, label_batch_temp = self.aug_item(image, label)
                    image_batch = torch.cat((image_batch, image_batch_temp), dim=0)
                    label_batch = torch.cat((label_batch, label_batch_temp), dim=0)
                image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)

                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(image_batch)
                loss = self.metric(output, label_batch)

                # Backward and optimize
                loss.backward()
                self.optimizer.step()

                # Learning rate shceduler
                self.scheduler.step(loss)

                sum_lr += self.optimizer.param_groups[0]['lr']
            
            tqdm.write(f'Epoch: [{epoch+1}/{self.epoch_num}], Loss: {loss.item(): .4f}, Current learning rate: {(sum_lr/len(data_loader)): .8f}')

        print("Training finished! Please wait for the model to be saved!\n")
        
        torch.save(self.model.state_dict(), save_path)
        print(f"Model successfully saved! The location of the saved model is: {save_path}\n")

class Tta:
    def __init__(self, loss_name, model_name, 
                in_chan, out_chan, filter_num,
                optimizer_name, learning_rate,
                optim_gamma, epoch_num,
                batch_mul, patch_size, 
                augmentation_mode):
        # loss metric
        self.metric = loss_metric(loss_name)
        # hardware config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model configuration
        self.model = model_chosen(model_name, in_chan, out_chan, filter_num).to(self.device)
        # optimizer
        self.optimizer = optim_chosen(optimizer_name, self.model.parameters(), learning_rate)
        # epoch number
        self.epoch_num = epoch_num
        # set optim scheduler
        optim_patience = np.int64(np.ceil(self.epoch_num * 0.2))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = optim_gamma, patience = optim_patience)
        # initialize the augmentation method
        self.aug_item = aug_utils(patch_size, augmentation_mode)
        # batch size multiplier
        self.batch_mul = batch_mul
    

