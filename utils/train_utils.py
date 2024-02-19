"""
Provides all the utilities used for training process

Last edited: 19/10/2023
"""
import os
import re
import shutil
import torch
from tqdm import tqdm
from .unet_utils import *
from .single_data_loader import single_channel_loader, multi_channel_loader
from .module_utils import prediction_and_postprocess
from models import Unet, ASPPCNN, CustomSegmentationNetwork, MainArchitecture

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


class TTA_Training:
    """
    A class that defines the training process for a model using Test Time Augmentation (TTA).
    TTA is a technique that involves augmenting test images with various transformations and averaging the predictions
    of the model on these augmented images to improve performance.
    
    Args:
    loss_name (str): The name of the loss metric to be used during training.
    model_name (str): The name of the model to be used during training.
    in_chan (int): The number of input channels for the model.
    out_chan (int): The number of output channels for the model.
    filter_num (int): The number of filters to be used in the model.
    optimizer_name (str): The name of the optimizer to be used during training.
    learning_rate (float): The learning rate to be used during training.
    optim_gamma (float): The gamma value to be used for the optimizer.
    epoch_num (int): The number of epochs to be used during training.
    batch_mul (int): The batch size multiplier to be used during training.
    patch_size (int): The size of the patches to be used during training.
    augmentation_mode (str): The type of augmentation to be used during training.
    pretrained_model (str): The path to the pre-trained model to be used during training.
    thresh (float): The threshold value to be used during training.
    connect_thresh (float): The connection threshold value to be used during training.
    
    Methods:
    loss_init(): Initializes the loss metric to be used during training.
    model_init(): Initializes the model to be used during training.
    scheduler_init(optimizer): Initializes the learning rate scheduler to be used during training.
    aug_init(): Initializes the augmentation object to be used during training.
    pretrained_model_loader(): Loads the pre-trained model to be used during training.
    training_loop(data_loader, model, save_path): Defines the training loop for the model.
    train(ps_path, seg_path, out_mo_path): Trains the model using the specified data and saves the trained model.
    test_time_adaptation(ps_path, px_path, out_path, out_mo_path, resource_opt): Applies the trained model to test data.
    """
    def __init__(self, loss_name, model_name, 
                in_chan, out_chan, filter_num,
                optimizer_name, learning_rate,
                optim_gamma, epoch_num,
                batch_mul,  
                patch_size, augmentation_mode,
                pretrained_model = None,
                thresh = None, connect_thresh = None):
        # type of the loss metric
        self.loss_name = loss_name
        # type of the model
        self.model_name = model_name
        # model inti configuration
        self.model_config = [in_chan, out_chan, filter_num]
        # type of the optimizer
        self.optimizer_name = optimizer_name
        # learning rate
        self.learning_rate = learning_rate
        # gamma value for optimizer
        self.optim_gamma = optim_gamma
        # epoch number
        self.epoch_num = epoch_num
        # batch size multiplier
        self.batch_mul = batch_mul
        # thresholding parameters
        self.threshhold_vector = [thresh, connect_thresh]
        # augmentation configuration
        self.aug_config = [patch_size, augmentation_mode]
        # pretrained model path
        self.pretrained_model = pretrained_model
        # hardware config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loss_init(self):
        return loss_metric(self.loss_name)  

    def model_init(self):
        return model_chosen(self.model_name, self.model_config[0], self.model_config[1], self.model_config[2]).to(self.device)
    
    def scheduler_init(self, optimizer):
        optim_patience = np.int64(np.ceil(self.epoch_num * 0.2))
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = self.optim_gamma, patience = optim_patience)
    
    def aug_init(self):
        return aug_utils(self.aug_config[0], self.aug_config[1])
    
    def pretrained_model_loader(self):
        load_model = self.model_init()
        # load the pre-trained model
        if torch.cuda.is_available() == True:
            print("Running with GPU")
            load_model.load_state_dict(torch.load(self.pretrained_model))
        else:
            print("Running with CPU")
            load_model.load_state_dict(torch.load(self.pretrained_model, map_location=torch.device('cpu')))
        load_model.eval()
        print(f"The chosen model is: {self.pretrained_model}")

        return load_model

    def training_loop(self, data_loader, model, save_path):
        
        # initialize optimizer & scheduler
        optimizer = optim_chosen(self.optimizer_name, model.parameters(), self.learning_rate)
        scheduler = self.scheduler_init(optimizer)
        # initialize loss metric
        metric = self.loss_init()
        # initialize augmentation object    
        aug_item = self.aug_init()

        # traning loop (this could be separate out )
        for epoch in tqdm(range(self.epoch_num)):
            #traverse every image, load a chunk with its augmented chunks to the model
            sum_lr = 0
            for file_idx in range(len(data_loader)):
                image, label = next(iter(data_loader[file_idx]))
                image_batch, label_batch = aug_item(image, label)
                for i in range(1, self.batch_mul):
                    image, label = next(iter(data_loader[file_idx]))
                    image_batch_temp, label_batch_temp = aug_item(image, label)
                    image_batch = torch.cat((image_batch, image_batch_temp), dim=0)
                    label_batch = torch.cat((label_batch, label_batch_temp), dim=0)
                image_batch, label_batch = image_batch.to(self.device), label_batch.to(self.device)

                optimizer.zero_grad()
                
                # Forward pass
                output = model(image_batch)
                loss = metric(output, label_batch)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Learning rate shceduler
                scheduler.step(loss)

                sum_lr += optimizer.param_groups[0]['lr']
            
            tqdm.write(f'Epoch: [{epoch+1}/{self.epoch_num}], Loss: {loss.item(): .4f}, Current learning rate: {(sum_lr/len(data_loader)): .8f}')

        print("Training finished! Please wait for the model to be saved!\n")
        
        torch.save(model.state_dict(), save_path)
        print(f"Model successfully saved! The location of the saved model is: {save_path}\n")

    def train(self, ps_path, seg_path, out_mo_path):
        # initialize the data loader
        step = int(self.epoch_num * self.batch_mul)
        multi_image_loder = multi_channel_loader(ps_path, seg_path, self.aug_config[0], step)
        # initialize the model
        model = self.model_init()

        print(f"\nIn this test, the batch size is {6 * self.batch_mul}\n")

        # training loop
        self.training_loop(multi_image_loder, model, out_mo_path)

    def test_time_adaptation(self, ps_path, px_path, out_path, out_mo_path, resource_opt):
        # traverse each image
        processed_data_list = os.listdir(ps_path)
        for i in range(len(processed_data_list)):
            # filename
            file_name = processed_data_list[i].split('.')[0]
            # if the proxies are not provided, 
            # then use the pre-trained model to generate the proxies
            if len(os.listdir(px_path)) != len(os.listdir(ps_path)):
                print("No proxies are provided, strating generating proxies...")
                # initialize the inference method for generating the proxies
                inference_postpo = prediction_and_postprocess(self.model_name, self.model_config[0], self.model_config[1], self.model_config[2], ps_path, px_path)
                # mip flag set to be False, cuz we don't want mip when generating proxies
                inference_postpo(self.threshhold_vector[0], self.threshhold_vector[1], self.pretrained_model, processed_data_list[i], mip_flag=False)

            # fintuning (generate all finetuned models)
            test_img_path = os.path.join(ps_path, processed_data_list[i]) # path of the preprocessed image
            # find the corresponding proxy
            bool_list = [bool(re.search(processed_data_list[i].split('.')[0], filename)) for filename in os.listdir(px_path)]
            assert True in bool_list, "No such proxy file!"
            print("Proxies are provided!")
            test_px_path = os.path.join(px_path, os.listdir(px_path)[bool_list.index(True)]) # path of the proxy seg

            #initialize the data loader
            data_loader = dict()
            data_loader.__setitem__(0, single_channel_loader(test_img_path, test_px_path, self.aug_config[0], self.epoch_num))

            # initialize model
            model = self.pretrained_model_loader()

            print("Finetuning procedure starts!")
            # full path of the intermediate model
            out_mo_name = os.path.join(out_mo_path, file_name)

            # training loop
            self.training_loop(data_loader, model, out_mo_name)

            # inference by using the finetuned model
            print(f"Final thresholding for {file_name} will start shortly!\n")
            # initialize the inference method for generating the proxies
            inference_postpo_final = prediction_and_postprocess(self.model_name, self.model_config[0], self.model_config[1], self.model_config[2], ps_path, out_path)
            # generate mip images at the final stage
            inference_postpo_final(self.threshhold_vector[0], self.threshhold_vector[1], out_mo_name, processed_data_list[i], mip_flag=True)
    
        print("The test-time adaptation is finished!\n")

        # checking the resource optimization flag
        if resource_opt == 0:
            print("Resource optimization is disabled, all intermediate files are saved locally!\n")
            print(f"Finetuned model -> {out_mo_path}\n")
            print(f"Intermediate proxy -> {px_path}\n")
        elif (resource_opt == 1):
            shutil.rmtree(px_path) # clear all the proxies
            shutil.rmtree(out_mo_path) # clear all the finetuned models
            print("Intermediate files have been cleaned!")