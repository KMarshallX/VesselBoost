#!/usr/bin/env python3

"""
Test time adpatation module

Editor: Marshall Xu
Last Edited: 06/19/2023
"""

import os
import shutil
import adapt_config

from utils.module_utils import preprocess, testAndPostprocess
from utils.unet_utils import *
from utils.new_data_loader import single_channel_loader
from models.unet_3d import Unet

args = adapt_config.args

ds_path = args.ds_path # path to original data
ps_path = args.ps_path # path to preprocessed data
out_path = args.out_path # path to infered data
if os.path.exists(out_path)==False:
    os.mkdir(out_path)
prep_mode = args.prep_mode # preprocessing mode
# when the preprocess is skipped, 
# directly take the raw data for inference
if prep_mode == 4:
    ps_path = ds_path
px_path = args.px_path # path to proxies    

model_type = args.mo # model type
in_chan = args.ic # input channel
ou_chan = args.oc # output channel
fil_num = args.fil # number of filters

threshold_vector = [args.thresh, args.cc]
pretrained_model = args.pretrained # path to pretrained model

# finetuning hyperparams
learning_rate = args.lr
optim_gamma = args.optim_gamma
optim_patience = args.optim_patience
epoch_num = args.ep
patch_size = args.osz
aug_mode = args.aug_mode

# output fintuned model path
out_mo_path = "./finetuned/"
if os.path.exists(out_mo_path)==False:
    os.mkdir(out_mo_path) # make directory "./finetuned/"

# Resource optimization flag
resource_opt = args.resource

if __name__ == "__main__":

    print("TTA session will start shortly..")

    # initialize the preprocessing method with input/output paths
    preprocessing = preprocess(ds_path, ps_path)
    # start or abort preprocessing 
    preprocessing(prep_mode)
    # traverse each image
    processed_data_list = os.listdir(ps_path)
    for i in range(len(processed_data_list)):
        # if the proxies are not provided, then use the pre-trained model to generate the proxies
        if px_path == None:
            print("No proxies are provided, strating generating proxies...")
            # make directory to proxies
            px_path = "./proxies/"
            if os.path.exists(px_path) == False:
                os.mkdir(px_path)
            # initialize the inference method for generating the proxies
            inference_postpo = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, px_path)
            inference_postpo(threshold_vector[0], threshold_vector[1], pretrained_model, processed_data_list[i])
        
        # fintuning (generate all finetuned models)
        test_img_path = ps_path + processed_data_list[i] # path of the preprocessed image
        # find the corresponding proxy
        assert (processed_data_list[i] in os.listdir(px_path)), "No such proxy file!"
        print("Proxies are provided!")
        test_px_path = px_path + processed_data_list[i] # path of the proxy seg
        
        #initialize the data loader
        data_loader = single_channel_loader(test_img_path, test_px_path, patch_size, epoch_num)
        
        # initialize pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_model = Unet(in_chan, ou_chan, fil_num).to(device)

        # load the pre-trained model
        load_model.load_state_dict(torch.load(pretrained_model))
        load_model.eval()
        print(f"The chosen model is: {pretrained_model}")

        # initialize optimizer & scheduler
        optimizer = torch.optim.Adam(load_model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=optim_gamma, patience=optim_patience)

        # initialize loss metric & optimizer
        metric = TverskyLoss()
        # initialize augmentation object
        aug_item = aug_utils(patch_size, aug_mode)

        print("Finetuning procedure starts!")
        # training loop
        for epoch in tqdm(range(epoch_num)):
            image, label = next(iter(data_loader))
            image_batch, label_batch = aug_item(image, label)
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = load_model(image_batch)
            loss = metric(output, label_batch)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Learning rate shceduler
            scheduler.step(loss)

            # TODO: debug message, delete this
            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f'Epoch: [{epoch+1}/{epoch_num}], Loss: {loss.item(): .4f}, Current learning rate: {current_lr: .8f}')

        file_name = processed_data_list[i].split('.')[0]
        out_mo_name = out_mo_path + file_name
        torch.save(load_model.state_dict(), out_mo_name)
        print(f"Training finished! The finetuning model of {file_name} successfully saved!\n")

        model_name = "./finetuned/" + file_name
        img_name = file_name + ".nii.gz"
        
        # inference by using the finetuned model
        print(f"Final thresholding for {file_name} will start shortly!\n")
        # initialize the inference method for generating the proxies
        inference_postpo_final = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, out_path)
        inference_postpo_final(threshold_vector[0], threshold_vector[1], model_name, processed_data_list[i])
    
    print("The test-time adaptation is finished!\n")

    # checking the resource optimization flag
    if resource_opt == 0:
        print("Resource optimization is disabled, all intermediate files are saved locally!\n")
        print("Finetuned model -> ./finetuned/\n")
        print("Intermediate proxy -> ./proxies/\n")
    elif (resource_opt == 1 and px_path == "./proxies/"):
        shutil.rmtree(px_path) # clear all the proxies
        shutil.rmtree(out_mo_path) # clear all the finetuned models
        print("Intermediate files have been cleaned!")





