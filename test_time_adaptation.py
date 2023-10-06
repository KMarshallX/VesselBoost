#!/usr/bin/env python3

"""
Test time adpatation module

Editor: Marshall Xu
Last Edited: 08/10/2023
"""

import os
import re
import shutil
import numpy as np
import config.adapt_config as adapt_config
from utils.module_utils import preprocess, testAndPostprocess
from utils.unet_utils import *
from utils.single_data_loader import single_channel_loader
from models.unet_3d import Unet

args = adapt_config.args

ds_path = args.ds_path # path to original data
ps_path = args.ps_path # path to preprocessed data
out_path = args.out_path # path to infered data
if os.path.exists(out_path) == False:
    print(f"{out_path} does not exist.")
    os.mkdir(out_path)
    print(f"{out_path} has been created!")


prep_mode = args.prep_mode # preprocessing mode
# when the preprocess is skipped, 
# directly take the raw data for inference
if prep_mode == 4:
    ps_path = ds_path

px_path = args.px_path # path to proxies
if px_path == None: # when the proxy segmentation is not provided
    px_path = os.path.join(out_path, "proxies", "")
    if os.path.exists(px_path) == False:
        print(f"{px_path} does not exist.")     
        os.mkdir(px_path) # create an intermediate output folder inside the output path
        print(f"{px_path} has been created!")
    assert os.path.exists(px_path) == True, "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

model_type = args.mo # model type
in_chan = args.ic # input channel
ou_chan = args.oc # output channel
fil_num = args.fil # number of filters

threshold_vector = [args.thresh, args.cc]
pretrained_model = args.pretrained # path to pretrained model

# finetuning hyperparams
learning_rate = args.lr
optim_gamma = args.optim_gamma
epoch_num = args.ep
optim_patience = np.int64(np.ceil(epoch_num * 0.2)) 
patch_size = args.osz
aug_mode = args.aug_mode

# output fintuned model path
out_mo_path = os.path.join(out_path, "finetuned", "")
if os.path.exists(out_mo_path) == False:
    print(f"{out_mo_path} does not exist.")     
    os.mkdir(out_mo_path) # create an intermediate output folder inside the output path
    print(f"{out_mo_path} has been created!")
assert os.path.exists(out_mo_path) == True, "Container doesn't initialize properly, contact for maintenance: https://github.com/KMarshallX/vessel_code"

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
        # if the proxies are not provided, 
        # then use the pre-trained model to generate the proxies
        if len(os.listdir(px_path)) != len(os.listdir(ds_path)):
            print("No proxies are provided, strating generating proxies...")
            # initialize the inference method for generating the proxies
            inference_postpo = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, px_path)
            # mip flag set to be False, cuz we don't want mip when generating proxies
            inference_postpo(threshold_vector[0], threshold_vector[1], pretrained_model, processed_data_list[i], mip_flag=False)
        
        # fintuning (generate all finetuned models)
        test_img_path = os.path.join(ps_path, processed_data_list[i]) # path of the preprocessed image
        # find the corresponding proxy
        bool_list = [bool(re.search(processed_data_list[i].split('.')[0], filename)) for filename in os.listdir(px_path)]
        assert True in bool_list, "No such proxy file!"
        print("Proxies are provided!")
        test_px_path = os.path.join(px_path, os.listdir(px_path)[bool_list.index(True)]) # path of the proxy seg
        
        #initialize the data loader
        data_loader = single_channel_loader(test_img_path, test_px_path, patch_size, epoch_num)
        
        # initialize pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_model = Unet(in_chan, ou_chan, fil_num).to(device)

        # load the pre-trained model
        if torch.cuda.is_available() == True:
            print("Running with GPU")
            load_model.load_state_dict(torch.load(pretrained_model))
        else:
            print("Running with CPU")
            load_model.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
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
        print(f"\nIn this test, the batch size is {6*args.batch_mul}\n")
        # training loop
        for epoch in tqdm(range(epoch_num)):
            image, label = next(iter(data_loader))
            image_batch, label_batch = aug_item(image, label)
            for j in range(1, args.batch_mul):
                image, label = next(iter(data_loader))
                image_batch_temp, label_batch_temp = aug_item(image, label)
                image_batch = torch.cat((image_batch, image_batch_temp), 0)
                label_batch = torch.cat((label_batch, label_batch_temp), 0)
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

            current_lr = optimizer.param_groups[0]['lr']
        
        tqdm.write(f'Epoch: [{epoch+1}/{epoch_num}], Loss: {loss.item(): .4f}, Current learning rate: {current_lr : .8f}')

        file_name = processed_data_list[i].split('.')[0]
        out_mo_name = os.path.join(out_mo_path, file_name)
        torch.save(load_model.state_dict(), out_mo_name)
        print(f"Training finished! The finetuning model of {file_name} successfully saved!\n")
        
        # inference by using the finetuned model
        print(f"Final thresholding for {file_name} will start shortly!\n")
        # initialize the inference method for generating the proxies
        inference_postpo_final = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, out_path)
        # generate mip images at the final stage
        inference_postpo_final(threshold_vector[0], threshold_vector[1], out_mo_name, processed_data_list[i], mip_flag=True)
    
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





