"""
eval.py for challenge
last edited: 03/30/2023
"""

import os
import time
from module_utils import preprocess, testAndPostprocess
import eval_config

from utils.unet_utils import *
from utils.new_data_loader import single_channel_loader
from models.unet_3d import Unet


if __name__ == "__main__":
    # initialize command line
    args = eval_config.args

    ds_path = args.ds_path # original data
    ds_path = ds_path + "test/"
    ps_path = args.ps_path # preprocessed data
    out_path = args.out_path # final segmentation
    px_path = "./proxies/"
    if os.path.exists(px_path) == False:
        os.mkdir(px_path)

    prep_mode = args.prep_mode
    if prep_mode == 4:
        ps_path = ds_path

    # model configuration
    model_type = args.mo
    in_chan = args.ic
    ou_chan = args.oc
    fil_num = args.fil

    # initial trained model name & path
    mo_path = "./saved_models/"
    init_mo = os.listdir(mo_path)[0]
    init_mo_path = mo_path + os.listdir(mo_path)[0]
    # checking the pretrained weights, make the corresponding output directory
    if init_mo == "Init_ep1000_lr1e3_tver":
        out_path = out_path + "koala_manual/"
    elif init_mo == "Init_ep1000_lr1e3_tver_OM1":
        out_path = out_path + "koala_om1/"
    elif init_mo == "Init_ep1000_lr1e3_tver_OM2":
        out_path = out_path + "koala_om2/"
    else:
        raise Exception("No corrsponding model found!")
    
    if os.path.exists(out_path)==False:
        os.mkdir(out_path) # make directory "/out_path/koala_manual(omelette1/omelette2)/"
    
    # output fintuned model path
    out_mo_path = "./finetuned/"
    if os.path.exists(out_mo_path)==False:
        os.mkdir(out_mo_path) # make directory "./finetuned/"

    # thresholding values
    init_thresh_vector = [0.1, 10] # initial thresholding values for predicting proxy
    final_thresh_vector = [0.1, 10] # final thresholding values for predicting segmentation

    # finetuning hyperparams
    learning_rate = args.eval_lr
    optim_gamma = args.eval_gamma
    optim_patience = args.eval_patience
    epoch_num = args.eval_ep


    """
    finetuning pipeline
    """
    # preprocessing - bias field correct and denoise all the raw data and store'em in the ps_path
    prep_object = preprocess(ds_path, ps_path)
    prep_object(prep_mode)
    
    # prediction and postprocessing
    pred_post_object = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, px_path) # initialize the object, takes all images from ps_path, output proxies to px_path
    # generate proxy segmentations by using the initial model - and store'em in the out_path
    processed_data_list = os.listdir(ps_path)
    for i in range(len(processed_data_list)): # generate finetuned model for each test image
        # generate proxy segmentation for all test data
        pred_post_object(init_thresh_vector[0], init_thresh_vector[1], init_mo_path, processed_data_list[i])
    
        # fintuning (generate all finetuned models)
        test_img_path = ps_path + processed_data_list[i] # path of the preprocessed image
        # find the corresponding proxy
        assert (processed_data_list[i] in os.listdir(px_path)), "No such proxy file!"
        test_px_path = px_path + processed_data_list[i] # path of the proxy seg
        
        #initialize the data loader
        data_loader = single_channel_loader(test_img_path, test_px_path, (64,64,64), epoch_num)
        
        # initialize pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_model = Unet(in_chan, ou_chan, fil_num).to(device)
        
        # load the pre-trained model
        load_model.load_state_dict(torch.load(init_mo_path))
        load_model.eval()
        time.sleep(2) # wait for 2 seconds, make sure the model has been loaded
        # TODO delete this message
        print(f"The chosen model is: {init_mo_path}")

        # initialize optimizer & scheduler
        # TODO: check this!!!!
        optimizer = torch.optim.Adam(load_model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=optim_gamma, patience=optim_patience)

        # initialize loss metric & optimizer
        metric = TverskyLoss()
        # initialize augmentation object
        aug_item = aug_utils((64,64,64), "mode1")

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
        print(f"Final thresholding for {file_name} will start shortly!\n")

        # thresholding for the final prediction
        pred_post_object_final = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, out_path)
        pred_post_object_final(final_thresh_vector[0], final_thresh_vector[1], model_name, img_name)


    print("The pipeline is finished!\n")

