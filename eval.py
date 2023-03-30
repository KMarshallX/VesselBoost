"""
eval.py for challenge
last edited: 03/30/2023
"""
from eval_utils import preprocess, testAndPostprocess, finetune
import config
import os

from utils.unet_utils import *
from utils.new_data_loader import single_channel_loader
from models.unet_3d import Unet
from models.siyu import CustomSegmentationNetwork
from models.asppcnn import ASPPCNN
from models.ra_unet import MainArchitecture

def model_chosen(model_name, in_chan, out_chan, filter_num):
    if model_name == "unet3d":
        return Unet(in_chan, out_chan, filter_num)
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
    # initialize command line
    args = config.args

    ds_path = args.ds_path # original data
    ps_path = args.ps_path # preprocessed data
    out_path = args.out_path # final segmentation

    prep_bool = args.prep_bool
    if prep_bool == "no":
        ps_path = args.ds_path
        prep_bool = False
    else:
        prep_bool = True

    # model configuration
    model_type = args.mo
    in_chan = args.ic
    ou_chan = args.oc
    fil_num = args.fil

    # initial trained model name & path
    init_mo = "Init_ep1000_lr1e3_tver"
    init_mo_path = "./saved_models/Init_ep1000_lr1e3_tver"
    # output fintuned model path
    out_mo_path = "./saved_models/finetuned/"

    # thresholding values
    init_thresh_vector = args.init_thresh_vector
    final_thresh_vector = args.final_thresh_vector

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
    prep_object(prep_bool)
    
    # prediction and postprocessing
    pred_post_object = testAndPostprocess(model_type, in_chan, ou_chan, fil_num, ps_path, out_path) # initialize the object
    # generate proxy segmentations by using the initial model - and store'em in the out_path
    processed_data_list = os.listdir(ps_path)
    for i in range(len(processed_data_list)):
        pred_post_object(init_thresh_vector[0], init_thresh_vector[1], init_mo, processed_data_list[i])
    
        # fintuning (generate all finetuned models)
        test_img_path = ps_path + processed_data_list[i]
        # find the corresponding proxy
        assert (processed_data_list[i] in os.listdir(out_path)), "No such proxy file!"
        test_px_path = out_path + processed_data_list[i]
        #initialize the data loader
        data_loader = single_channel_loader(test_img_path, test_px_path, (64,64,64), epoch_num)
        # initialize pre-trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_model = model_chosen("unet3d", 1, 1, 16).to(device)
        load_model.load_state_dict(torch.load(init_mo_path))
        load_model.eval()

        # initialize optimizer & scheduler
        optimizer = optim_chosen('adam', load_model.parameters(), 1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=10)

        # initialize loss metric & optimizer
        metric = loss_metric("tver")
        # initialize augmentation object
        aug_item = aug_utils((64,64,64), "mode1")

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
        file_name = processed_data_list[i].split('.')[0]
        out_mo_name = out_mo_path + file_name
        torch.save(load_model.state_dict(), out_mo_name)
        print(f"Training finished! The finetuning model of {file_name} successfully saved!\n")

        model_name = "finetuned/" + file_name
        img_name = file_name + ".nii.gz"
        print(f"Final thresholding for {file_name} will start shortly!\n")
        pred_post_object(final_thresh_vector[0], final_thresh_vector[1], model_name, img_name)



    # # final prediction and thresholding - and override all the proxies in the output path
    # finetuned_model_list = os.listdir("./saved_models/finetuned/")
    # print("Final prediction starts!\n")
    # for idx in range(len(finetuned_model_list)):
    #     model_name = "finetuned/" + finetuned_model_list[idx]
    #     img_name = finetuned_model_list[idx] + ".nii.gz" #TODO: check this !
    #     if img_name in os.listdir(ps_path):
    #         pred_post_object(final_thresh_vector[0], final_thresh_vector[1], model_name, img_name)
    #     else:
    #         print(f"No such image file: {img_name}")
    #         continue

    print("The pipeline is finished!\n")


