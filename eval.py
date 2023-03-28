"""
eval.py for challenge
last edited: 03/27/2023
"""
from eval_utils import preprocess, testAndPostprocess, finetune
import config
import os

if __name__ == "__main__":
    # initialize command line
    args = config.args

    ds_path = args.ds_path # original data
    ps_path = args.ps_path # preprocessed data
    out_path = args.out_path # final segmentation

    prep_bool = args.prep_bool
    if prep_bool == False:
        ps_path = args.ds_path

    # model configuration
    model_type = args.mo
    in_chan = args.ic
    ou_chan = args.oc
    fil_num = args.fil

    # initial trained model name
    init_mo = args.init_tm

    # thresholding values
    init_thresh_vector = args.init_thresh_vector
    final_thresh_vector = args.final_thresh_vector


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
    finetune_object = finetune(ps_path, out_path, model_type, in_chan, ou_chan, fil_num, init_mo)

    # final prediction and thresholding - and override all the proxies in the output path
    finetuned_model_list = os.listdir("./saved_models/finetuned/")
    print("Final prediction starts!\n")
    for idx in range(len(finetuned_model_list)):
        model_name = finetuned_model_list[idx]
        img_name = model_name + ".nii.gz" #TODO: check this !
        if img_name in os.listdir(ps_path):
            pred_post_object(final_thresh_vector[0], final_thresh_vector[1], model_name, img_name)
        else:
            print(f"No such image file: {img_name}")
            continue

    print("The pipeline is finished!\n")


