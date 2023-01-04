import nibabel as nib
import matplotlib.pyplot as plt

test_arr = nib.load("./saved_image/bce_50_lr_5e3_1img_slab3_1.nii.gz").get_fdata()
ground_truth = nib.load("./data/seg1/seg_TOF_3D_160um_TR20_TE6p56_sli52_FA18_FCY_BW100_17_biasCor_noiseCor_VT260_lVT200_VENP10.nii.gz").get_fdata()


chosen_slice = 1
plt.figure()
plt.imshow(test_arr[:,:,chosen_slice],cmap='gray')
plt.colorbar()

plt.figure()
plt.imshow(ground_truth[:,:,chosen_slice],cmap='gray')
plt.colorbar()
plt.show()