"""
helper functions library for evaluation

Editor: Marshall Xu
Last Edited: 21/07/2023
"""

import os
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def out_file_reader(file_path):
    loss_values = []
    

    with open(file_path, 'r') as file:
        for line in file:
            # Use regular expression to find values following "loss: " and ","
            match = re.search(r'Loss:  (\d+\.\d+),', line)
            if match:
                loss_value = float(match.group(1))
                loss_values.append(loss_value)
    
    # Create a line plot
    plt.figure()
    plt.plot(range(1, len(loss_values) + 1), loss_values, linestyle='-')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Loss Values Over Iterations')
    plt.grid(True)
    plt.show()

def mra_deskull(img_path, msk_path, mip_flag):
    """
    apply a mask on the target nifti image, 
    and generate an mip image
    """
    # Extract the file_name and dir_name
    file_name = os.path.basename(img_path)
    dir_name = os.path.dirname(img_path)

    # Load the nifti image and its mask
    img = nib.load(img_path)
    affine = img.affine # type: ignore
    header = img.header
    img_arr = img.get_fdata() # type: ignore
    msk_arr = nib.load(msk_path).get_fdata() # type: ignore
    
    # Apply the mask
    masked_arr = np.multiply(img_arr, msk_arr)
    masked_nifti = nib.Nifti1Image(masked_arr, affine, header)
    new_file_name = "MASKED_" + file_name
    save_path_nifti = os.path.join(dir_name, new_file_name)
    nib.save(masked_nifti, save_path_nifti)
    print("Masked Nifti image has been sucessfully saved!")

    # When the mip_flag is on,
    # generate an mip image to the same folder as the input image path
    if mip_flag == True:
        masked_mip = np.max(masked_arr, 2)
        masked_mip = np.rot90(masked_mip, axes=(0, 1))
        mip_name = new_file_name.split('.')[0] + ".jpg"
        save_path_mip = os.path.join(dir_name, mip_name)
        plt.imsave(save_path_mip, masked_mip, cmap='gray')
        print("MIP image has been successfully saved!")

class eval_scores:
    def __init__(self, seg, gt) -> None:
        """
        seg, gt: numpy array
        """
        self.seg = seg.reshape(-1,1)
        self.gt = gt.reshape(-1,1)
        self.tn, self.fp, self.fn, self.tp = self._four_cardinalities(self.seg, self.gt)
        self.n = self.tn + self.fp + self.fn + self.tp

    def _four_cardinalities(self, seg, gt):
        tp = (seg*gt).sum()
        fp = (seg*(1-gt)).sum()
        fn = ((1-seg)*gt).sum()
        tn = ((1-seg)*(1-gt)).sum()
        return tn, fp, fn, tp
    
    def _dice(self):
        # dice score
        return (2*self.tp) / (2*self.tp + self.fp + self.fn)
    
    def _jaccard(self):
        # jaccard score
        return self.tp / (self.tp + self.fp + self.fn)
    
    def _vol_sim(self):
        # volume similarity
        return 1 - np.abs(self.fn - self.fp) / (2*self.tp + self.fp + self.fn)
    
    def _mutual_info(self):
        # mutual information (discarded)
        psg1 = (self.tp + self.fn) / self.n
        psg2 = (self.tn + self.fn) / self.n
        pst1 = (self.tp + self.fp) / self.n
        pst2 = (self.tn + self.fp) / self.n

        Hsg = -(psg1*np.log(psg1) + psg2*np.log(psg2))
        Hst = -(pst1*np.log(pst1) + pst2*np.log(pst2))
        Hsgst = -((self.tp/self.n)*np.log(self.tp/self.n) + (self.fn/self.n)*np.log(self.fn/self.n) + (self.fp/self.n)*np.log(self.fp/self.n) + (self.tn/self.n)*np.log(self.tn/self.n))
        MI = Hsg + Hst - Hsgst

        return 2 * MI / (Hsg + Hst)
    
    def _adjusted_ri(self):
        # adjusted rand index (replacement for Balanced Average Hausdorff Distance)

        # basic cardinalities
        a = (self.tp*(self.tp-1) + self.fp*(self.fp-1) + self.tn*(self.tn-1) + self.fn*(self.fn-1)) / 2
        b = ((self.tp+self.fn)**2 + (self.tn+self.fp)**2 - (self.tp**2+self.tn**2+self.fp**2+self.fn**2)) / 2
        c = ((self.tp+self.fp)**2 + (self.tn+self.fn)**2 - (self.tp**2+self.tn**2+self.fp**2+self.fn**2)) / 2
        d = self.n*(self.n-1)/2 - (a+b+c)

        return 2*(a*d - b*c) / (c**2 + b**2 + 2*a*d + (a+d)*(c+b))
    
    def __call__(self):
        a = self._dice()
        b = self._jaccard()
        c = self._vol_sim()
        d = self._mutual_info()
        e = self._adjusted_ri()
        
        return [a,b,c,d,e]