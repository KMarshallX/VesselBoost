"""
5 Evaluation metrics used to evaluate the resulting segmentation

Marshall @ 03/07/2023
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
import nibabel as nib

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

        return 2*MI / (Hsg + Hst)
    
    def _norm_mi(self):
        # normalized mutual information
        return normalized_mutual_info_score(self.gt.squeeze(1), self.seg.squeeze(1))
    
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

if __name__ == "__main__":

    ground_truth_path = "./data/validate_label/sub017.nii"
    ground_truth_img = nib.load(ground_truth_path)
    ground_truth = ground_truth_img.get_fdata()

    saved_img_path = "./saved_image/"
    eval_img_list = [saved_img_path+"week13/val_re_unet_ep5000_bce_dice_17.nii.gz", saved_img_path+"week13/val_re_unet_ep5000_dice_bce_17.nii.gz", saved_img_path+"week13/val_re_unet_ep5000_dice_tver_17.nii.gz"]

    vals_list = []
    for i in range(len(eval_img_list)):
        out_img = nib.load(eval_img_list[i]).get_fdata()
        scores_itm = eval_scores(out_img, ground_truth.round())
        vals = scores_itm()
        print(vals)
        vals += vals[:1]
        vals_list.append(vals)

    categories = ['Dice', 'Jaccard', 'Vol_Sim', 'Norm_MI', 'ARI']
    angles = [n / float(5) * 2 * np.pi for n in range(5)]
    angles += angles[:1]
    label_list = ["bce_dice", "dice_bce", "dice_tver"]
    # Initialise the spider plot
    plt.figure()
    ax = plt.subplot(111, polar=True)
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='black', size=8)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6,0.8], ["0.2","0.4","0.6","0.8"], color="grey", size=7)
    plt.ylim(0,1)
    
    # Plot data
    for j in range(len(eval_img_list)):
        ax.plot(angles, vals_list[j], color=f"C{j}", linewidth=1, linestyle='solid', label=label_list[j])
        # Fill area
        ax.fill(angles, vals_list[j], color=f"C{j}", alpha=0.1)
    
    # Show the graph
    plt.legend()
    plt.show()

