"""
5 Evaluation metrics used to evaluate the resulting segmentation

Marshall @ 03/01/2023
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
import nibabel as nib

class eval_radar:
    def __init__(self, seg, gt) -> None:
        """
        seg, gt: numpy array
        """
        self.seg = seg.reshape(-1,1)
        self.gt = gt.reshape(-1,1) 

        self.tn, self.fp, self.fn, self.tp = confusion_matrix(self.gt, self.seg).ravel()
        self.n = self.tn + self.fp + self.fn + self.tp
    
    def _dice(self):
        # dice score
        return (2*np.sum(self.seg * self.gt)) / (np.sum(self.seg) + np.sum(self.gt))
    
    def _jaccard(self):
        # jaccard score
        return (np.sum(self.seg * self.gt)) / (np.sum(np.logical_or(self.seg, self.gt)))
    
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

        return Hsg + Hst - Hsgst
    
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
    
    def __call__(self, pr=False):
        a = self._dice()
        b = self._jaccard()
        c = self._vol_sim()
        d = self._norm_mi()
        e = self._adjusted_ri()
        categories = ['Dice', 'Jaccard', 'Vol_Sim', 'Norm_MI', 'ARI']
        val_list = [a,b,c,d,e]
        val_list += val_list[:1]

        angles = [n / float(5) * 2 * np.pi for n in range(5)]
        angles += angles[:1]

        if pr == True:
            print(f"Dice score: {a}\n")
            print(f"Jaccard score: {b}\n")
            print(f"Volume Similarity: {c}\n")
            print(f"Mutual Information: {d}\n")
            print(f"Adjusted Rand Index: {e}\n")

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
        ax.plot(angles, val_list, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, val_list, 'b', alpha=0.1)
        # Show the graph
        plt.show()


        return

if __name__ == "__main__":

    test_seg_path = "./saved_image/week11/postpro_15_2.nii.gz" 
    test_seg = nib.load(test_seg_path).get_fdata()

    test_data_path = "./data/all_label/15.nii"
    test_img = nib.load(test_data_path)
    ground_truth = test_img.get_fdata()

    omlette_path = "./data/train_OMELETTElabel/sub015_seg_g0.02.nii.gz"
    omlette_seg = nib.load(omlette_path).get_fdata()

    plausible_path = "./data/train_plausiblelabel/sub015_Segmentation_9.nii.gz"
    plausible_seg = nib.load(plausible_path).get_fdata()

    radar = eval_radar(test_seg, ground_truth.round())

    radar(pr=True)