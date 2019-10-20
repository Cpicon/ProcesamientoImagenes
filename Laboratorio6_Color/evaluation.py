import os
import os.path as osp
import skimage.io as io
import numpy as np
import pdb

def evaluation(ann_folder, pred_folder):
    # Function that performs the evaluation of predictions given
    # the path to a folder with annotations and the path to the folder
    # with the predictions.
    # Args:
    #   ann_folder: path to folder with annotations.
    #   pred_folder: path to folder with predictions.
    # Returns:
    #   aiou and miou
    # To use in another python script import as:
    #   from evaluation import evaluation
    ann_names = os.listdir(ann_folder)

    def compute_mask_IU(mask, target):
        assert target.shape[-2:] == mask.shape[-2:]
        temp = mask * target
        intersection = temp.sum()
        union = ((mask + target) - temp).sum()
        return intersection, union

    jaccards = []
    inters = 0
    unions = 0
    for name in ann_names:
        # get names of ann and pred
        ann_name = osp.join(ann_folder, name)
        pred_name = osp.join(pred_folder, name)
        # read ann and pred
        this_ann = io.imread(ann_name)
        this_pred = io.imread(pred_name)
        # remove from ann instance information and convert to float64
        this_ann = (this_ann > 0).astype('float64')
        # make sure pred is binary
        this_pred = (this_pred > 0).astype('float64')
        # conpute intersection and union between ann and pred
        inter, union = compute_mask_IU(this_ann, this_pred)
        jaccards.append(inter/union)
        inters += inter
        unions += union
    # calculate metrics
    aiou = np.mean(np.asarray(jaccards))
    miou = inters/unions
    return aiou, miou