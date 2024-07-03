import cv2
import numpy as np
import os, glob
 
 
def rgb2masks(label_name):
    lbl_id = os.path.split(label_name)[-1].split('.')[0]
    lbl = cv2.imread(label_name, 1)
    h, w = lbl.shape[:2]
    leaf_dict = {}
    idx = 0
    white_mask = np.ones((h, w, 3), dtype=np.uint8) * 255
    for i in range(h):
        for j in range(w):
            if tuple(lbl[i][j]) in leaf_dict or tuple(lbl[i][j]) == (0, 0, 0):
                continue
            leaf_dict[tuple(lbl[i][j])] = idx
            mask = (lbl == lbl[i][j]).all(-1)
            # leaf = lbl * mask[..., None]      # colorful leaf with black background
            # np.repeat(mask[...,None],3,axis=2)    # 3D mask
            leaf = np.where(mask[..., None], white_mask, 0)
            mask_name = './shapes/train/annotations/' + lbl_id + '_leaf_' + str(idx) + '.png'
            cv2.imwrite(mask_name, leaf)
            idx += 1
 
 
label_dir = './labels'
label_list = glob.glob(os.path.join(label_dir, '*.png'))
for label_name in label_list:
    rgb2masks(label_name)