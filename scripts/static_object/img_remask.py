import os
import cv2
import json
import numpy as np

from dlabsim import DLABSIM_ROOT_DIR

if __name__ == "__main__":
    object_name = "jimu"
    save_dir = os.path.join(DLABSIM_ROOT_DIR, f"data/object/{object_name}")

    obj_img_forder = os.path.join(save_dir, "rembg_images")
    obj_img_files = sorted(os.listdir(obj_img_forder))
    imgs = [cv2.imread(os.path.join(obj_img_forder, img_file)) for img_file in obj_img_files]

    save_img_forder = os.path.join(save_dir, "remask_images")
    if os.path.exists(save_img_forder):
        os.system(f"rm -r {save_img_forder}")
    os.makedirs(save_img_forder)

    with open(os.path.join(save_dir, "mask_range.json"), "r") as fp:
        mask_pose_dict = json.load(fp)
    
    for key in mask_pose_dict:
        obj_img = cv2.imread(os.path.join(obj_img_forder, key.replace(".jpg", ".png")))
        bg_img = np.zeros((1080, 1920, 3), np.uint8)

        box = mask_pose_dict[key]
        bg_img[box[0][1]:box[1][1], box[0][0]:box[1][0], :] = obj_img

        cv2.imwrite(os.path.join(save_img_forder, key), bg_img)
