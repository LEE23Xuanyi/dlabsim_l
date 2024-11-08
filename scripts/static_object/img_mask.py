import os
import cv2
import json
import numpy as np

from dlabsim import DLABSIM_ROOT_DIR

left_button_down = False
box = []

def mouseCallback(event, x, y, flags, param):
    global left_button_down, box
    if not left_button_down and flags == cv2.EVENT_FLAG_LBUTTON:
        box = [(x,y), (x,y)]
        left_button_down = True
    elif left_button_down and flags == cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE:
        box[1] = (x,y)
    else:
        left_button_down = False

    if flags == cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE:
        pass
    elif flags == cv2.EVENT_FLAG_RBUTTON and event == cv2.EVENT_MOUSEMOVE:
        pass
    elif flags == cv2.EVENT_FLAG_MBUTTON and event == cv2.EVENT_MOUSEMOVE:
        pass


if __name__ == "__main__":
    object_name = "jimu"
    save_dir = os.path.join(DLABSIM_ROOT_DIR, f"data/object/{object_name}")

    img_forder = os.path.join(save_dir, "images")
    img_files = sorted(os.listdir(img_forder))
    imgs = [cv2.imread(os.path.join(img_forder, img_file)) for img_file in img_files]
    masked_imgs = [img.copy() for img in imgs]
    mask_range_dict = {}
    obj_imgs = [None] * len(imgs)

    mask_img_forder = os.path.join(save_dir, "mask_images")
    if not os.path.exists(mask_img_forder):
        os.makedirs(mask_img_forder)
    else:
        os.system(f"rm -r {mask_img_forder}")
        os.makedirs(mask_img_forder)

    obj_img_forder = os.path.join(save_dir, "object_images")
    if not os.path.exists(obj_img_forder):
        os.makedirs(obj_img_forder)
    else:
        os.system(f"rm -r {obj_img_forder}")
        os.makedirs(obj_img_forder)

    width, height = imgs[0].shape[1], imgs[0].shape[0]

    cv_windowname = "img"
    cv2.namedWindow(cv_windowname, cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow(cv_windowname)
    cv2.resizeWindow(cv_windowname, width, height)
    cv2.setMouseCallback(cv_windowname, mouseCallback)
    
    if_save = False
    img_id = 0
    while True:
        if len(box) == 2:
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask[box[0][1]:box[1][1], box[0][0]:box[1][0],:] = 1
            img_show = imgs[img_id] * mask
        else:
            img_show = imgs[img_id]

        cv2.imshow(cv_windowname, img_show)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("s"):
            if if_save:
                for i, masked_img in enumerate(masked_imgs):
                    cv2.imwrite(os.path.join(mask_img_forder, img_files[i]), masked_img)
                for i, obj_img in enumerate(obj_imgs):
                    cv2.imwrite(os.path.join(obj_img_forder, img_files[i]), obj_img)
                with open(os.path.join(save_dir, "mask_range.json"), "w") as fp:
                    json.dump(mask_range_dict, fp)
                break
            else:
                print("Please mask all the images")
        elif key == 84:
            masked_imgs[img_id] = img_show.copy()
            obj_imgs[img_id] = img_show[box[0][1]:box[1][1], box[0][0]:box[1][0],:]
            mask_range_dict[img_files[img_id]] = box
            img_id += 1
            if img_id >= len(imgs):
                if_save = True
                print("All images masked")
            img_id = min(len(imgs)-1, img_id)
        elif key == 82:
            masked_imgs[img_id] = img_show.copy()
            obj_imgs[img_id] = img_show[box[0][1]:box[1][1], box[0][0]:box[1][0],:]
            mask_range_dict[img_files[img_id]] = box
            img_id -= 1
            img_id = max(0, img_id)
        elif key == -1:
            pass
        else:
            print(key)