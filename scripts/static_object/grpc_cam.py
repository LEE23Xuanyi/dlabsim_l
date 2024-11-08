import os
import cv2
import numpy as np

import sys
sys.path.append("/home/tatp/ws/mmk2_demo/mmk2_grpc")
from mmk2_client import MMK2
robot = MMK2("mmk2", -1, "192.168.11.200")  # 192.168.11.200

grpc_cam_name = "head"

window_name = f"grpc_{grpc_cam_name}_cam"
color_image = robot.get_image(grpc_cam_name, "color")
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(window_name, color_image.shape[1], color_image.shape[0])

img_id = 0
# save_dir = "/home/tatp/ws/GreatWall/DLabSim/data/object/calibration/checkboard_grpc_right_hand_realsense_435i"
save_dir = "/home/tatp/Desktop/tmp"

while True:
    color_image = robot.get_image(grpc_cam_name, "color")
    depth_image = np.clip(robot.get_image(grpc_cam_name, "depth"), 0, 5.0).astype(np.float32)

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=255/5), cv2.COLORMAP_JET)

    # Stack both images horizontally
    images = np.hstack((color_image, depth_colormap))

    cv2.imshow(window_name, images)

    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break
    elif key == ord("s"):
        cv2.imwrite(f"{save_dir}/color_{img_id:03d}.png", color_image)
        np.save(os.path.join(save_dir, f"depth_{img_id:03d}.npy".format(img_id)), depth_image)
        img_id += 1
