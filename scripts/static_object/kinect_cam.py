import os
import cv2
import time
import pykinect_azure as pykinect
import numpy as np

from dlabsim import DLABSIM_ROOT_DIR
class k4a_driver:
    def __init__(self, model_path='/usr/lib/x86_64-linux-gnu/libk4a.so'):
        pykinect.initialize_libraries(model_path)
        self.device_config = pykinect.default_configuration
        self.device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
        # self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        self.depth_scale = [0.25, 1.]

        self.device = pykinect.start_device(config=self.device_config)
        cal_rgb = self.device.get_calibration(pykinect.K4A_CALIBRATION_TYPE_COLOR, self.device_config.color_resolution)

    def update(self):
        return self.device.update()

if __name__ == "__main__":
    k4a = k4a_driver()

    cap_img = False
    auto_save = False
    if auto_save:
        cap_img = True

    if cap_img:
        save_dir = os.path.join(DLABSIM_ROOT_DIR, "data/object/jimu/4")
        # save_dir = os.path.join(DLABSIM_ROOT_DIR, "data/object/calibration/kinect_cap")
        if os.path.exists(save_dir):
            os.system(f"rm -r {save_dir}")
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "images"))
        os.makedirs(os.path.join(save_dir, "points"))

    cv2.namedWindow("img")
    cv2.namedWindow("depth")
    
    cnt = 0
    img_id = 0
    last_cap_t = time.time()
    while True:
        capture = k4a.update()
        ret_color, color_img = capture.get_color_image()
        ret_depth, depth_img = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth or color_img is None or depth_img is None:
            continue

        # delta_t = time.time() - last_cap_t
        # cv2.putText(color_img, "FPS: {:.2f}".format(1.0/delta_t), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # last_cap_t = time.time()
        cv2.imshow("img", color_img)

        # Convert depth image to pseudo-color image
        depth_img_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.255), cv2.COLORMAP_JET)
        cv2.imshow("depth", depth_img_color)

        cnt += 1
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif cap_img and key == ord("s") or key == ord(" ") or (auto_save and cnt % 30 == 0):
            ret_points, points = capture.get_pointcloud()
            if ret_points:
                np.save(os.path.join(save_dir, "points/color_{:03d}.npy".format(img_id)), points)
                cv2.imwrite(os.path.join(save_dir, "images/color_{:03d}.png".format(img_id)), color_img)
            img_id += 1
        
        if auto_save and cnt > 30 * 30:
            break

    cv2.destroyAllWindows()
