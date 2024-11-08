import cv2
from hand_detect import HandDetect

import sys
sys.path.append("/home/tatp/ws/mmk2_demo/mmk2_grpc")
from mmk2_client import MMK2
robot = MMK2("mmk2", -1, "192.168.11.200")  # 192.168.11.200

if __name__ == "__main__":
    detector = HandDetect()

    grpc_cam_name = "head"
    window_name = f"grpc_{grpc_cam_name}_cam"
    color_image = robot.get_image(grpc_cam_name, "color")

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(window_name, color_image.shape[1], color_image.shape[0])

    while True:
        color_image = robot.get_image(grpc_cam_name, "color")
        print(detector.detect(color_image, False))

        cv2.imshow(window_name, color_image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break