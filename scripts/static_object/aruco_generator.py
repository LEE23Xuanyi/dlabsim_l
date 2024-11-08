import cv2
import numpy as np

if __name__ == '__main__':

    aruco_dict_pre = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

    for i in range(450):
        markerImage = aruco_dict_pre.generateImageMarker(i, 300)
        cv2.imwrite(f"/home/tatp/ws/GreatWall/DLabSim/data/aruco_4_4/{i:03d}.png", markerImage)
        # 50 150 250
        # [0,50)
        # [50,200)
        # [200,450)

