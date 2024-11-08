import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim.scripts.aruco_tools.pose_from_aruco import ArucoDetector

if __name__ == "__main__":

    fx = 608.39551
    fy = 608.58689
    cx = 316.7065
    cy = 260.29088

    k1 = 0.08828
    k2 = -0.00769
    p1 = 0.00005
    p2 = 0.00059
    k3 = -0.56199

    camera_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera_distCoeffs = np.array([k1, k2, p1, p2, k3])
    marker_size = 0.16 # meter

    det = ArucoDetector(cv2.aruco.DICT_4X4_50, camera_intrinsic, camera_distCoeffs, marker_size)

    img = cv2.imread("/home/tatp/Desktop/color_000.png")
    print(img.shape)

    img, Rmats, tvecs, ids, _ = det.detect(img)
    print(Rmats)
    print(tvecs)
    print(ids)
    
    # [-153.45618542    0.33208447   -0.77442935]
    euler = Rotation.from_matrix(Rmats[0]).as_euler("xyz", degrees=True)
    print(euler)

    cv2.namedWindow("img")
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
