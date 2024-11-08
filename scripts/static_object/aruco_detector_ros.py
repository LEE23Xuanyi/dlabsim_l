import cv2
import sys
import numpy as np
from scipy.spatial.transform import Rotation

import rospy
from geometry_msgs.msg import Pose, PoseArray

from dlabsim.scripts.aruco_tools.pose_from_aruco import ArucoDetector


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python3 aruco_detector_ros.py [grpc_cam_name]")
        sys.exit(1)
    elif sys.argv[1] in ["left", "right", "head"]:
        grpc_cam_name = sys.argv[1]
    else:
        print("Usage: python3 aruco_detector_ros.py [grpc_cam_name]\n(grpc_cam_name: left, right, head)")
        sys.exit(1)

    if grpc_cam_name == "right":
        # orangepi@192.168.11.200
        # right hand camera
        fx = 605.4401
        fy = 605.13804
        cx = 315.97933
        cy = 237.09578

        k1 =  0.04842
        k2 =  0.32434
        p1 = -0.00014
        p2 =  0.00221
        k3 = -1.39195
    elif grpc_cam_name == "head":
        # orangepi@192.168.11.200
        # head camera
        fx = 607.61376953125
        fy = 607.174560546875
        cx = 315.6040954589844
        cy = 259.5807189941406

        k1 =  0.08828
        k2 = -0.00769
        p1 =  0.00005
        p2 =  0.00059
        k3 = -0.56199

    camera_intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    camera_distCoeffs = np.array([k1, k2, p1, p2, k3])
    marker_size = 0.03 # meter

    det = ArucoDetector(cv2.aruco.DICT_4X4_50, camera_intrinsic, camera_distCoeffs, marker_size)

    import sys
    sys.path.append("/home/tatp/ws/mmk2_demo/mmk2_grpc")
    from mmk2_client import MMK2

    robot = MMK2("mmk2", -1, "192.168.11.200")

    rospy.init_node(f"detect_{grpc_cam_name}_aruco")
    aurco_pose_puber = rospy.Publisher(f"{grpc_cam_name}_detect_aruco", PoseArray, queue_size=10)

    cv2.namedWindow(f"{grpc_cam_name}_img")
    while True:
        color_img = robot.get_image(grpc_cam_name, "color")
        img, Rmats, tvecs, ids, _ = det.detect(color_img.copy())

        if not ids is None and len(ids):
            pose_array_msg = PoseArray()
            pose_array_msg.header.stamp = rospy.Time.now()
            pose_array_msg.header.frame_id = ""
            pose_array_msg.poses = []
            for i in range(len(ids)):
                pose_array_msg.header.frame_id += f"{ids[i][0]} "
                p = Pose()
                p.position.x = tvecs[i][0]
                p.position.y = tvecs[i][1]
                p.position.z = tvecs[i][2]
                quat = Rotation.from_matrix(Rmats[i]).as_quat()
                p.orientation.x = quat[0]
                p.orientation.y = quat[1]
                p.orientation.z = quat[2]
                p.orientation.w = quat[3]
                pose_array_msg.poses.append(p)
            aurco_pose_puber.publish(pose_array_msg)

        cv2.imshow(f"{grpc_cam_name}_img", img)
        key = cv2.waitKey(50)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()