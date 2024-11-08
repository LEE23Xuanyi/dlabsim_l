import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from mmk2_base import MMK2Base, MMK2Cfg

if __name__ == "__main__":
    rospy.init_node("mmk2_base_rosimg")

    cfg = MMK2Cfg()
    cfg.obs_camera_id = -1
    exec_node = MMK2Base(cfg)

    # Convert bgr_img to ROS Image message
    bridge = CvBridge()
    img_pub = rospy.Publisher('image_topic', Image, queue_size=10)

    obs = exec_node.reset()
    print(obs.keys())

    action_list = np.zeros(19)
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action_list)
        
        ros_img = bridge.cv2_to_imgmsg(obs["img"], "rgb8")
        img_pub.publish(ros_img)
