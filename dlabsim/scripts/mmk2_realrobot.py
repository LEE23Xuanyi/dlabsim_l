import time
import json
import mujoco
import requests
import numpy as np
from scipy.spatial.transform import Rotation

import rospy
from geometry_msgs.msg import PolygonStamped, PoseArray

from dlabsim.mmk2 import MMK2FIK
from dlabsim.envs import SimulatorGSBase
from dlabsim.envs.mmk2_base import MMK2Cfg
from dlabsim.scripts.mmk2_joy import MMK2JOY
from dlabsim.scripts.hand_detect.hand_detect import HandDetect
from dlabsim.utils import get_site_tmat, get_body_tmat

ARROW_KEY_DICT = {
    "up"    : 82,
    "down"  : 84,
    "left"  : 81,
    "right" : 83,
    "backspace" : 8,
    "enter" : 13,
    "zero" : 48,
}

class MMK2RealRobot(MMK2JOY):
    def __init__(self, cfg):
        super().__init__(cfg)

        import sys
        sys.path.append("/home/ghz/Work/Demos/mmk2_demo/mmk2_grpc")
        from mmk2_client import MMK2
        self.real_robot = MMK2("mmk2", -1, "192.168.11.200")
        
        slamtec_ip = "192.168.11.1"
        self.slamtec_url = f"http://{slamtec_ip}:1448/api/core/slam/v1/localization/pose"
        self.prize_polygon_msg = None

    def get_current_states(self):
        all_joints = self.real_robot.get_all_joint_states()
        head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints
        return head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw

    def update_prize_pose(self):
        if self.prize_polygon_msg is None:
            return

        ps_cnt = np.zeros(3, dtype=np.int32)
        logo_cnt = 0
        bcircle_cnt = 0

        nps = self.prize_polygon_msg.header.frame_id.split(" ")
        nps = [n for n in nps if n != ""]

        for i, n in enumerate(nps):
            posi_wrt_camera = np.array([
                self.prize_polygon_msg.polygon.points[i].x,
                self.prize_polygon_msg.polygon.points[i].y,
                self.prize_polygon_msg.polygon.points[i].z,
                1.0
            ])
            if n == "first_prize":
                ps_cnt[0] += 1
                if ps_cnt[0] > self.ps_cnt_max[0]:
                    continue
                self.mj_model.body(f"p1_{ps_cnt[0]:02d}").pos[:] = (self.tmat_cam_head @ posi_wrt_camera)[:3]
                self.mj_model.body(f"p1_{ps_cnt[0]:02d}").quat[:] = self.mj_data.body("mmk2").xquat[:]
            elif n == "second_prize":
                ps_cnt[1] += 1
                if ps_cnt[1] > self.ps_cnt_max[1]:
                    continue
                self.mj_model.body(f"p2_{ps_cnt[1]:02d}").pos[:] = (self.tmat_cam_head @ posi_wrt_camera)[:3]
                self.mj_model.body(f"p2_{ps_cnt[1]:02d}").quat[:] = self.mj_data.body("mmk2").xquat[:]
            elif n == "third_prize":
                ps_cnt[2] += 1
                if ps_cnt[2] > self.ps_cnt_max[2]:
                    continue
                self.mj_model.body(f"p3_{ps_cnt[2]:02d}").pos[:] = (self.tmat_cam_head @ posi_wrt_camera)[:3]
                self.mj_model.body(f"p3_{ps_cnt[2]:02d}").quat[:] = self.mj_data.body("mmk2").xquat[:]
            elif n == "white_airbot":
                logo_cnt += 1
                if logo_cnt > self.white_airbot_logo_cnt:
                    continue
                self.mj_model.body(f"white_airbot_logo_{logo_cnt:03d}").pos[:] = (self.tmat_cam_head @ posi_wrt_camera)[:3]
                self.mj_model.body(f"white_airbot_logo_{logo_cnt:03d}").quat[:] = self.mj_data.body("mmk2").xquat[:]
            elif n == "blue_circle":
                bcircle_cnt += 1
                if bcircle_cnt > self.blue_circle_cnt:
                    continue
                self.mj_model.body(f"blue_circle_{bcircle_cnt:03d}").pos[:] = (self.tmat_cam_head @ posi_wrt_camera)[:3]
                self.mj_model.body(f"blue_circle_{bcircle_cnt:03d}").quat[:] = self.mj_data.body("mmk2").xquat[:]

        for ei, (i, imax) in enumerate(zip(ps_cnt, self.ps_cnt_max)):
            for j in range(i, imax):
                self.mj_model.body(f"p{ei+1}_{j+1:02d}").pos[2] = -0.3

        for i in range(logo_cnt, self.white_airbot_logo_cnt):
            self.mj_model.body(f"white_airbot_logo_{i+1:03d}").pos[2] = -0.3
        
        for i in range(bcircle_cnt, self.blue_circle_cnt):
            self.mj_model.body(f"blue_circle_{i+1:03d}").pos[2] = -0.3

    def prize_polygon_callback(self, msg: PolygonStamped):
        self.prize_polygon_msg = msg
        self.update_prize_pose()
    
    def update_aruco_pose(self, cam:str):
        if not cam in {"left", "right", "head"}:
            return
        elif eval(f"self.{cam}_aurco_posearray_msg") is None:
            return

        pose_msg = eval(f"self.{cam}_aurco_posearray_msg")
        tmat_cam = eval(f"self.tmat_cam_{cam}")
        nps = pose_msg.header.frame_id.split(" ")
        nps = [n for n in nps if n != ""]
        id_up_set = set()
        for i, n in enumerate(nps):
            if int(n) in self.aruco_head_ids:
                id_up_set.add(int(n))
                posi_wrt_camera = np.eye(4)
                posi_wrt_camera[:3,:3] = Rotation.from_quat([
                    pose_msg.poses[i].orientation.x,
                    pose_msg.poses[i].orientation.y,
                    pose_msg.poses[i].orientation.z,
                    pose_msg.poses[i].orientation.w,
                ]).as_matrix()
                posi_wrt_camera[:3,3] = np.array([
                    pose_msg.poses[i].position.x,
                    pose_msg.poses[i].position.y,
                    pose_msg.poses[i].position.z,
                ])
                tmat_world2arupose = tmat_cam @ posi_wrt_camera
                rot_quat = Rotation.from_matrix(tmat_world2arupose[:3,:3]).as_quat()
                self.mj_model.body(f"aruco_{cam}_{int(n):03d}").pos[:] = tmat_world2arupose[:3,3]
                self.mj_model.body(f"aruco_{cam}_{int(n):03d}").quat[:] = rot_quat[[3,0,1,2]]
        for n in eval(f"self.aruco_{cam}_ids") - id_up_set:
            self.mj_model.body(f"aruco_{cam}_{int(n):03d}").pos[2] = -0.1

    def update_cam_site(self):
        self.tmat_cam_head = get_site_tmat(self.mj_data, "headeye")
        self.tmat_cam_left = get_site_tmat(self.mj_data, "lft_handeye")
        self.tmat_cam_right = get_site_tmat(self.mj_data, "rgt_handeye")

    def head_aruco_pose_callback(self, msg:PoseArray):
        self.head_aurco_posearray_msg = msg
        self.update_aruco_pose("head")

    def left_aruco_pose_callback(self, msg:PoseArray):
        self.left_aurco_posearray_msg = msg
        self.update_aruco_pose("left")

    def right_aruco_pose_callback(self, msg:PoseArray):
        self.right_aurco_posearray_msg = msg
        self.update_aruco_pose("right")

    def set_robot_spine(self):
        print(f"<MMK2> set_robot_spine:{self.mj_slide_joint[0]:.2f}")
        self.mj_slide_joint[0] = np.clip(self.mj_slide_joint[0], self.mj_model.joint("slide_joint").range[0], self.mj_model.joint("slide_joint").range[1])
        self.real_robot.set_spine(self.mj_slide_joint[0])
        for _ in range(30):
            if abs(self.real_robot.spine_position - self.mj_slide_joint[0]) < 1e-2:
                break
            else:
                time.sleep(0.1)
        else:
            print(f"\033[0;31;40mset spine failed: target=({self.mj_slide_joint[0]}) current=({self.real_robot.spine_position})\033[0m")
            self.mj_slide_joint[0] = self.real_robot.spine_position
    
    def set_robot_head(self):
        print("<MMK2> set_robot_head:", np.array(self.mj_head_joint))
        self.real_robot.set_head(self.mj_head_joint.tolist())
        for _ in range(10):
            all_joints = self.real_robot.get_all_joint_states()
            head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints
            if np.allclose(head, self.mj_head_joint, atol=5e-2):
                return True
            else:
                time.sleep(0.1)
        else:
            print(f"\033[0;31;40mset head failed: target=({self.mj_head_joint}) current=({head})\033[0m")

    def check_arm_joints(self, arm:str, joints:list):
        all_joints = self.real_robot.get_all_joint_states()
        head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints
        if arm == "l" and np.allclose(left_arm, joints, atol=5e-2):
            return True
        elif arm == "r" and np.allclose(right_arm, joints, atol=5e-2):
            return True
        elif arm == "a" and np.allclose(left_arm, joints[0], atol=5e-2) and np.allclose(right_arm, joints[1], atol=5e-2):
            return True
        else:
            return False
        
    def check_both_arms(self, left_joints, right_joints):
        for _ in range(20):
            if self.check_arm_joints("a", [left_joints, right_joints]):
                return True
            else:
                time.sleep(0.1)
        else:
            raise ValueError(f"set_robot_arm joints:{left_joints}, {right_joints} failed")

    def set_robot_arm(self, arm:str, joints:list, exec=True, wait=True):
        print("<MMK2> set_robot_arm:", arm, np.array(joints))
        self.real_robot.set_arms(joints, arm, False)
        if exec:
            self.real_robot.execute_trajectory(True)
            if wait:
                for _ in range(20):
                    if self.check_arm_joints(arm, joints):
                        return True
                    else:
                        time.sleep(0.1)
                else:
                    raise ValueError(f"set_robot_arm '{arm}' joints:{joints} failed")
            else:
                return True

    def send_cmd(self, spine_first=True):
        if spine_first:
            self.set_robot_spine()
            self.set_robot_head()
            self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
            self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
            self.real_robot.execute_trajectory(True)
            self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())
            self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)
        else:
            self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
            self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
            self.real_robot.execute_trajectory(True)
            self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())
            self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)
            self.set_robot_spine()
            self.set_robot_head()

    def sysc_real_robot(self, show_msg=True):
        all_joints = self.real_robot.get_all_joint_states()
        head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints

        if show_msg:
            print("-" * 50)
            print("xy_yaw        : ", np.array(xy_yaw))
            print("head          : ", np.array(head))
            print("lift          : ", np.array(lift))
            print("left_arm      : ", np.array(left_arm))
            print("right_arm     : ", np.array(right_arm))
            print("left_gripper  : ", np.array(left_gripper))
            print("right_gripper : ", np.array(right_gripper))

        self.mj_base[:2] = xy_yaw[:2]
        self.mj_base[[4,5,6,3]] = Rotation.from_euler('zyx', [xy_yaw[2], 0.0, 0.0]).as_quat()

        self.mj_slide_joint[0] = lift[0]
        self.mj_left_arm_joint[:] = left_arm
        self.mj_right_arm_joint[:] = right_arm
        self.mj_head_joint[:] = head
        self.mj_left_gripper_joint[0] = left_gripper[0] * 0.04
        self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
        self.mj_right_gripper_joint[0] = right_gripper[0] * 0.04
        self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]

        mujoco.mj_forward(self.mj_model, self.mj_data)

        tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
        lft_posi_homo = np.append(self.mj_data.site("lft_endpoint").xpos.copy(), 1)
        rgt_posi_homo = np.append(self.mj_data.site("rgt_endpoint").xpos.copy(), 1)

        self.lft_arm_target_pose[:] = (np.linalg.inv(tmat_mmk2) @ lft_posi_homo)[:3]
        rmat_arm_end  = self.mj_data.site("lft_endpoint").xmat.reshape((3,3))
        rmat_arm_base = Rotation.from_quat(self.mj_data.body("lft_arm_base").xquat[[1,2,3,0]]).as_matrix()
        arm_base_2_arm_end = rmat_arm_base.T @ rmat_arm_end
        self.lft_end_euler[:] = Rotation.from_matrix(MMK2FIK().action_rot[self.arm_action]["l"].T @ arm_base_2_arm_end).as_euler("zyx")

        self.rgt_arm_target_pose[:] = (np.linalg.inv(tmat_mmk2) @ rgt_posi_homo)[:3]
        rmat_arm_end  = self.mj_data.site("rgt_endpoint").xmat.reshape((3,3))
        rmat_arm_base = Rotation.from_quat(self.mj_data.body("rgt_arm_base").xquat[[1,2,3,0]]).as_matrix()
        arm_base_2_arm_end = rmat_arm_base.T @ rmat_arm_end
        self.rgt_end_euler[:] = Rotation.from_matrix(MMK2FIK().action_rot[self.arm_action]["r"].T @ arm_base_2_arm_end).as_euler("zyx")

        self.update_cam_site()
        self.update_prize_pose()

    def base_move(self, linear_vel, angular_vel):
        if hasattr(self, "last_lv"):
            if abs(linear_vel) < 1e-3 and abs(angular_vel) < 1e-3 and abs(self.last_lv) < 1e-3 and abs(self.last_av) < 1e-3:
                return
        linear_vel = np.clip(linear_vel, -0.3, 0.3)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        self.real_robot.set_base_vel([linear_vel, angular_vel])
        self.last_lv = linear_vel
        self.last_av = angular_vel

    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        if key == ord("u"):
            self.sysc_real_robot()
        elif key == ord(" "):
            self.send_cmd()
            self.update_cam_site()
        elif key == ARROW_KEY_DICT["up"]:
            self.slamtec_simple_move(0.2, 0.0)
        elif key == ARROW_KEY_DICT["down"]:
            self.slamtec_simple_move(-0.2, 0.0)
        elif key == ARROW_KEY_DICT["left"]:
            self.slamtec_simple_move(0.0, np.pi/6)
        elif key == ARROW_KEY_DICT["right"]:
            self.slamtec_simple_move(0.0, -np.pi/6)
        elif key == ord("-"):
            self.slamtec_lateral_translation(0.3)
        elif key == ord("="):
            self.slamtec_lateral_translation(-0.3)
        return ret

    def teleopProcess_view(self):
        super().teleopProcess_view()
        if self.teleop.get_raising_edge(0): # A
            self.send_cmd()
            self.update_cam_site()
            self.update_prize_pose()
        if self.teleop.get_raising_edge(1): # B
            self.sysc_real_robot()
        if self.teleop.get_raising_edge(2): # X
            self.running = False

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("mj_data.qpos :")
        print("    base      = {}".format(np.array2string(self.mj_data.qpos[:7], separator=', ')))
        print("    chassis   = {}".format(np.array2string(self.mj_data.qpos[7:9], separator=', ')))
        print("    lift      = {}".format(np.array2string(self.mj_data.qpos[9:10], separator=', ')))
        print("    head      = {}".format(np.array2string(self.mj_data.qpos[10:12], separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.mj_data.qpos[12:19], separator=', ')))
        print("    right arm = {}".format(np.array2string(self.mj_data.qpos[20:27], separator=', ')))

        tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
        lft_posi_homo = np.append(self.mj_data.site("lft_endpoint").xpos.copy(), 1)
        rgt_posi_homo = np.append(self.mj_data.site("rgt_endpoint").xpos.copy(), 1)

        print("    lft_endp  = {}".format((np.linalg.inv(tmat_mmk2) @ lft_posi_homo)[:3]))
        print("    rgt_endp  = {}".format((np.linalg.inv(tmat_mmk2) @ rgt_posi_homo)[:3]))

        print("    lft_ende  = {}".format(self.lft_end_euler))
        print("    rgt_ende  = {}".format(self.rgt_end_euler))

        print("    lft_camera = \n{}".format(get_site_tmat(self.mj_data, "lft_handeye")))
        print("    rgt_camera = \n{}".format(get_site_tmat(self.mj_data, "rgt_handeye")))

        print("    action_carry_height = {}".format(self.action_carry_height))

        print("-" * 50)
        all_joints = self.real_robot.get_all_joint_states()
        head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints
        print("real_robot.get_all_joint_states:")
        print("head          : ", np.array(head))
        print("lift          : ", np.array(lift))
        print("left_arm      : ", np.array(left_arm))
        print("right_arm     : ", np.array(right_arm))
        print("left_gripper  : ", np.array(left_gripper))
        print("right_gripper : ", np.array(right_gripper))
        print("xy_yaw        : ", np.array(xy_yaw))

        # print("-" * 50)
        # for bn in ["aruco_head_000", "aruco_right_000"]:
        #     print(f"{bn}\t: {self.mj_model.body(bn).pos}, {self.mj_model.body(bn).quat}")

        print("-" * 50)
        for sn in ["box_001", "box_002", "box_003"]:
            print(f"{sn}\t {self.mj_data.site(sn).xpos}")
        
        print("-" * 50)
        for i, pmax in enumerate(self.ps_cnt_max):
            if i == "0" and pmax:
                print("-" * 50)
            for j in range(pmax):
                if self.mj_model.body(f'p{i+1}_{j+1:02d}').pos[2] > 0.:
                    print(f"p{i+1}_{j+1:02d} : {self.mj_model.body(f'p{i+1}_{j+1:02d}').pos}, {self.mj_model.body(f'p{i+1}_{j+1:02d}').quat}")
            if pmax:
                print()
        print("-" * 100)

    def get_base_pose(self):
        payload = {}
        headers = {"Content-Type": "application/json"}
        response = requests.request("GET", self.slamtec_url, headers=headers, data=payload)
        pose = json.loads(response.text)
        return [pose["x"], pose["y"], pose["yaw"]]

    def tmp_angle_diff(self, a, b):
        diff = a - b
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def slamtec_simple_move(self, translation:float, rotation:float=0.0):
        # translate first, then rotate
        pose = self.get_base_pose()
        ori_x, ori_y, ori_yaw = pose[0], pose[1], pose[2]

        if abs(translation) > 0.01:
            tar_x = pose[0] + translation * np.cos(pose[2])
            tar_y = pose[1] + translation * np.sin(pose[2])

            for i in range(int(abs(translation) / 0.3 * 20 * 5)):
                move_dist = np.hypot(pose[0] - ori_x, pose[1] - ori_y)
                tar_dist = np.hypot(pose[0] - tar_x, pose[1] - tar_y)
                if i > 40 and move_dist < 0.03:
                    print(f"\033[0;31;40mslamtec_base_move: failed!\033[0m")
                    return
                if tar_dist < 0.02 or move_dist > abs(translation) - 0.01:
                    break
                pose = self.get_base_pose()
                self.base_move(tar_dist * np.sign(translation), 0.0)
                time.sleep(0.05)
            self.base_move(0.0, 0.0)
            print("slamtec_base_move: translation done")
        
        if abs(rotation) > 0.01:
            rotation = np.clip(rotation, -np.pi, np.pi)
            tar_yaw = ori_yaw + rotation
            if tar_yaw > np.pi:
                tar_yaw -= 2 * np.pi
            elif tar_yaw < -np.pi:
                tar_yaw += 2 * np.pi
            for i in range(int(abs(rotation) / 0.5 * 20 * 5)):
                ang_diff = self.tmp_angle_diff(tar_yaw, pose[2])
                if i > 40 and abs(self.tmp_angle_diff(pose[2], ori_yaw)) < 0.02:
                    print(f"\033[0;31;40mslamtec_base_rotate: failed!\033[0m")
                    return
                if abs(ang_diff) < 0.01:
                    break
                pose = self.get_base_pose()
                self.base_move(0.0, ang_diff * 1.5)
                time.sleep(0.05)
            self.base_move(0.0, 0.0)
            print("rotate_control: rotation done")

    def slamtec_lateral_translation(self, translation: float):
        dist = np.abs(translation)/np.sqrt(2)
        if translation > 0.0:
            self.slamtec_simple_move(0.0, -np.pi / 4)
            self.slamtec_simple_move(-dist)
            self.slamtec_simple_move(0.0, np.pi / 2)
            self.slamtec_simple_move(dist)
            self.slamtec_simple_move(0.0, -np.pi / 4)
        else:
            self.slamtec_simple_move(0.0, np.pi / 4)
            self.slamtec_simple_move(-dist)
            self.slamtec_simple_move(0.0, -np.pi / 2)
            self.slamtec_simple_move(dist)
            self.slamtec_simple_move(0.0, np.pi / 4)

    def moveto_backhome(self):
        pose = self.get_base_pose()
        tar_ang = np.arctan2(pose[1], pose[0])
        rot_ang = self.tmp_angle_diff(tar_ang, pose[2])
        if abs(rot_ang) > 0.05:
            self.slamtec_simple_move(0.0, rot_ang)
        dist = np.hypot(pose[0], pose[1])
        if dist > 0.02:
            self.slamtec_simple_move(-dist, 0.0)

    def moveto_fromhome(self, tar_pose):
        pose = self.get_base_pose()
        if np.hypot(pose[0], pose[1]) > 0.02:
            self.moveto_backhome()
            pose = self.get_base_pose()

        for _ in range(3):
            pose = self.get_base_pose()
            tar_ang = np.arctan2(tar_pose[1], tar_pose[0])
            rot_ang = self.tmp_angle_diff(tar_ang, pose[2])
            if abs(rot_ang) > 0.05:
                self.slamtec_simple_move(0.0, rot_ang)
            else:
                break

        pose = self.get_base_pose()
        dist = np.hypot(tar_pose[0] - pose[0], tar_pose[1] - pose[1])
        rot_ang = self.tmp_angle_diff(tar_pose[2], pose[2])
        self.slamtec_simple_move(dist, rot_ang)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    rospy.init_node('mujoco_node', anonymous=True)

    cfg = MMK2Cfg()

    cfg.init_key = "pick"

    cfg.obs_camera_id = -2
    cfg.render_set["fps"] = 25
    cfg.render_set["width"] = 1440
    cfg.render_set["height"] = 1080

    cfg.mjcf_file_path = "mjcf/exhibition_conference.xml"
    cfg.rb_link_list =  [
        "agv_link", "slide_link", "head_yaw_link", "head_pitch_link",
        "lft_arm_base", "lft_arm_link1", "lft_arm_link2",
        "lft_arm_link3", "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
        "lft_finger_left_link", "lft_finger_right_link",
        "rgt_arm_base", "rgt_arm_link1", "rgt_arm_link2",
        "rgt_arm_link3", "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
        "rgt_finger_left_link", "rgt_finger_right_link"
    ]
    cfg.obj_list = []

    exec_node = MMK2RealRobot(cfg)
    if isinstance(exec_node, SimulatorGSBase):
        models_lst = [
            "exhibition/booth.ply",

            "mmk2/mmk2_base/agv_link.ply",
            "mmk2/mmk2_base/slide_link.ply",
            "mmk2/mmk2_base/head_pitch_link.ply",
            "mmk2/mmk2_base/head_yaw_link.ply",

            "mmk2/left_arm/lft_arm_base.ply",
            "mmk2/left_arm/lft_arm_link1.ply",
            "mmk2/left_arm/lft_arm_link2.ply",
            "mmk2/left_arm/lft_arm_link3.ply",
            "mmk2/left_arm/lft_arm_link4.ply",
            "mmk2/left_arm/lft_arm_link5.ply",
            "mmk2/left_arm/lft_arm_link6.ply",
            "mmk2/left_arm/lft_finger_left_link.ply",
            "mmk2/left_arm/lft_finger_right_link.ply",
 
            "mmk2/right_arm/rgt_arm_base.ply",
            "mmk2/right_arm/rgt_arm_link1.ply",
            "mmk2/right_arm/rgt_arm_link2.ply",
            "mmk2/right_arm/rgt_arm_link3.ply",
            "mmk2/right_arm/rgt_arm_link4.ply",
            "mmk2/right_arm/rgt_arm_link5.ply",
            "mmk2/right_arm/rgt_arm_link6.ply",
            "mmk2/right_arm/rgt_finger_left_link.ply",
            "mmk2/right_arm/rgt_finger_right_link.ply"
        ]
        exec_node.init_gs_render(models_lst)

    exec_node.reset()
    while exec_node.running:
        exec_node.view()
