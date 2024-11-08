import numpy as np
from scipy.spatial.transform import Rotation

import time
import rospy
from geometry_msgs.msg import PolygonStamped, PoseArray

from dlabsim.mmk2 import MMK2FIK
from dlabsim.utils import get_body_tmat
from dlabsim.envs import SimulatorGSBase
from dlabsim.envs.mmk2_base import MMK2Cfg
from dlabsim.scripts.mmk2_realrobot import MMK2RealRobot
from dlabsim.scripts.hand_detect.hand_detect import HandDetect
import dlabsim.scripts.demo_config as cfg
from copy import copy


class MMK2Exhibition(MMK2RealRobot):
    target_yaw = -np.pi*0.5
    box_desk_position_map = {
        'first_prize' : [ 0.0, -0.45, target_yaw],
        'second_prize': [ 0.4, -0.45, target_yaw],
        'third_prize' : [-0.4, -0.45, target_yaw]
    }

    box_shelf_position_map = {
        1 : [0.45, 0.33, 0.0],
        2 : [0.45, 0.02, 0.0],
    }

    map_prize_id_2_name = {
        0: 'first_prize',
        1: 'second_prize',
        2: 'third_prize'
    }

    table_height = 0.74
    # set_action_carry_height = 0.11
    set_action_carry_height = 0.42
    # set_action_carry_height = 0.74
    # set_action_carry_height = 1.06

    PickupExhibitionStateMachineKeyPose = cfg.sm_keypose

    tmatarm_mmk2sdk_2_mujoco = np.linalg.inv(np.array([
        [ 1., 0., 0., -0.016],
        [ 0., 1., 0., -0.002],
        [ 0., 0., 1., -0.004],
        [ 0., 0., 0.,  1.   ],
    ]))

    stay_normal_state = {"l":False, "r":False}
    last_action = None
    at_desk = True

    forward_distance = 0.25

    pick_prize_select = 0
    pick_prize_select_front_pose = None

    action_carry_joy = False
    action_carry_height = set_action_carry_height
    pick_box_select = 0
    
    pick_box_pose = None
    holding_box = True

    def __init__(self, cfg:MMK2Cfg):
        super().__init__(cfg)
        self.sysc_real_robot()
        self.hand_detector = HandDetect()

        self.ps_cnt_max = np.zeros(3, dtype=np.int32)
        self.white_airbot_logo_cnt = 0
        self.blue_circle_cnt = 0

        name_lst = self.mj_model.names.decode("utf-8").split("\x00")
        self.aruco_head_ids = set()
        self.aruco_left_ids = set()
        self.aruco_right_ids = set()
        for n in name_lst:
            if n.startswith("p1_"):
                self.ps_cnt_max[0] += 1
            elif n.startswith("p2_"):
                self.ps_cnt_max[1] += 1
            elif n.startswith("p3_"):
                self.ps_cnt_max[2] += 1
            elif n.startswith("aruco_head_"):
                self.aruco_head_ids.add(int(n.split("_")[-1]))
            elif n.startswith("aruco_left_"):
                self.aruco_left_ids.add(int(n.split("_")[-1]))
            elif n.startswith("aruco_right_"):
                self.aruco_right_ids.add(int(n.split("_")[-1]))
            elif n.startswith("white_airbot_logo_"):
                self.white_airbot_logo_cnt += 1
            elif n.startswith("blue_circle_"):
                self.blue_circle_cnt += 1

        self.update_cam_site()
        self.prize_polygon_msg = None
        self.head_aurco_posearray_msg = None
        self.left_aurco_posearray_msg = None
        self.right_aurco_posearray_msg = None

        print("lft_arm_target_pose = \n", self.lft_arm_target_pose)
        print("rgt_arm_target_pose = \n", self.rgt_arm_target_pose)

        rospy.Subscriber("/detect_prize", PolygonStamped, self.prize_polygon_callback,)
        # rospy.Subscriber("/head_detect_aruco", PoseArray, self.head_aruco_pose_callback)
        # rospy.Subscriber("/left_detect_aruco", PoseArray, self.left_aruco_pose_callback)
        # rospy.Subscriber("/right_detect_aruco", PoseArray, self.right_aruco_pose_callback)

    def select_blue_circle(self, circle_height = 1.5):
        posis = []
        dists = []
        for i in range(self.blue_circle_cnt):
            posi = self.mj_model.body(f"blue_circle_{(i+1):03d}").pos[:]
            if posi[2] > 0.0 and abs(posi[2] - circle_height) < 0.05:
                point3d_homogeneous = np.append(posi, 1)
                tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
                posi_local = (np.linalg.inv(tmat_mmk2) @ point3d_homogeneous)[:3]
                posis.append(posi_local.tolist())
                dists.append(np.hypot(posi_local[0], posi_local[1]))
        if len(posis):
            pposi = posis[dists.index(min(dists))]
            print(f"blue circle position local: {np.array(pposi)}")
            return pposi
        else:
            print("No blue circle detected")
            return None

    def select_box_nearest(self, plant_height):
        posis = []
        dists = []
        for i in range(self.white_airbot_logo_cnt):
            posi = self.mj_data.site(f"box_{(i+1):03d}").xpos
            if posi[2] > 0.0 and abs(posi[2] - 0.05 - plant_height) < 0.05:
                point3d_homogeneous = np.append(posi, 1)
                tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
                posi_local = (np.linalg.inv(tmat_mmk2) @ point3d_homogeneous)[:3]
                posis.append(posi_local.tolist())
                dists.append(np.hypot(posi_local[0], posi_local[1]))
        if len(posis):
            pposi = posis[dists.index(min(dists))]
            print(f"box position local: {np.array(pposi)}")
            return pposi
        else:
            print("No box detected")
            return None

    def select_box(self, box_id:int):
        # :TODO: 不区分id 选择最近的箱子
        # [0,1,2]: 一二三等奖的箱子
        self.pick_box_select = box_id
        print(f"choose box[0,1,2]: {self.pick_box_select}")
        posi = self.mj_data.site(f"box_{(self.pick_box_select+1):03d}").xpos
        if posi[2] > 0.0:
            point3d_homogeneous = np.append(posi, 1)
            tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
            pposi = (np.linalg.inv(tmat_mmk2) @ point3d_homogeneous)[:3]
            print(f"box position local: {pposi}")
            return pposi.tolist()
        else:
            print("No box detected")
            return None
    
    def select_prize(self, prize_id:int):
        # [0,1,2]: 一二三等奖
        self.pick_prize_select = prize_id
        print(f"choose prize {self.pick_prize_select} [0,1,2]")
        posis = []
        tmat_mmk2 = get_body_tmat(self.mj_data, "mmk2")
        for i in range(self.ps_cnt_max[self.pick_prize_select]):
            posi = self.mj_model.body(f"p{(self.pick_prize_select+1)}_{(i+1):02d}").pos
            posi_homogeneous = np.append(posi, 1)
            posi_local = (np.linalg.inv(tmat_mmk2) @ posi_homogeneous)[:3]
            if posi_local[2] > 0.5:
                posis.append(posi_local.tolist())
        if len(posis):
            pposi = sorted(posis, key = lambda x:x[0])[0]
            print(f"prize position local: {np.array(pposi)}")
            return pposi
        else:
            print("No prize detected")
            return None

    def action_look(self):
        if self.arm_action == "pick":

            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["look"]
            self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["look"]

            self.lft_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["left_arm"]["look"]
            self.rgt_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["right_arm"]["look"]

            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint)
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint)

            self.mj_left_gripper_joint[0] = self.mj_left_gripper_joint[1] = 0.0
            self.mj_right_gripper_joint[0] = self.mj_right_gripper_joint[1] = 0.0

        elif self.arm_action == "carry":
            
            spain_h = 1.4 - self.action_carry_height

            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["carry"] + spain_h
            self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["carry"]

            self.lft_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["left_arm"]["carry"]
            self.lft_arm_target_pose[2] -= spain_h
            self.rgt_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["right_arm"]["carry"]
            self.rgt_arm_target_pose[2] -= spain_h
            self.lft_end_euler[:] = [ 0.978, 0.0,  1.924]
            self.rgt_end_euler[:] = [-0.978, 0.0, -1.924]

            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint, Rotation.from_euler('zyx', self.lft_end_euler).as_matrix())
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', self.rgt_end_euler).as_matrix())

            self.mj_left_gripper_joint[0] = self.mj_left_gripper_joint[1] = 0.0
            self.mj_right_gripper_joint[0] = self.mj_right_gripper_joint[1] = 0.0

        self.send_cmd(spine_first=False)
        self.sysc_real_robot()
    
    def descartes_control(self, arm="a", exec=True, check=True):
        if arm == "l" or arm == "a":
            tmat_lft_target = self.tmatarm_mmk2sdk_2_mujoco @ self.arm_fik.properFK(self.mj_left_arm_joint)
            lft_target_quat = Rotation.from_matrix(tmat_lft_target[:3,:3]).as_quat()
            lft_target = tmat_lft_target[:3,3].tolist() + lft_target_quat.tolist()
            self.real_robot.set_arms(lft_target, "l", True)
        if arm == "r" or arm == "a":
            tmat_rgt_target = self.tmatarm_mmk2sdk_2_mujoco @ self.arm_fik.properFK(self.mj_right_arm_joint)
            rgt_target_quat = Rotation.from_matrix(tmat_rgt_target[:3,:3]).as_quat()
            rgt_target = tmat_rgt_target[:3,3].tolist() + rgt_target_quat.tolist()
            self.real_robot.set_arms(rgt_target, "r", True)
        if exec:
            self.real_robot.execute_trajectory(True)
            if check:
                if arm == "a":
                    self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())
                else:
                    if arm == 'l':
                        arm_joints = self.mj_left_arm_joint.tolist()
                    elif arm == "r":
                        arm_joints = self.mj_right_arm_joint.tolist()
                    for _ in range(20):
                        if self.check_arm_joints(arm, arm_joints):
                            return True
                        else:
                            time.sleep(0.1)
                    else:
                        raise ValueError(f"set_robot_arm {arm} joints:{arm_joints} failed")

    def action_carry(self, plant_height:float):
        if not self.pick_box_pose is None:
            # 调节抓取高度
            self.mj_slide_joint[0] = 0.89 - plant_height
            self.set_robot_spine()

            # 调整夹爪开口
            self.mj_left_gripper_joint[0] = 0.04
            self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
            self.mj_right_gripper_joint[0] = 0.04
            self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
            self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")    
            self.real_robot.execute_trajectory(True)

            # 控制机械臂到身前
            self.lft_arm_target_pose[:] = copy(self.pick_box_pose)
            self.rgt_arm_target_pose[:] = copy(self.pick_box_pose)
            self.lft_arm_target_pose[0] -= 0.2275
            self.rgt_arm_target_pose[0] -= 0.2275
            self.lft_arm_target_pose[1] += 0.145
            self.rgt_arm_target_pose[1] -= 0.145
            self.lft_arm_target_pose[2] = plant_height + 0.185 + 0.01 # :FINETUNE:
            self.rgt_arm_target_pose[2] = plant_height + 0.185 + 0.0  # :FINETUNE:
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
            self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
            self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
            self.real_robot.execute_trajectory(True)
            self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())

            # 控制机械臂到carry位置
            self.lft_arm_target_pose[0] = self.pick_box_pose[0] + 0.0625
            self.rgt_arm_target_pose[0] = self.pick_box_pose[0] + 0.0625
            self.lft_arm_target_pose[1] -= 0.06 # :FINETUNE:
            self.rgt_arm_target_pose[1] += 0.055  # :FINETUNE:
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
            self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
            self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
            self.real_robot.execute_trajectory(True)
            self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())

            # 夹住
            self.mj_left_gripper_joint[1] = self.mj_left_gripper_joint[0] = 0.0
            self.mj_right_gripper_joint[1] = self.mj_right_gripper_joint[0] = 0.0
            self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)
            time.sleep(0.25)

            # 往高抬一
            self.mj_slide_joint[0] -= 0.035
            self.set_robot_spine()

            # 往后申
            if 1: # Descartes space control
                self.lft_arm_target_pose[2] += 0.025 + 0.01
                self.rgt_arm_target_pose[2] += 0.025
                self.lft_arm_target_pose[0] -= self.forward_distance
                self.rgt_arm_target_pose[0] -= self.forward_distance
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
                self.descartes_control()

            else: # joint control
                #################################################################################################################################################
                self.lft_arm_target_pose[2] += 0.025
                self.rgt_arm_target_pose[2] += 0.025
                for _ in range(3):
                    print("-----------------------------")
                    all_joints = self.real_robot.get_all_joint_states()
                    head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints
                    tm_l = self.arm_fik.properFK(left_arm)
                    print("enlp:\n", tm_l[:3,3], Rotation.from_matrix(tm_l[:3,:3]).as_quat())
                    tm_r = self.arm_fik.properFK(right_arm)
                    print("enrp:\n", tm_r[:3,3], Rotation.from_matrix(tm_r[:3,:3]).as_quat())
                    ap = np.array(self.real_robot.get_arms_pose("a"))
                    print("arm end pose:\n\t", ap[:7], "\n\t", ap[7:])
                    tm_get_l = np.eye(4)
                    tm_get_l[:3,3] = ap[:7][:3]
                    tm_get_l[:3,:3] = Rotation.from_quat(ap[:7][3:]).as_matrix()

                    tm_get_r = np.eye(4)
                    tm_get_r[:3,3] = ap[7:][:3]
                    tm_get_r[:3,:3] = Rotation.from_quat(ap[7:][3:]).as_matrix()

                    print("*******************************")
                    """
                    tm_l     : mujoco base -> endpoint
                    tm_get_l : sdk base    -> endpoint
                    term mat : sdk base    -> mujoco base
                    target   : sdk base    -> mujoco base -> endpoint
                    """
                    print((np.linalg.inv(tm_get_l) @ tm_l))
                    print((np.linalg.inv(tm_get_r) @ tm_r))
                    print((np.linalg.inv(tm_get_l) @ tm_l).tolist())
                    print((np.linalg.inv(tm_get_r) @ tm_r).tolist())
                    print("*******************************")

                    self.lft_arm_target_pose[0] -= self.forward_distance / 3.
                    self.rgt_arm_target_pose[0] -= self.forward_distance / 3.
                    self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
                    self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
                    self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
                    self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
                    self.real_robot.execute_trajectory(True)
                    self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())
                print("-----------------------------")
                all_joints = self.real_robot.get_all_joint_states()
                head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints
                tm_l = self.arm_fik.properFK(left_arm)
                print("enlp:\n", tm_l[:3,3], Rotation.from_matrix(tm_l[:3,:3]).as_quat())
                tm_r = self.arm_fik.properFK(right_arm)
                print("enrp:\n", tm_r[:3,3], Rotation.from_matrix(tm_r[:3,:3]).as_quat())
                ap = np.array(self.real_robot.get_arms_pose("a"))
                print("arm end pose:\n\t", ap[:7], "\n\t", ap[7:])
                #################################################################################################################################################

            self.slamtec_simple_move(-0.15, 0)

            self.mj_slide_joint[0] = 0.89 - self.table_height
            self.set_robot_spine()

            # 下降一点
            if self.at_desk:
                self.lft_arm_target_pose[2] -= 0.25
                self.rgt_arm_target_pose[2] -= 0.25
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
                self.descartes_control()

            self.pick_box_pose = None
            self.holding_box = True
            self.last_action = "carry_box"
        else:
            print("No box detected")
    
    def action_put_box(self, plant_height:float=0.74):
        if self.holding_box:
            print("<MMK2> action_put_box_to_{:.2f}m".format(plant_height))

            self.mj_slide_joint[0] = 0.89 - plant_height - 0.05
            self.set_robot_spine()

            # 臂先抬高一点
            self.lft_arm_target_pose[2] = plant_height + 0.19 + 0.01 # :FINETUNE:
            self.rgt_arm_target_pose[2] = plant_height + 0.19 + 0.0  # :FINETUNE:
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
            self.descartes_control()

            if 1:
                self.lft_arm_target_pose[0] += self.forward_distance
                self.rgt_arm_target_pose[0] += self.forward_distance

                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
                self.descartes_control()

            else: # joint space control
                for _ in range(3):
                    self.lft_arm_target_pose[0] += self.forward_distance / 3.
                    self.rgt_arm_target_pose[0] += self.forward_distance / 3.
                    self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
                    self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
                    self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
                    self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
                    self.real_robot.execute_trajectory(True)
                    self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())

            self.mj_left_gripper_joint[0] = 0.04
            self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
            self.mj_right_gripper_joint[0] = 0.04
            self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
            self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)

            self.mj_slide_joint[0] = 0.89 - plant_height - 0.04
            self.set_robot_spine()

            self.lft_arm_target_pose[1] += 0.035
            self.rgt_arm_target_pose[1] -= 0.035
            self.lft_arm_target_pose[2] = plant_height + 0.19 + 0.01 # :FINETUNE:
            self.rgt_arm_target_pose[2] = plant_height + 0.19 + 0.0  # :FINETUNE:
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
            self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
            self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
            self.real_robot.execute_trajectory(True)
            self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())

            self.lft_arm_target_pose[0] -= 0.20
            self.rgt_arm_target_pose[0] -= 0.20
            self.lft_arm_target_pose[1] += 0.015
            self.rgt_arm_target_pose[1] -= 0.015
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', [ np.pi/2., 0,  0.35]).as_matrix())
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', [-np.pi/2., 0, -0.35]).as_matrix())
            self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
            self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
            self.real_robot.execute_trajectory(True)
            self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())
            
            # 在货架时先回到pick_look位姿
            if not self.at_desk:
                self.lft_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["left_arm"]["look_new"]
                self.rgt_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["right_arm"]["look_new"]

                self.lft_arm_target_pose[2] -= (self.mj_slide_joint[0] - self.PickupExhibitionStateMachineKeyPose["lift"]["look_new"])
                self.rgt_arm_target_pose[2] -= (self.mj_slide_joint[0] - self.PickupExhibitionStateMachineKeyPose["lift"]["look_new"])

                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["pick_new"]).as_matrix())
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["pick_new"]).as_matrix())
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
                self.real_robot.execute_trajectory(True)
                self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())
                self.stay_normal_state["l"] = False
                self.stay_normal_state["r"] = False

            self.holding_box = False
            self.last_action = "put_box"
        else:
            print("No box holding")

    def action_pickup(self):

        if self.pick_prize_select == 0:
            gripper_set = 0.04
            up_bias = 0.03
        else:
            gripper_set = 0.025
            up_bias = 0.0

        self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["pick_high"] - up_bias
        self.set_robot_spine()

        if not self.pick_prize_select_front_pose is None:
            pick_high_posi = self.pick_prize_select_front_pose.copy()
            if self.pick_prize_select == 0:
                pick_high_posi[0] += 0.025
            pick_high_posi[2] = self.PickupExhibitionStateMachineKeyPose["arm_height"]["pick_high"] + up_bias

            if pick_high_posi[1] > 0.0: # left
                self.lft_arm_target_pose[:] = pick_high_posi
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint)
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist())
                
                self.mj_left_gripper_joint[0] = gripper_set
                self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")

            else: # right
                self.rgt_arm_target_pose[:] = pick_high_posi
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint)
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist())

                self.mj_right_gripper_joint[0] = gripper_set
                self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)

            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["pick_down"] - up_bias
            self.set_robot_spine()

            # 关闭夹爪
            if pick_high_posi[1] > 0.0: # left
                self.mj_left_gripper_joint[0] = 0.0
                self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            else:
                self.mj_right_gripper_joint[0] = 0.0
                self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)
            time.sleep(0.25)

            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["pick_up"]
            self.set_robot_spine()

            if pick_high_posi[1] > 0.0: # left
                self.lft_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["left_arm"]["pick_up"]
                lft_end_euler = self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["handover"]
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint, Rotation.from_euler('zyx', lft_end_euler).as_matrix())
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist())
            else:
                self.rgt_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["right_arm"]["pick_up"]
                rgt_end_euler = self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["handover"]
                self.mj_right_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', rgt_end_euler).as_matrix())
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist())

            self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["handover"]
            self.set_robot_head()

            self.wait_for_handover(pick_high_posi)

            self.pick_prize_select_front_pose = None

    def wrist_roll(self, lr="a"):
        if lr in ["l", "a"]:
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["pick_new"]).as_matrix())
            self.mj_left_arm_joint[:]  =  [-0.139, -0.359, 0.343,  2.382, -1.02, -0.629]
        if lr in ["r", "a"]:
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["pick_new"]).as_matrix())
            self.mj_right_arm_joint[:] =  [ 0.139, -0.359, 0.343, -2.382,  1.02,  0.629]
        self.send_cmd(spine_first=False)
        self.sysc_real_robot()

    def action_new_look(self):
        # 再控制升降
        if self.arm_action == "pick" and self.last_action != "pick_look":
            self.action_normal(only_arm=True)
            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["look_new"]
            self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["look_new"]
            self.last_action = "pick_look"
            self.send_cmd(spine_first=False)
            self.sysc_real_robot()
        elif self.arm_action == "carry" and self.last_action != "carry_look":
            spain_delta = 0.74 - self.action_carry_height

            self.lft_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["left_arm"]["look_new"]
            self.rgt_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["right_arm"]["look_new"]

            self.lft_arm_target_pose[2] -= (self.mj_slide_joint[0] - self.PickupExhibitionStateMachineKeyPose["lift"]["look_new"])
            self.rgt_arm_target_pose[2] -= (self.mj_slide_joint[0] - self.PickupExhibitionStateMachineKeyPose["lift"]["look_new"])

            if spain_delta > 0:
                print("Carry height too low, move arm to new look pose first")
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, "pick", "l", self.mj_slide_joint[0], self.mj_left_arm_joint , Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["pick_new"]).as_matrix())
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, "pick", "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["pick_new"]).as_matrix())
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
                self.real_robot.execute_trajectory(True)
                self.check_both_arms(self.mj_left_arm_joint.tolist(), self.mj_right_arm_joint.tolist())
                self.stay_normal_state["l"] = False
                self.stay_normal_state["r"] = False

            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["look_new"] + spain_delta
            self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["carry_new"]
            self.lft_arm_target_pose[2] -= spain_delta
            self.rgt_arm_target_pose[2] -= spain_delta
            self.last_action = "carry_look"
            self.send_cmd(spine_first=False)
            self.sysc_real_robot()
        elif self.arm_action in ["pick", "carry"]:
            print(f"Already in {self.last_action} state")

    def action_new_pickup(self, wait_time_s=20):
        if not self.pick_prize_select_front_pose is None:
            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["pick_new"]
            self.set_robot_spine()

            if self.pick_prize_select == 0:
                gripper_set = 0.04
                up_bias = 0.03
            else:
                gripper_set = 0.025
                up_bias = 0.0

            pick_high_posi = self.pick_prize_select_front_pose.copy()
            if self.pick_prize_select == 0:
                pick_high_posi[0] += 0.025
            pick_high_posi[2] = self.PickupExhibitionStateMachineKeyPose["arm_height"]["pick_new"] + up_bias + 0.05

            if pick_high_posi[1] > 0.0: # left
                # set to normal_p pose and flip wrist
                self.set_action_to_normal_p("l")
                self.wrist_roll("l")
                # set to pick_new pose
                self.lft_arm_target_pose[:] = pick_high_posi
                self.lft_arm_target_pose[1] += 0.08
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["pick_new"]).as_matrix())
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist())
                self.mj_left_gripper_joint[0] = gripper_set
                self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            else: # right
                # set to normal_p pose and flip wrist
                self.set_action_to_normal_p("r")
                self.wrist_roll("r")
                self.rgt_arm_target_pose[:] = pick_high_posi
                self.rgt_arm_target_pose[1] -= 0.08
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["pick_new"]).as_matrix())
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist())

                self.mj_right_gripper_joint[0] = gripper_set
                self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)

            pick_high_posi[2] -= 0.05
            if pick_high_posi[1] > 0.0: # left
                self.lft_arm_target_pose[:] = pick_high_posi
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["pick_new"]).as_matrix())
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist())
            else: # right
                self.rgt_arm_target_pose[:] = pick_high_posi
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["pick_new"]).as_matrix())
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist())

            # 关闭夹爪
            if pick_high_posi[1] > 0.0: # left
                self.mj_left_gripper_joint[0] = 0.0
                self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
            else:
                self.mj_right_gripper_joint[0] = 0.0
                self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
                self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
            self.real_robot.execute_trajectory(True)
            time.sleep(0.25)

            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["pick_up"]
            self.set_robot_spine()

            self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["handover"]
            self.set_robot_head()

            if pick_high_posi[0] < 0.6:
                if pick_high_posi[1] > 0.0: # left
                    self.lft_arm_target_pose[0] = 0.6
                    lft_end_euler = self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["handover"]
                    self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.PickupExhibitionStateMachineKeyPose["lift"]["pick_new"], self.mj_left_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["pick_new"]).as_matrix())
                    self.set_robot_arm("l", self.mj_left_arm_joint.tolist())
                else:
                    self.rgt_arm_target_pose[0] = 0.6
                    rgt_end_euler = self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["handover"]
                    self.mj_right_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.PickupExhibitionStateMachineKeyPose["lift"]["pick_new"], self.mj_right_arm_joint, Rotation.from_euler('zyx', self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["pick_new"]).as_matrix())
                    self.set_robot_arm("r", self.mj_right_arm_joint.tolist())

            if pick_high_posi[1] > 0.0: # left
                self.lft_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["left_arm"]["pick_up"]
                lft_end_euler = self.PickupExhibitionStateMachineKeyPose["arm_euler"]["left"]["handover"]
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint, Rotation.from_euler('zyx', lft_end_euler).as_matrix())
                self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist(), False)
            else:
                self.rgt_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["right_arm"]["pick_up"]
                rgt_end_euler = self.PickupExhibitionStateMachineKeyPose["arm_euler"]["right"]["handover"]
                self.mj_right_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', rgt_end_euler).as_matrix())
                self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist(), False)
            self.real_robot.execute_trajectory(True)

            ret = self.wait_for_handover(pick_high_posi, wait_time_s)
            self.pick_prize_select_front_pose = None
            self.last_action = "pick_up"
            return ret
        else:
            print("No prize selected")
            return None

    def wait_for_handover(self, pick_high_posi, wait_time_s=20):
        ret = True
        for i in range(int(10 * wait_time_s)):
            if self.hand_detector.detect(self.real_robot.get_image("head", "color"), False):
                print("Hand detected")
                break
            time.sleep(0.1)
        else:
            print("No hand detected")
            ret = False
            if pick_high_posi[1] > 0.0: # left
                self.lft_arm_target_pose[:] = pick_high_posi
                self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint)
                # self.mj_left_arm_joint[:] = [0.305, -1.454,  1.371,  2.642,  1.075,  0.626]
                self.set_robot_arm("l", self.mj_left_arm_joint.tolist())
            else: # right
                self.rgt_arm_target_pose[:] = pick_high_posi
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint)
                # self.mj_right_arm_joint[:] = [-0.133, -0.369,  0.347,  2.378, -1.012, -0.628]
                self.set_robot_arm("r", self.mj_right_arm_joint.tolist())

        if pick_high_posi[1] > 0.0: # left
            self.mj_left_gripper_joint[0] = 0.04
            self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
            self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
        else:
            self.mj_right_gripper_joint[0] = 0.04
            self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
            self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
        self.real_robot.execute_trajectory(True)

        self.sysc_real_robot()
        return ret

    def set_action_to_normal_p(self, lr="a"):
        exp = self.PickupExhibitionStateMachineKeyPose
        states = self.get_current_states()
        spine_state = states[1][0]
        if lr in ["l", "a"]:
            self.lft_arm_target_pose[:] = exp["left_arm"]["normal_p"]
            self.lft_arm_target_pose[2] -= spine_state - exp["lift"]["normal"]
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", spine_state, self.mj_left_arm_joint)
            self.stay_normal_state["l"] = False
        if lr in ["r", "a"]:
            self.rgt_arm_target_pose[:] = exp["right_arm"]["normal_p"]
            self.rgt_arm_target_pose[2] -= spine_state - exp["lift"]["normal"]
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", spine_state, self.mj_right_arm_joint)
            self.stay_normal_state["r"] = False
        self.send_cmd(spine_first=True)
        self.sysc_real_robot()

    def set_action_to_normal(self, lr="a", only_arm=False):
        if not only_arm:
            self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["normal"]
            self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["normal"]
        if lr in ["l", "a"]:
            self.lft_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["left_arm"]["normal"]
            self.mj_left_arm_joint[:]  = MMK2FIK().get_armjoint_pose_wrt_footprint(self.lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint)
            self.stay_normal_state["l"] = True
        if lr in ["r", "a"]:
            self.rgt_arm_target_pose[:] = self.PickupExhibitionStateMachineKeyPose["right_arm"]["normal"]
            self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(self.rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint)
            self.stay_normal_state["r"] = True
        self.send_cmd(spine_first=True)
        self.sysc_real_robot()

    def action_normal(self, only_arm=False, nop=False):
        sns = self.stay_normal_state
        if not nop:
            if not sns["l"] and not sns["r"]:
                self.set_action_to_normal_p("a")
            elif not sns["l"]:
                self.set_action_to_normal_p("l")
            elif not sns["r"]:
                self.set_action_to_normal_p("r")
        self.set_action_to_normal("a", only_arm)

    def base_servo_putbox(self):
        posi = self.select_blue_circle()
        if not posi is None:
            valid_posi = True
            target_front_dist = 0.88
            disarr = posi[:2] - np.array([target_front_dist, 0.0])
            if np.hypot(disarr[0], disarr[1]) > 0.08:
                valid_posi = False
                if posi[0] - target_front_dist < 0.0:
                    self.slamtec_simple_move(posi[0] - target_front_dist, 0.0)
                if abs(posi[1]) > 0.05:
                    self.slamtec_lateral_translation(posi[1])
                if posi[0] - target_front_dist > 0.0:
                    self.slamtec_simple_move(posi[0] - target_front_dist, 0.0)
            self.sysc_real_robot()
            return valid_posi
        else:
            return False

    def base_servo_pick(self):
        valid_pick_pose = True
        if abs(self.pick_prize_select_front_pose[1]) > 0.1:
            self.slamtec_lateral_translation(self.pick_prize_select_front_pose[1])
            valid_pick_pose = False
            self.pick_prize_select_front_pose = None
        if self.pick_prize_select_front_pose[0] < 0.4: # :TODO: 注意安全， 设置全局的位置坐标限制
            self.slamtec_simple_move(self.pick_prize_select_front_pose[0] - 0.45, 0.0)
            valid_pick_pose = False
            self.pick_prize_select_front_pose = None
        elif self.pick_prize_select_front_pose[0] > 0.55: # :TODO: 注意安全， 设置全局的位置坐标限制
            self.slamtec_simple_move(self.pick_prize_select_front_pose[0] - 0.55, 0.0)
            valid_pick_pose = False
        if not valid_pick_pose:
            self.pick_prize_select_front_pose = None
        self.sysc_real_robot()

    def base_servo_carry(self):
        # :TODO: 增加高度判断
        target_front_dist = 0.7 #0.665
        disarr = self.pick_box_pose[:2] - np.array([target_front_dist, 0.0])
        if np.hypot(disarr[0], disarr[1]) > 0.2:
            self.pick_box_pose = None
            print("<WARNING>: pick_box_pose too far!")
        elif np.hypot(disarr[0], disarr[1]) > 0.025:
            if self.pick_box_pose[0] - target_front_dist < 0.0:
                self.slamtec_simple_move(self.pick_box_pose[0] - target_front_dist, 0.0)
            if abs(self.pick_box_pose[1]) > 0.05:
                self.slamtec_lateral_translation(self.pick_box_pose[1])
            if self.pick_box_pose[0] - target_front_dist > 0.0:
                self.slamtec_simple_move(self.pick_box_pose[0] - target_front_dist, 0.0)
            self.pick_box_pose = None
        self.sysc_real_robot()

    def moveto_shelf_servo(self):
        self.mj_head_joint[:] = self.PickupExhibitionStateMachineKeyPose["head"]["servo_putbox"]
        self.set_robot_head()
        self.mj_slide_joint[0] = self.PickupExhibitionStateMachineKeyPose["lift"]["servo_putbox"]
        self.set_robot_spine()
        time.sleep(1.)
        self.sysc_real_robot()

        for _ in range(5):
            if self.base_servo_putbox():
                break
            time.sleep(0.5)
        else:
            raise Exception("<MMK2_EXHIBITION>: Can not move to putbox position!")
        self.at_desk = False
        return True

    def moveto_prize(self, prize_id):
        # prize_id: 0, 1, 2
        pose = self.get_base_pose()
        dist_s = []
        for key in self.box_desk_position_map.keys():
            dist_s.append(np.hypot(pose[0] - self.box_desk_position_map[key][0], pose[1] - self.box_desk_position_map[key][1]))
        min_id = np.argmin(dist_s)
        min_dist = dist_s[min_id]

        if min_dist < 0.1:
            rot_ang = self.tmp_angle_diff(self.target_yaw, pose[2])
            if abs(rot_ang) > 0.05:
                self.slamtec_simple_move(0.0, rot_ang)
                pose = self.get_base_pose()
            if min_id != prize_id:
                tx_wrt_map = self.box_desk_position_map[self.map_prize_id_2_name[prize_id]][0] - pose[0]
                ty_wrt_map = self.box_desk_position_map[self.map_prize_id_2_name[prize_id]][1] - pose[1]
                tx_wrt_base = tx_wrt_map * np.cos(pose[2]) + ty_wrt_map * np.sin(pose[2])
                ty_wrt_base = -tx_wrt_map * np.sin(pose[2]) + ty_wrt_map * np.cos(pose[2])
                self.slamtec_lateral_translation(ty_wrt_base)
                self.slamtec_simple_move(tx_wrt_base, 0.0)
                pose = self.get_base_pose()
                rot_ang = self.tmp_angle_diff(self.target_yaw, pose[2])
                if abs(rot_ang) > 0.05:
                    self.slamtec_simple_move(0.0, rot_ang)
                    pose = self.get_base_pose()
        else:
            target_pose = self.box_desk_position_map[self.map_prize_id_2_name[prize_id]]
            self.moveto_fromhome(target_pose)
        self.last_prize_id = prize_id
        self.at_desk = True

    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        if key == ord("n"):
            self.action_normal()
        elif key == ord("l"):
            self.action_new_look()
            # self.action_look()
        elif key == ord(";"):
            if self.arm_action == 'pick':
                self.action_new_pickup()
            elif self.arm_action == 'carry':
                self.action_carry(self.action_carry_height)
            else:
                raise ValueError(f"arm_action: {self.arm_action} not in ['pick', 'carry']")
        elif key == ord("e"):
            # self.select_blue_circle()
            self.moveto_shelf_servo()

        elif key == ord("c"):
            self.action_carry_joy = not self.action_carry_joy
            if self.action_carry_joy:
                self.action_carry_height = self.set_action_carry_height
                self.arm_action = "carry"
                self.sysc_real_robot()
            else:
                self.arm_action = "pick"
                self.sysc_real_robot()
            print(f"carry joy: {self.action_carry_joy}")

        elif key == ord("'"):
            if self.arm_action == 'pick' and not self.pick_prize_select_front_pose is None:
                if self.pick_prize_select_front_pose[1] > 0.0: # left
                    self.mj_left_gripper_joint[0] = 0.04
                    self.mj_left_gripper_joint[1] = -self.mj_left_gripper_joint[0]
                    self.real_robot.set_gripper(25.*self.mj_left_gripper_joint[0] , "l")
                else:
                    self.mj_right_gripper_joint[0] = 0.04
                    self.mj_right_gripper_joint[1] = -self.mj_right_gripper_joint[0]
                    self.real_robot.set_gripper(25.*self.mj_right_gripper_joint[0], "r")
                self.real_robot.execute_trajectory(True)
                self.pick_prize_select_front_pose = None

            elif self.arm_action == 'carry':
                self.action_put_box(self.action_carry_height)

        elif key == ord("0"):
            self.moveto_backhome()

        elif ord("5") <= key <= ord("6"):
            target_pose = self.box_shelf_position_map[key - ord("5") + 1]
            self.moveto_fromhome(target_pose)

        elif ord("7") <= key <= ord("9"):
            prize_id = key - ord("7")
            self.moveto_prize(prize_id)

        elif ord("1") <= key <= ord("3"):
            if self.arm_action == "pick":
                self.pick_prize_select_front_pose = self.select_prize(key - ord("1"))
                if not self.pick_prize_select_front_pose is None:
                    self.base_servo_pick()

            elif self.arm_action == "carry":
                # self.pick_box_pose = self.select_box(key - ord("1"))
                self.pick_box_pose = self.select_box_nearest(self.action_carry_height)
                if not self.pick_box_pose is None:
                    self.base_servo_carry()

        elif key == ord("i"):
            try:
                self.action_carry_height = float(input(">>> Input carry height(m):"))
            except:
                self.action_carry_height = self.set_action_carry_height
            print(f"carry height: {self.action_carry_height}")

        return ret

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    rospy.init_node('mujoco_exhibition_pickup_statemachine_node', anonymous=True)

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

    exec_node = MMK2Exhibition(cfg)

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

