import os
import rospy
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim.envs.mmk2_base import MMK2Base, MMK2Cfg
from dlabsim.airbot_play import AirbotPlayFIK
from dlabsim.mmk2 import MMK2FIK
from dlabsim import DLABSIM_ASSERT_DIR

class MMK2JOY(MMK2Base):
    arm_action_init_position = {
        "look" : {
            "l" : np.array([0.42263,  0.19412,  1.26702]),
            "r" : np.array([0.42263, -0.19412,  1.26702]),
        },
        "pick" : {
            "l" : np.array([0.223,  0.21, 1.07055]),
            "r" : np.array([0.223, -0.21, 1.07055]),
        },
        "carry" : {
            "l" : np.array([0.223,  0.21, 1.07055]),
            "r" : np.array([0.223, -0.21, 1.07055]),
        }
    }

    def __init__(self, config: MMK2Cfg):
        self.arm_action = config.init_key

        super().__init__(config)
        self.mj_base = self.mj_data.qpos[:7]
        self.mj_wheel_joint = self.mj_data.qpos[7:9]
        self.mj_slide_joint = self.mj_data.qpos[9:10]
        self.mj_head_joint = self.mj_data.qpos[10:12]
        self.mj_left_arm_joint = self.mj_data.qpos[12:18]
        self.mj_left_gripper_joint = self.mj_data.qpos[18:20]
        self.mj_right_arm_joint = self.mj_data.qpos[20:26]
        self.mj_right_gripper_joint = self.mj_data.qpos[26:28]

        self.lft_arm_target_pose = self.arm_action_init_position[self.arm_action]["l"].copy()
        self.lft_end_euler = np.zeros(3)
        self.rgt_arm_target_pose = self.arm_action_init_position[self.arm_action]["r"].copy()
        self.rgt_end_euler = np.zeros(3)

        self.arm_fik = AirbotPlayFIK(urdf = os.path.join(DLABSIM_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf"))

    def resetState(self):
        super().resetState()
        self.lft_arm_target_pose = self.arm_action_init_position[self.arm_action]["l"].copy()
        self.lft_end_euler = np.zeros(3)
        self.rgt_arm_target_pose = self.arm_action_init_position[self.arm_action]["r"].copy()
        self.rgt_end_euler = np.zeros(3)

    def teleopProcess_view(self):
        if self.teleop.joy_cmd.buttons[4]:   # left arm
            tmp_lft_arm_target_pose = self.lft_arm_target_pose.copy()
            tmp_lft_arm_target_pose[0] += self.teleop.joy_cmd.axes[7] * 0.1 / self.render_fps
            tmp_lft_arm_target_pose[1] += self.teleop.joy_cmd.axes[6] * 0.1 / self.render_fps
            tmp_lft_arm_target_pose[2] += self.teleop.joy_cmd.axes[1] * 0.1 / self.render_fps

            delta_gripper = 0.04 * (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]) * 1. / self.render_fps
            self.mj_left_gripper_joint[1] += delta_gripper
            self.mj_left_gripper_joint[1] = np.clip(self.mj_left_gripper_joint[1], self.mj_model.joint("lft_finger_left_joint").range[0], self.mj_model.joint("lft_finger_left_joint").range[1])
            self.mj_left_gripper_joint[0] = -self.mj_left_gripper_joint[1]
            el = self.lft_end_euler.copy()
            el[0] += self.teleop.joy_cmd.axes[4] * 0.35 / self.render_fps
            el[1] += self.teleop.joy_cmd.axes[3] * 0.35 / self.render_fps
            el[2] += self.teleop.joy_cmd.axes[0] * 0.35 / self.render_fps
            try:
                self.mj_left_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(tmp_lft_arm_target_pose, self.arm_action, "l", self.mj_slide_joint[0], self.mj_left_arm_joint, Rotation.from_euler('zyx', el).as_matrix())
                self.lft_arm_target_pose[:] = tmp_lft_arm_target_pose
                self.lft_end_euler[:] = el
            except ValueError:
                print("Invalid left arm target position:", tmp_lft_arm_target_pose)

        elif self.teleop.joy_cmd.buttons[5]: # right arm
            tmp_rgt_arm_target_pose = self.rgt_arm_target_pose.copy()
            tmp_rgt_arm_target_pose[0] += self.teleop.joy_cmd.axes[7] * 0.1 / self.render_fps
            tmp_rgt_arm_target_pose[1] += self.teleop.joy_cmd.axes[6] * 0.1 / self.render_fps
            tmp_rgt_arm_target_pose[2] += self.teleop.joy_cmd.axes[1] * 0.1 / self.render_fps

            delta_gripper = 0.04 * (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]) * 1. / self.render_fps
            self.mj_right_gripper_joint[1] += delta_gripper
            self.mj_right_gripper_joint[1] = np.clip(self.mj_right_gripper_joint[1], self.mj_model.joint("rgt_finger_left_joint").range[0], self.mj_model.joint("rgt_finger_left_joint").range[1])
            self.mj_right_gripper_joint[0] = -self.mj_right_gripper_joint[1]
            el = self.rgt_end_euler.copy()
            el[0] += self.teleop.joy_cmd.axes[4] * 0.35 / self.render_fps
            el[1] += self.teleop.joy_cmd.axes[3] * 0.35 / self.render_fps
            el[2] += self.teleop.joy_cmd.axes[0] * 0.35 / self.render_fps
            try:
                self.mj_right_arm_joint[:] = MMK2FIK().get_armjoint_pose_wrt_footprint(tmp_rgt_arm_target_pose, self.arm_action, "r", self.mj_slide_joint[0], self.mj_right_arm_joint, Rotation.from_euler('zyx', el).as_matrix())
                self.rgt_arm_target_pose[:] = tmp_rgt_arm_target_pose
                self.rgt_end_euler[:] = el
            except ValueError:
                print("Invalid right arm target position:", tmp_rgt_arm_target_pose)

        else:
            delta_height = (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]) * 0.2 / self.render_fps
            if self.mj_slide_joint[0] + delta_height< self.mj_model.joint("slide_joint").range[0]:
                delta_height = self.mj_model.joint("slide_joint").range[0] - self.mj_slide_joint[0]
            elif self.mj_slide_joint[0] + delta_height > self.mj_model.joint("slide_joint").range[1]:
                delta_height = self.mj_model.joint("slide_joint").range[1] - self.mj_slide_joint[0]
            self.mj_slide_joint[0] += delta_height
            self.lft_arm_target_pose[2] -= delta_height
            self.rgt_arm_target_pose[2] -= delta_height

            self.mj_head_joint[0] += self.teleop.joy_cmd.axes[3] * 1. / self.render_fps
            self.mj_head_joint[1] += self.teleop.joy_cmd.axes[4] * 1. / self.render_fps
            self.mj_head_joint[0] = np.clip(self.mj_head_joint[0], self.mj_model.joint("head_yaw_joint").range[0], self.mj_model.joint("head_yaw_joint").range[1])
            self.mj_head_joint[1] = np.clip(self.mj_head_joint[1], self.mj_model.joint("head_pitch_joint").range[0], self.mj_model.joint("head_pitch_joint").range[1])

            linear_vel  = 1.0 * self.teleop.joy_cmd.axes[1]**2 * np.sign(self.teleop.joy_cmd.axes[1])
            angular_vel = 2.0 * self.teleop.joy_cmd.axes[0]**2 * np.sign(self.teleop.joy_cmd.axes[0])
            self.base_move(linear_vel, angular_vel)

    def base_move(self, linear_vel, angular_vel):
        yaw = Rotation.from_quat(self.mj_base[[4,5,6,3]]).as_euler('zyx')[0]
        self.mj_base[0] += np.cos(yaw) * linear_vel / self.render_fps
        self.mj_base[1] += np.sin(yaw) * linear_vel / self.render_fps
        yaw += angular_vel / self.render_fps
        self.mj_base[[4,5,6,3]] = Rotation.from_euler('zyx', [yaw, 0.0, 0.0]).as_quat()

    def view(self):
        if self.teleop:
            self.teleopProcess_view()
        super().view()

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
        print("    lft_endp  = {}".format(self.mj_data.site("lft_endpoint").xpos))
        print("    rgt_endp  = {}".format(self.mj_data.site("rgt_endpoint").xpos))
        print("-" * 100)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()

    cfg.init_key = "pick"
    cfg.obs_camera_id = None
    cfg.render_set["fps"] = 60
    cfg.mjcf_file_path = "mjcf/exhibition_conference.xml"
    cfg.gs_model_dict.insert(0, "iros_booth/booth.ply")

    rospy.init_node('mujoco_node', anonymous=True)
    exec_node = MMK2JOY(cfg)
    exec_node.reset()

    # exec_node.options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # exec_node.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # exec_node.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # exec_node.options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # exec_node.options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # exec_node.options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True

    # exec_node.options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
    # exec_node.options.frame = mujoco.mjtFrame.mjFRAME_SITE.value

    while exec_node.running:
        # obs, pri_obs, rew, ter, info = exec_node.step(action_list)
        exec_node.view()
