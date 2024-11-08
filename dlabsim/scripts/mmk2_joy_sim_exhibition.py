import cv2
import time
import rospy
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim.mmk2 import MMK2FIK
from dlabsim.envs.mmk2_base import MMK2Cfg
from dlabsim.scripts.mmk2_joy import MMK2JOY

class MMK2JOY_SIM_EXHIBITION(MMK2JOY):
    # def render(self):
    #     self.render_cnt += 1
    #     if not self.renderer._depth_rendering:
    #         self.img_rgb_obs = self.getRgbImg(self.config.obs_camera_id)

    #         if self.cam_id == self.config.obs_camera_id:
    #             img_vis = cv2.cvtColor(self.img_rgb_obs, cv2.COLOR_RGB2BGR)
    #         else:
    #             img_rgb = self.getRgbImg(self.cam_id)
    #             img_vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    #         resize_img = cv2.resize(self.img_rgb_obs, (self.config.render_set["width"]//3, self.config.render_set["height"]//3))
    #         img_vis[0:self.config.render_set["height"]//3, self.config.render_set["width"]*2//3+1:self.config.render_set["width"], :] = resize_img
    #     else:
    #         img_depth = self.getDepthImg(self.cam_id)
    #         if not img_depth is None:
    #             img_vis = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=25.5), cv2.COLORMAP_JET)

    #     if not self.config.headless:
    #         cv2.imshow(self.windowname, img_vis)
    #         if self.config.sync:
    #             wait_time_ms = max(1, int((1./self.render_fps - time.time() + self.last_render_time) * 1000)-1)
    #         else:
    #             wait_time_ms = 1
    #         if not self.cv2WindowKeyPressCallback(cv2.waitKey(wait_time_ms)) or not cv2.getWindowProperty(self.windowname, cv2.WND_PROP_VISIBLE):
    #             self.running = False

    #         if self.config.sync:
    #             self.last_render_time = time.time()

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

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = MMK2Cfg()

    cfg.init_key = "pick"
    cfg.obs_camera_id = None
    cfg.render_set["fps"] = 30
    cfg.render_set["width"]  = 1920
    cfg.render_set["height"] = 1080
    cfg.use_gaussian_renderer = True
    cfg.mjcf_file_path = "mjcf/exhibition_iros.xml"
    cfg.gs_model_dict.insert(0, "iros_booth/booth.ply")

    rospy.init_node('mujoco_node', anonymous=True)

    exec_node = MMK2JOY_SIM_EXHIBITION(cfg)
    exec_node.reset()
    exec_node.cam_id = 0

    point_id = -1
    arrived = True
    points = [
        [ 0.5, -2.5,  np.pi    ],
        [-2.5, -2.5,  np.pi    ],
        [-2.5, -2.5,  np.pi*0.5],
        [-2.5, -2.5,  np.pi*0.5],
        [-2.5,  0.5,  np.pi*0.5],
        [-2.5,  0.5, -np.pi*0.5],
        [-2.5, -2.5, -np.pi*0.5],
        [-2.5, -2.5,  0.0      ],
        [ 0.5, -2.5,  0.0      ],
    ]

    linear = 0.0
    anguler = 0.0
    def step(cur, tar, stp):
        if np.abs(cur - tar) < stp:
            return tar
        return cur + np.sign(tar - cur) * stp

    exec_node.mj_data.qpos[:2]  = points[0][:2]
    exec_node.mj_data.qpos[3:7] = Rotation.from_euler('z', points[0][2]).as_quat()[[3, 0, 1, 2]]
    while exec_node.running:
        if arrived:
            point_id = (point_id + 1) % len(points)
            target = points[point_id]
            arrived = False

        yaw = Rotation.from_quat(exec_node.mj_data.qpos[[4,5,6,3]]).as_euler('zyx')[0]
        dang = target[2] - yaw
        if dang > np.pi:
            dang -= np.pi * 2
        elif dang < -np.pi:
            dang += np.pi * 2
        anguler = step(anguler, 1. * dang, 1. / exec_node.render_fps)
 
        darr = np.array([target[0] - exec_node.mj_data.qpos[0], target[1] - exec_node.mj_data.qpos[1]])
        dist = darr[0] * np.cos(yaw) + darr[1] * np.sin(yaw)
        linear = step(linear, 0.5 * dist, 1.5 / exec_node.render_fps)

        if np.abs(dist) < 1e-2 and np.abs(anguler) < 2e-2:
            arrived = True

        exec_node.base_move(linear, anguler)
        exec_node.view()
