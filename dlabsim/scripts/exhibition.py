import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim.envs import SimulatorGSBase
from dlabsim.envs.mmk2_base import MMK2Base, MMK2Cfg

class MMK2Exhibition(MMK2Base):
    key_direction = {
        "linux" : {
            "key_up"    : 82,
            "key_down"  : 84,
            "key_left"  : 81,
            "key_right" : 83,
        }
    }

    cmd_vel_2d = np.zeros(2)
    def cv2WindowKeyPressCallback(self, key):
        if key == self.key_direction["linux"]["key_up"]:
            self.cmd_vel_2d[0] = 2.0
        elif key == self.key_direction["linux"]["key_down"]:
            self.cmd_vel_2d[0] = -2.0
        else:
            self.cmd_vel_2d[0] = 0.0
        self.cmd_vel_2d[0] = np.clip(self.cmd_vel_2d[0], -1.0, 1.0)

        if key == self.key_direction["linux"]["key_left"]:
            self.cmd_vel_2d[1] = 1.0
        elif key == self.key_direction["linux"]["key_right"]:
            self.cmd_vel_2d[1] = -1.0
        else:
            self.cmd_vel_2d[1] = 0.0
        return super().cv2WindowKeyPressCallback(key)

if __name__ == "__main__":
    cfg = MMK2Cfg()

    cfg.obs_camera_id = -2
    cfg.render_set["fps"] = 25
    cfg.mjcf_file_path = "mjcf/exhibition.xml"
    # cfg.mjcf_file_path = "mjcf/exhibition_digua.xml"
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
            # "exhibition/booth_digua.ply",
            "exhibition/booth.ply",

            # "exhibition/box_for_second_prize.ply",
            # "exhibition/box_for_third_prize.ply",
            # "exhibition/second_prize.ply",
 
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

    obs = exec_node.reset()

    # quat = Rotation.from_euler("xyz", [0, 0, -90], degrees=True).as_quat()
    # exec_node.gs_renderer.set_obj_pose("box_for_second_prize", np.array([-0.77, -1.81, 0.7]), quat)
    # exec_node.gs_renderer.set_obj_pose("box_for_third_prize",  np.array([-1.47, -1.81, 0.7]), quat)

    # quat = Rotation.from_euler("xyz", [-90, -90, 0], degrees=True).as_quat()
    # exec_node.gs_renderer.set_obj_pose("second_prize", np.array([-0.77, -1.79, 0.83]), quat)

    action_list = np.zeros(19)
    i = 0

    while exec_node.running:
        action_list[:2] = exec_node.cmd_vel_2d
        obs, pri_obs, rew, ter, info = exec_node.step(action_list)
        i += 1
