import numpy as np

import sys
sys.path.append("/home/tatp/ws/mmk2_demo/mmk2_grpc")
from mmk2_client import MMK2

use_viewer = False

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=1000)

    robot = MMK2("mmk2", -1, "192.168.11.200")
    all_joints = robot.get_all_joint_states()
    print("all joint states:", all_joints)

    from dlabsim.envs import SimulatorGSBase, SimulatorBase
    from dlabsim.envs.mmk2_base import MMK2Base, MMK2Cfg

    cfg = MMK2Cfg()
    cfg.obs_camera_id = -2
    cfg.render_set["fps"] = 24
    cfg.mjcf_file_path = "mjcf/mmk2_floor.xml"
    cfg.rb_link_list =  [
        "head_pitch_link", "lift_link", "assembly_board_link", "agv", 
        "lft_arm_base", "lft_arm_link1", "lft_arm_link2", 
        "lft_arm_link3", "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
        "lft_finger_left_link", "lft_finger_right_link", 
        "rgt_arm_base", "rgt_arm_link1", "rgt_arm_link2", 
        "rgt_arm_link3", "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
        "rgt_finger_left_link", "rgt_finger_right_link"
    ]
    cfg.obj_list = []

    view_node = MMK2Base(cfg)

    if isinstance(view_node, SimulatorGSBase):
        models_lst = [
            # "exhibition/booth.ply",
            "exhibition/booth_digua.ply",

            # "exhibition/box_for_second_prize.ply",
            # "exhibition/box_for_third_prize.ply",
            # "exhibition/second_prize.ply",

            "mmk2/mmk2_base/head_pitch_link.ply",
            "mmk2/mmk2_base/lift_link.ply",
            "mmk2/mmk2_base/assembly_board_link.ply",
            "mmk2/mmk2_base/agv.ply",

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
        view_node.init_gs_render(models_lst)

    mj_wheel_joint = view_node.mj_data.qpos[7:9]
    mj_slide_joint = view_node.mj_data.qpos[9:10]
    mj_head_joint = view_node.mj_data.qpos[10:12]
    mj_left_arm_joint = view_node.mj_data.qpos[12:18]
    mj_left_gripper_joint = view_node.mj_data.qpos[18:20]
    mj_right_arm_joint = view_node.mj_data.qpos[20:26]
    mj_right_gripper_joint = view_node.mj_data.qpos[26:28]

    while view_node.running:
        all_joints = robot.get_all_joint_states()
        head, lift, left_arm, right_arm, left_gripper, right_gripper, xy_yaw = all_joints
        mj_slide_joint[:] = lift
        mj_head_joint[0] = head[0]
        mj_head_joint[1] = head[1]
        mj_left_arm_joint[:] = left_arm
        mj_left_gripper_joint[0] = left_gripper[0] * 0.04
        mj_left_gripper_joint[1] = -mj_left_gripper_joint[0]
        mj_right_arm_joint[:] = right_arm
        mj_right_gripper_joint[0] = right_gripper[0] * 0.04
        mj_right_gripper_joint[1] = -mj_right_gripper_joint[0]
        view_node.view()

    print(view_node.mj_data.qpos)