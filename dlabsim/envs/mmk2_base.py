import mujoco
import numpy as np

from dlabsim.envs import SimulatorBase
from dlabsim.utils.base_config import BaseConfig

class MMK2Cfg(BaseConfig):
    mjcf_file_path = "mjcf/mmk2_floor.xml"
    timestep       = 0.0025
    decimation     = 4
    sync           = True
    headless       = False
    init_key       = "home"
    render_set     = {
        "fps"    : 60,
        "width"  : 1920,
        "height" : 1080 
    }
    obs_camera_id  = None
    rb_link_list   = [
        "agv_link", "slide_link", "head_yaw_link", "head_pitch_link",
        "lft_arm_base", "lft_arm_link1", "lft_arm_link2", 
        "lft_arm_link3", "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
        "lft_finger_left_link", "lft_finger_right_link", 
        "rgt_arm_base", "rgt_arm_link1", "rgt_arm_link2", 
        "rgt_arm_link3", "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
        "rgt_finger_left_link", "rgt_finger_right_link"
    ]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict  = {
        "agv_link"              :   "mmk2/agv_link.ply",
        "slide_link"            :   "mmk2/slide_link.ply",
        "head_pitch_link"       :   "mmk2/head_pitch_link.ply",
        "head_yaw_link"         :   "mmk2/head_yaw_link.ply",

        "lft_arm_base"          :   "airbot_play/arm_base.ply",
        "lft_arm_link1"         :   "airbot_play/link1.ply",
        "lft_arm_link2"         :   "airbot_play/link2.ply",
        "lft_arm_link3"         :   "airbot_play/link3.ply",
        "lft_arm_link4"         :   "airbot_play/link4.ply",
        "lft_arm_link5"         :   "airbot_play/link5.ply",
        "lft_arm_link6"         :   "airbot_play/link6.ply",
        "lft_finger_left_link"  :   "airbot_play/left.ply",
        "lft_finger_right_link" :   "airbot_play/right.ply",

        "rgt_arm_base"          :   "airbot_play/arm_base.ply",
        "rgt_arm_link1"         :   "airbot_play/link1.ply",
        "rgt_arm_link2"         :   "airbot_play/link2.ply",
        "rgt_arm_link3"         :   "airbot_play/link3.ply",
        "rgt_arm_link4"         :   "airbot_play/link4.ply",
        "rgt_arm_link5"         :   "airbot_play/link5.ply",
        "rgt_arm_link6"         :   "airbot_play/link6.ply",
        "rgt_finger_left_link"  :   "airbot_play/left.ply",
        "rgt_finger_right_link" :   "airbot_play/right.ply"
    }
    """
    njqpos=29
    [0:7]-base; 7-lft_wheel; 8-rgt_wheel; 9-slide; 10-head_yaw"; 11-head_pitch; [12:20]-lft_arm ; [20:28]-rgt_arm

    njctrl=19
    0-forward; 1-turn; 2-lift; 3-yaw; 4-pitch; [5:12]-lft_arm; [12:19]-rgt_arm
    """

class MMK2Base(SimulatorBase):
    def __init__(self, config: MMK2Cfg):
        self.njq = 28
        self.njv = 27
        self.njctrl = 19

        super().__init__(config)
        self.init_joint_pose = self.mj_model.key(self.config.init_key).qpos[:self.njq]
        self.jq = np.zeros(self.njq)
        self.jv = np.zeros(self.njv)

        ip_cp = self.init_joint_pose.copy()
        self.init_ctrl = np.array(
            [0.0, 0.0] + 
            ip_cp[[9,10,11]].tolist() + 
            ip_cp[12:19].tolist() + 
            ip_cp[20:27].tolist()
        )

        self.resetState()

    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.njq)
        self.jv = np.zeros(self.njv)

        self.mj_data.qpos[:self.njq] = self.init_joint_pose[:self.njq].copy()
        self.mj_data.ctrl[:self.njctrl] = self.init_ctrl.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateState(self):
        self.jq = self.mj_data.qpos[:self.njq]
        self.jv = self.mj_data.qvel[:self.njv]

    def updateControl(self, action):
        for i in range(self.njctrl):
            self.mj_data.ctrl[i] = action[i]

    def checkTerminated(self):
        return False

    def post_physics_step(self):
        pass

    def getObservation(self):
        self.obs = {
            "jq"  : self.jq.tolist(),
            "jv"  : self.jv.tolist(),
            "img" : self.img_rgb_obs_s
        }
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

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

        print("mj_data.qvel :")
        print("    base      = {}".format(np.array2string(self.mj_data.qvel[:6], separator=', ')))
        print("    chassis   = {}".format(np.array2string(self.mj_data.qvel[6:8], separator=', ')))
        print("    lift      = {}".format(np.array2string(self.mj_data.qvel[8:9], separator=', ')))
        print("    head      = {}".format(np.array2string(self.mj_data.qvel[9:11], separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.mj_data.qvel[11:18], separator=', ')))
        print("    right arm = {}".format(np.array2string(self.mj_data.qvel[19:26], separator=', ')))

        print("mj_data.ctrl :")
        print("    chassis   = {}".format(np.array2string(self.mj_data.ctrl[0:2], separator=', ')))
        print("    lift      = {}".format(np.array2string(self.mj_data.ctrl[2:3], separator=', ')))
        print("    head      = {}".format(np.array2string(self.mj_data.ctrl[3:5], separator=', ')))
        print("    left  arm = {}".format(np.array2string(self.mj_data.ctrl[5:12], separator=', ')))
        print("    right arm = {}".format(np.array2string(self.mj_data.ctrl[12:19], separator=', ')))

        print("-" * 100)

if __name__ == "__main__":

    cfg = MMK2Cfg()
    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = "iros_booth/booth.ply"

    exec_node = MMK2Base(cfg)

    obs = exec_node.reset()
    action_list = np.zeros(19)
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action_list)
