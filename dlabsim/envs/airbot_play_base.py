import mujoco
import numpy as np

from dlabsim.utils.base_config import BaseConfig
from dlabsim.envs import SimulatorBase

class AirbotPlayCfg(BaseConfig):
    mjcf_file_path = "mjcf/il.xml"
    decimation     = 4
    timestep       = 0.005
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720,
    }
    obs_camera_id  = None
    rb_link_list   = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    obj_list       = []
    use_gaussian_renderer = False
    gs_model_dict = {
        "arm_base"  : "airbot_play/arm_base.ply",
        "link1"     : "airbot_play/link1.ply",
        "link2"     : "airbot_play/link2.ply",
        "link3"     : "airbot_play/link3.ply",
        "link4"     : "airbot_play/link4.ply",
        "link5"     : "airbot_play/link5.ply",
        "link6"     : "airbot_play/link6.ply",
        "left"      : "airbot_play/left.ply",
        "right"     : "airbot_play/right.ply",
    }
    init_joint_pose = {
        "joint1"  : 0,
        "joint2"  : 0,
        "joint3"  : 0,
        "joint4"  : 0,
        "joint5"  : 0,
        "joint6"  : 0,
        "gripper" : 0.5
    }

class AirbotPlayBase(SimulatorBase):
    def __init__(self, config: AirbotPlayCfg):
        self.nj = 7
        super().__init__(config)

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)

        self.init_joint_pose = []
        for i in range(self.nj-1):
            self.init_joint_pose.append(self.config.init_joint_pose["joint{}".format(i+1)])
        self.init_joint_pose.append(self.config.init_joint_pose["gripper"] * 0.04)
        self.init_joint_pose = np.array(self.init_joint_pose)

        self.resetState()
        self.updateState()


    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)

        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.ctrl[:self.nj] = self.init_joint_pose.copy()

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def updateState(self):
        self.jq = self.mj_data.qpos[:self.nj]
        self.jv = self.mj_data.qvel[:self.nj]

    def updateControl(self, action):
        if self.mj_data.qpos[self.nj-1] < 0.0:
            self.mj_data.qpos[self.nj-1] = 0.0

        for i in range(self.nj):
            if i == self.nj-1:
                self.mj_data.ctrl[i] = action[i] * 0.04 # gripper action ionverse normalization
            elif i in {3,5}:
                self.mj_data.ctrl[i] = self.step_func(self.mj_data.ctrl[i], action[i], 16 * self.mj_model.opt.timestep)
            else:
                self.mj_data.ctrl[i] = action[i]
            self.mj_data.ctrl[i] = np.clip(self.mj_data.ctrl[i], self.mj_model.actuator_ctrlrange[i][0], self.mj_model.actuator_ctrlrange[i][1])

    def step_func(self, current, target, step):
        if current < target - step:
            return current + step
        elif current > target + step:
            return current - step
        else:
            return target

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
        self.obs["jq"][6] *= 25.0 # gripper normalization
        self.obs["jv"][6] *= 25.0 # gripper normalization
        return self.obs

    def getPrivilegedObservation(self):
        return self.obs

    def getReward(self):
        return None

if __name__ == "__main__":
    cfg = AirbotPlayCfg()
    cfg.mjcf_file_path = "mjcf/il_aloha.xml"
    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["background"] = "qz11/operation_table.ply"

    cfg.gs_model_dict["cup_blue"] = "object/cup_blue.ply"
    cfg.gs_model_dict["cup_pink"] = "object/cup_pink.ply"
    cfg.obj_list = ["cup_blue", "cup_pink"]

    exec_node = AirbotPlayBase(cfg)

    obs = exec_node.reset()

    action = exec_node.init_joint_pose[:exec_node.nj]
    while exec_node.running:
        obs, pri_obs, rew, ter, info = exec_node.step(action)
