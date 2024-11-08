import shutil
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import os
import json
import copy
import mediapy

import multiprocessing as mp

from dlabsim.airbot_play import AirbotPlayFIK
from dlabsim import DLABSIM_ROOT_DIR, DLABSIM_ASSERT_DIR
from dlabsim.envs.airbot_play_base import AirbotPlayBase, AirbotPlayCfg

data_set_size = 100



class SimNode(AirbotPlayBase):
    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.nj)
        self.jv = np.zeros(self.nj)
        self.mj_data.qpos[:self.nj] = self.init_joint_pose.copy()
        self.mj_data.qpos[self.nj] = -self.mj_data.qpos[self.nj-1]
        self.mj_data.ctrl[:self.nj] = self.init_joint_pose.copy()
        
        self.mj_data.qpos[self.nj+8] += np.random.random()*0.1 -0.05
        self.mj_data.qpos[self.nj+9] += np.random.random()*0.1 -0.05  


        mujoco.mj_forward(self.mj_model, self.mj_data)

    def getObservation(self):
        obj_pose = {}
        for name in self.config.obj_list + self.config.rb_link_list:
            obj_pose[name] = self.getObjPose(name)
        obj_pose["camera0"] = self.getCameraPose(0)
        obj_pose["camera1"] = self.getCameraPose(1)

        self.obs = {
            "time"     : self.mj_data.time,
            "jq"       : self.jq.tolist(),
            "img"      : self.img_rgb_obs_s,
            "obj_pose" : copy.deepcopy(obj_pose)
        }
        self.obs["jq"][6] *= 25.0 # gripper normalization
        return self.obs

def recoder(save_path, obs_lst):
    os.mkdir(save_path)
    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time" : [o['time'] for o in obs_lst],
            "obs"  : {
                "jq" : [o['jq'] for o in obs_lst],
            },
            "act"  : act_lst,
            "obj_pose" : {}
        }
        for name in obs_lst[0]["obj_pose"].keys():
            obj["obj_pose"][name] = [tuple(map(list, o["obj_pose"][name])) for o in obs_lst]
        json.dump(obj, fp)


    mediapy.write_video(os.path.join(save_path, "arm_video.mp4"), [o['img'][0] for o in obs_lst], fps=cfg.render_set["fps"])
    mediapy.write_video(os.path.join(save_path, "global_video.mp4"), [o['img'][1] for o in obs_lst], fps=cfg.render_set["fps"])

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    cfg = AirbotPlayCfg()
    cfg.use_gaussian_renderer = True
    cfg.gs_model_dict["block_green"] = "object/xxx.ply"
    cfg.gs_model_dict["bowl_pink"]   = "object/xxx.ply"

    cfg.mjcf_file_path = "mjcf/1block1bowl_stack.xml"
    cfg.rb_link_list = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "right", "left"]
    cfg.obj_list     = ["block_green", "bowl_pink"]
    cfg.timestep     = 1/240.
    cfg.decimation   = 4
    cfg.sync         = False
    cfg.headless     = True
    cfg.decimation   = 4
    cfg.render_set   = {
        "fps"    : 60,
        "width"  : 640,
        "height" : 480
    }
    cfg.obs_camera_id   = [0,1]
    cfg.init_joint_pose = {
        "joint1"  :  0,
        "joint2"  : -0.71966516,
        "joint3"  :  1.2772779,
        "joint4"  : -1.57079633,
        "joint5"  :  1.5563286271618402,
        "joint6"  :  1.57079633,
        "gripper" :  1
    }

    sim_node = SimNode(cfg)

    base_posi, base_quat = sim_node.getObjPose("arm_base")
    Tmat_base = np.eye(4)
    Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
    Tmat_base[:3, 3] = base_posi
    Tmat_base_inv = np.linalg.inv(Tmat_base)
    

    arm_rot_mat = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])

    tar_end_rot = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])



    state_idx = 0

    urdf_path = os.path.join(DLABSIM_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    arm_fik = AirbotPlayFIK(urdf_path)

    obs = sim_node.reset()
    
    

    data_idx = 0
    obs_lst, act_lst = [], []

    save_dir = os.path.join(DLABSIM_ROOT_DIR, "data/1cup1bowl_stack")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    process_list = []

    action = sim_node.init_joint_pose[:sim_node.nj].copy()
    
    action[6] *= 25.0

    init_control = action.copy()
    tarjq = init_control.copy()
    
    
    
    pick_pose_bias_above = [0, 0, 0.08]
    Tmat_move_bias = np.eye(4)
    Tmat_move_bias[:3,3] = pick_pose_bias_above 

    while sim_node.running:
        try:
            if state_idx == 0: # move to block_green above
                posi, quat = obs["obj_pose"]["block_green"]
                Tmat_block_g_global = np.eye(4)
                Tmat_block_g_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_block_g_global[:3, 3] = posi
                Tmat_block_g_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_block_g_global

                tar_end_pose = Tmat_block_g_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 
                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                state_idx += 1

                


            elif state_idx == 1: # moving to block_green above
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1
                    

            elif state_idx == 2: # move to block_green
                posi, quat = obs["obj_pose"]["block_green"]
                Tmat_block_g_global = np.eye(4)
                Tmat_block_g_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_block_g_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.025
                

                Tmat_block_g_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_block_g_global
                tar_end_pose = Tmat_block_g_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                state_idx += 1
               
            elif state_idx == 3: # move to block_green
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1
                  
                    

            elif state_idx == 4: # close gripper
                
                tarjq[6] = 0
                state_cnt = 0
                state_idx += 1
               

            elif state_idx == 5: # closing gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.6:
                    state_idx += 1
                   
    
            elif state_idx == 6: # pickup block_green
                posi, quat = obs["obj_pose"]["block_green"]
                Tmat_block_g_global = np.eye(4)
                Tmat_block_g_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_block_g_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.1
                

                Tmat_block_g_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_block_g_global

                tar_end_pose = Tmat_block_g_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                state_idx += 1
          

            elif state_idx == 7: # picking up block_green
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1
                    

            elif state_idx == 8: # move the block_green to bowl_pink_above
                posi, quat = obs["obj_pose"]["bowl_pink"]
                Tmat_block_g_global = np.eye(4)
                Tmat_block_g_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_block_g_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.14
                Tmat_block_g_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_block_g_global

                tar_end_pose = Tmat_block_g_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                state_idx += 1
                

            elif state_idx == 9: # moving the block_green to bowl_pink_above
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1
                

            elif state_idx == 10: # move the block_green to bowl_pink
                posi, quat = obs["obj_pose"]["bowl_pink"]
                Tmat_block_g_global = np.eye(4)
                Tmat_block_g_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_block_g_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.07
                Tmat_block_g_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_block_g_global

                tar_end_pose = Tmat_block_g_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                state_idx += 1
                

            elif state_idx == 11: # moving the block_green to bowl_pink
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1    
                              


            elif state_idx == 12: # open gripper
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    tarjq[6] = 0.9413705208843094
                    state_cnt = 0
                    state_idx += 1
                    

            elif state_idx == 13: # opening gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.35:
                    state_idx += 1
                  
                    

            elif state_idx == 14: # move to bowl_pink_above
                posi, quat = obs["obj_pose"]["bowl_pink"]
                Tmat_block_g_global = np.eye(4)
                Tmat_block_g_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_block_g_global[:3, 3] = posi
                Tmat_move_bias[2,3] = 0.14
                Tmat_block_g_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_block_g_global

                tar_end_pose = Tmat_block_g_local[:3, 3]
                rot = tar_end_rot @ arm_rot_mat 

                tarjq[:6] = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                state_idx += 1
                

            elif state_idx == 15: # moving to bowl_pink_above
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    state_idx += 1



            elif state_idx == 16: # init
                tarjq = init_control.copy()
                
                state_idx += 1
               

            elif state_idx == 17: # initing
                if np.allclose(obs["jq"][:6], tarjq[:6], atol=1e-2):
                    
                    state_idx += 1
                    action = init_control.copy()
                  
            
 
        except ValueError:
            state_cnt = 0
            tarjq = init_control.copy()
            action = init_control.copy()

            state_idx = 0
            sim_node.reset()
            obs_lst, act_lst = [], []

        for i in range(6):
            action[i] = sim_node.step_func(action[i], tarjq[i], 0.7 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
        action[6] = sim_node.step_func(action[6], tarjq[6], 2.5 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

        obs_lst.append(obs)
        act_lst.append(action.tolist())

        if state_idx > 17:
            posi_1, quat_1 = obs["obj_pose"]["bowl_pink"]
            posi_2, quat_2 = obs["obj_pose"]["block_green"]
            suc = np.linalg.norm((posi_1 - posi_2)[:2]) < 0.02
            if not suc:
                print(f"\n{data_idx} fail")
            else:

                save_path = os.path.join(save_dir, "{:03d}{}".format(data_idx, "" if suc else "_fail"))
                process = mp.Process(target=recoder, args=(save_path, obs_lst))
                process.start()
                process_list.append(process)

                data_idx += 1
                print("\r{:4}/{:4}".format(data_idx, data_set_size), end="")

            if data_idx >= data_set_size:
                break

            state_idx = 0
            sim_node.reset()

            obs_lst, act_lst= [], []

            tarjq = init_control.copy()
            action = init_control.copy()

        obs, pri_obs, rew, ter, info = sim_node.step(action)

    print("")
    for p in process_list:
        p.join()
