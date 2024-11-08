#import source
import shutil
import mujoco
import numpy as np
from enum import Enum, auto
from scipy.spatial.transform import Rotation

import os
import json
import mediapy
import traceback
import multiprocessing as mp

from dlabsim.airbot_play import AirbotPlayFIK
from dlabsim import DLABSIM_ROOT_DIR, DLABSIM_ASSERT_DIR
from dlabsim.envs.mmk2_base import MMK2Base, MMK2Cfg


class SimNode(MMK2Base):
    def resetState(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.teleop:
            self.teleop.reset()

        self.jq = np.zeros(self.njq)
        self.jv = np.zeros(self.njv)
        #self.mj_data.qpos[:self.njq] = self.init_joint_pose.copy()
        #self.mj_data.ctrl[:self.njctrl] = self.init_joint_pose.copy()
        
        self.mj_data.qpos[self.njq] = np.random.random() * 0.07 + 0.45 - 0.035
        self.mj_data.qpos[self.njq+1] = np.random.random() * 0.07 - 0.3375 - 0.035

        self.mj_data.qpos[self.njq+7] = np.random.random() * 0.07 + 0.45 - 0.035
        self.mj_data.qpos[self.njq+8] = np.random.random() * 0.07 + 0.2625 - 0.035


        mujoco.mj_forward(self.mj_model, self.mj_data)
    def getObservation(self):
        obj_pose = {} 
        for name in self.config.obj_list:
            obj_pose[name] = self.getObjPose(name)

        self.obs = {
            "time"     : self.mj_data.time, # current time
            "jq"       : self.jq.tolist(), # joint position
            "img"      : self.img_rgb_obs, # RGB image observation
            "obj_pose" : obj_pose # dictionary containing the poses of all trackeed objects
        }
        return self.obs

    def updateControl(self, action):
        super().updateControl(action)
        # self.mj_data.qpos[:7] = self.init_joint_pose[:7].copy()
        # self.mj_data.qvel[:6] = 0.0
def recoder(save_path, obs_lst):
    os.mkdir(save_path)
    with open(os.path.join(save_path, "obs_action.json"), "w") as fp:
        obj = {
            "time" : [o['time'] for o in obs_lst],
            "obs"  : {
                "jq" : [o['jq'] for o in obs_lst],
            },
            "act"  : act_lst
        }
        json.dump(obj, fp)
        
    mediapy.write_video(os.path.join(save_path, "video.mp4"), [o['img'] for o in obs_lst], fps=cfg.render_set["fps"])

# ValueError: Fail to solve inverse kinematics: pos=[0.521 0.037 0.367], ori=[[-0.707 -0.707  0.   ]
#  [ 0.     0.     1.   ]
#  [-0.707  0.707  0.   ]]

# discribe robot in all kinds of action state
class StateBuildBlocks(Enum):
    SBB_SLEEPING                = auto()
    SBB_LIFT_DOWN               = auto()
    SBB_LIFT_DOWN_ING           = auto()
    SBB_MOVE_TO_CUBE_ABOVE      = auto()
    SBB_MOVE_TO_CUBE_ABOVE_ING  = auto()
    SBB_MOVE_TO_CUBE            = auto()
    SBB_MOVE_TO_CUBE_ING        = auto()
    SBB_CLOSE_GRIPPER           = auto()
    SBB_CLOSE_GRIPPER_ING       = auto()
    SBB_PICK_UP                 = auto()
    SBB_PICK_UP_ING             = auto()
    SBB_LIFT_UP                 = auto()
    SBB_LIFT_UP_ING             = auto()
    SBB_MOVE_TO_TARGET          = auto()
    SBB_MOVE_TO_TARGET_ING      = auto()
    SBB_OPEN_GRIPPER            = auto()
    SBB_OPEN_GRIPPER_ING        = auto()
    SBB_ARM_BACK                = auto()
    SBB_ARM_BACK_ING            = auto()
    SBB_END                     = auto()
    SBB_BREAK                   = auto()

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True, linewidth=500)

    #print setting  0.000 max500linewidth without science notation used

    cfg = MMK2Cfg()
    cfg.mjcf_file_path = "/home/leexuanyi/.mujoco/mujoco-3.2.2-linux-x86_64/DLABSIM（复件）/models/mjcf/cupcupcup.xml"
    cfg.rb_link_list = []
    cfg.obj_list     = ["cup_blue", "cup_pink","cup_purple"]
    cfg.sync         = False # the simulator will run in asynchronous mode
    cfg.headless     = False # the simulation should run with a graphical interface
    cfg.decimation   = 4 # 4 step render 1 clip
    cfg.render_set   = {
        "fps"    : 50,  
        "width"  : 1920,
        "height" : 1080,
        # "width"  : 1280,
        # "height" : 720,
    }
    cfg.obs_camera_id   = 0

    sim_node = SimNode(cfg)

    # sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    # sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # sim_node.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # # options.flags[mujoco.mjtVisFlag.mjVIS_COM] = True
    # # options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    # # options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    # sim_node.options.frame = mujoco.mjtFrame.mjFRAME_BODY.value
 
    cube_idx = 0

    arm_rot_mat = np.array([
        [ 0., -0.,  1.],
        [ 0.,  1.,  0.],
        [-1.,  0.,  0.]
    ])

    tar_end_rot_left = np.array([
        [ 0., -0.707,  0.707],
        [ 1.,  0.   ,  0.   ],
        [ 0.,  0.707,  0.707],
    ])

    tar_end_rot_right = np.array([
        [ 0.,  0.707,  0.707],
        [-1.,  0.   ,  0.   ],
        [ 0., -0.707,  0.707],
    ])

    Tmat_move_bias = np.eye(4)

    state_sbb = StateBuildBlocks.SBB_SLEEPING

    urdf_path = os.path.join(DLABSIM_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
    arm_fik = AirbotPlayFIK(urdf_path)

    obs = sim_node.reset()

    data_idx = 0
    data_set_size = 200
    obs_lst, act_lst = [], []

    save_dir = os.path.join(DLABSIM_ROOT_DIR, "data/stackingcup_mmk2")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    process_list = []
    
    action = sim_node.init_ctrl.copy()
    chassis_cmd = action[:2]
    lift_cmd = action[2:3]
    head_cmd = action[3:5]
    left_arm_cmd = action[5:12]
    right_arm_cmd = action[12:19]

    chassis_cmd_buf = chassis_cmd.copy()
    lift_cmd_buf = lift_cmd.copy()
    head_cmd_buf = head_cmd.copy()
    left_arm_cmd_buf = left_arm_cmd.copy()
    right_arm_cmd_buf = right_arm_cmd.copy()

    head_cmd_buf[1] = 1


    pick_obj_name_goal = cfg.obj_list[len(cfg.obj_list)-1]
    posi_goal, quat_goal = obs["obj_pose"][pick_obj_name_goal]
    posi_goal[2] += 0.09


    #input("Press Enter to continue...")

    while sim_node.running:
        if state_sbb == StateBuildBlocks.SBB_SLEEPING: # move to cup above
            state_sbb = StateBuildBlocks(state_sbb.value+1)

        try:
            if state_sbb == StateBuildBlocks.SBB_LIFT_DOWN: # move to cup above
                lift_cmd_buf[0] = 0.6
                left_arm_cmd_buf[6] = 0.04
                right_arm_cmd_buf[6] = 0.04
                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_LIFT_DOWN_ING:
                if np.allclose(obs["jq"][7+2], lift_cmd_buf[0], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
                    

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE_ABOVE:
                pick_obj_name = cfg.obj_list[cube_idx]

                posi, quat = obs["obj_pose"][pick_obj_name]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi

                arm_site_name = "lft_arm_base_link" if posi[1] > -0.0375 else "rgt_arm_base_link"

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.19
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])

                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                    right_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, -1.5708, -0.88] 
                else:
                    left_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, 1.5708, 0.88]
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE_ABOVE_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
                   

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE:
                posi, quat = obs["obj_pose"][pick_obj_name]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi
                Tmat_cup_b_global[2,3] += 0.05

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.02
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global
                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                else:
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_CUBE_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
                  

            elif state_sbb == StateBuildBlocks.SBB_CLOSE_GRIPPER:
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[6] = 0.0
                else:
                    right_arm_cmd_buf[6] = 0.0
                state_cnt = 0
                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_CLOSE_GRIPPER_ING:
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.5:
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
               

            elif state_sbb == StateBuildBlocks.SBB_PICK_UP:
                posi, quat = obs["obj_pose"][pick_obj_name]
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.1
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                else:
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)
               

            elif state_sbb == StateBuildBlocks.SBB_PICK_UP_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
               
            elif state_sbb == StateBuildBlocks.SBB_LIFT_UP:
                lift_cmd_buf[0] = 0.5 
                state_sbb = StateBuildBlocks(state_sbb.value+1)

            elif state_sbb == StateBuildBlocks.SBB_LIFT_UP_ING:
                if np.allclose(obs["jq"][7+2], lift_cmd_buf[0], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
               

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_TARGET: # move the cup_blue/pink to cup_purple
                pick_obj_name_goal = cfg.obj_list[len(cfg.obj_list)-1]
                posi_goal, quat_goal = obs["obj_pose"][pick_obj_name_goal]
                posi_goal[2] += 0.18
                if "lft" in arm_site_name:
                    posi_goal[1] -= 0.029
                else:
                    posi_goal[1] += 0.029
                posi = posi_goal
                quat = np.array([1.0, 0.0, 0.0, 0.0])
                Tmat_cup_b_global = np.eye(4)
                Tmat_cup_b_global[:3,:3] = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
                Tmat_cup_b_global[:3, 3] = posi

                base_posi, base_quat = sim_node.getObjPose(arm_site_name)
                Tmat_base = np.eye(4)
                Tmat_base[:3,:3] = Rotation.from_quat(base_quat[[1,2,3,0]]).as_matrix()
                Tmat_base[:3, 3] = base_posi
                Tmat_base_inv = np.linalg.inv(Tmat_base)

                Tmat_move_bias[1,3] = 0.05
                if "lft" in arm_site_name:
                    Tmat_move_bias[1,3] *= -1

                Tmat_cup_b_local = Tmat_move_bias @ Tmat_base_inv @ Tmat_cup_b_global

                tar_end_pose = Tmat_cup_b_local[:3, 3]
                if "lft" in arm_site_name:
                    rot = tar_end_rot_left @ arm_rot_mat
                else:
                    rot = tar_end_rot_right @ arm_rot_mat

                jres = arm_fik.inverseKin(tar_end_pose, rot, np.array(obs["jq"])[:6])
                if "lft" in arm_site_name:
                    left_arm_cmd_buf[:6] = jres
                else:
                    right_arm_cmd_buf[:6] = jres

                state_sbb = StateBuildBlocks(state_sbb.value+1)
                print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_MOVE_TO_TARGET_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_cnt = 0
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
               
            elif state_sbb == StateBuildBlocks.SBB_OPEN_GRIPPER: # opening gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.25:
                    if "lft" in arm_site_name:
                        left_arm_cmd_buf[6] = 0.035
                    else:
                        right_arm_cmd_buf[6] = 0.035
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)

            elif state_sbb == StateBuildBlocks.SBB_OPEN_GRIPPER_ING: # opening gripper
                state_cnt += 1
                if state_cnt * sim_node.config.decimation * sim_node.mj_model.opt.timestep > 0.5:
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
           

            elif state_sbb == StateBuildBlocks.SBB_ARM_BACK:
                left_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, 1.5708, 0.88]
                right_arm_cmd_buf[:6] = [0.0, -0.847,  1.2 ,  0.0, -1.5708, -0.88]
                state_sbb = StateBuildBlocks(state_sbb.value+1)

            elif state_sbb == StateBuildBlocks.SBB_ARM_BACK_ING:
                if np.allclose(obs["jq"][12:18], left_arm_cmd_buf[:6], atol=1e-2) and np.allclose(obs["jq"][20:26], right_arm_cmd_buf[:6], atol=1e-2):
                    state_sbb = StateBuildBlocks(state_sbb.value+1)
                    print(state_sbb)
            

            elif state_sbb == StateBuildBlocks.SBB_END:
                state_sbb = StateBuildBlocks.SBB_SLEEPING
                cube_idx += 1
                #print(state_sbb.value)
                #print(cube_idx)
                #print((len(cfg.obj_list)-1))
                #if cube_idx == len(cfg.obj_list)-1:
                #    input("Press Enter to continue...")
                #  break

        except ValueError:
            traceback.print_exc()
            break

            state_cnt = 0
            action = sim_node.init_ctrl.copy()
            state_sbb = StateBuildBlocks.SBB_SLEEPING

            sim_node.reset()
            obs_lst, act_lst = [], []

        for i in range(2):
            chassis_cmd[i] = sim_node.step_func(chassis_cmd[i], chassis_cmd_buf[i], 2.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

        for i in range(1):
            lift_cmd[i] = sim_node.step_func(lift_cmd[i], lift_cmd_buf[i], 5.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
        
        for i in range(2):
            head_cmd[i] = sim_node.step_func(head_cmd[i], head_cmd_buf[i], 5.0 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
        
        for i in range(7):
            if i == 2:
                left_arm_cmd[i]  = sim_node.step_func(left_arm_cmd[i] , left_arm_cmd_buf[i] , 2.4 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
                right_arm_cmd[i] = sim_node.step_func(right_arm_cmd[i], right_arm_cmd_buf[i], 2.4 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
            elif i == 6:
                left_arm_cmd[i]  = sim_node.step_func(left_arm_cmd[i] , left_arm_cmd_buf[i] , 10. * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
                right_arm_cmd[i] = sim_node.step_func(right_arm_cmd[i], right_arm_cmd_buf[i], 10. * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
            else:
                left_arm_cmd[i]  = sim_node.step_func(left_arm_cmd[i] , left_arm_cmd_buf[i] , 1.6 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)
                right_arm_cmd[i] = sim_node.step_func(right_arm_cmd[i], right_arm_cmd_buf[i], 1.6 * sim_node.config.decimation * sim_node.mj_model.opt.timestep)

        obs_lst.append(obs)
        act_lst.append(action.tolist())

        #if state_sbb.value > 20:
        if  cube_idx == len(cfg.obj_list)-1:

            posi_add = 0
            for i in range(len(cfg.obj_list)):
             pick_obj_name = cfg.obj_list[i]
             posi,quat = obs["obj_pose"][pick_obj_name_goal]
             posi_add += posi[1]

            posi_add = posi_add + 0.0375*(i+1)

            if abs(posi_add) > 0.02*i : # y_pos_diff of each cup compared with y_pos_goal is less than 0.02
                continue
            else :
                print("successful operation")
          

            save_path = os.path.join(save_dir, "{:03d}".format(data_idx))
            process = mp.Process(target=recoder, args=(save_path, obs_lst))
            process.start()
            process_list.append(process)
            data_idx += 1
            print("\r{:4}/{:4}".format(data_idx, data_set_size), end="") 
            print("\n")
            cube_idx = 0          
            if data_idx >= 200 : #data_set_size:               
                 break
            

            state_sbb = StateBuildBlocks.SBB_SLEEPING

            #chassis_cmd_buf = chassis_cmd.copy()
            #lift_cmd_buf = lift_cmd.copy()
            #head_cmd_buf = head_cmd.copy()
            #left_arm_cmd_buf = left_arm_cmd.copy()
            #right_arm_cmd_buf = right_arm_cmd.copy()
          
            sim_node.reset()
            obs_lst, act_lst= [], []

        obs, pri_obs, rew, ter, info = sim_node.step(action)

    print("")
for p in process_list:
    p.join()
