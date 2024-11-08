import os
import rospy
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim import DLABSIM_ASSERT_DIR
from dlabsim.airbot_play import AirbotPlayFIK

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
from std_srvs.srv import Empty, EmptyResponse, EmptyRequest

from airbot_play_base import AirbotPlayCfg, AirbotPlayBase

class AirbotPlayJoyCtl(AirbotPlayBase):
    def __init__(self, config: AirbotPlayCfg):
        super().__init__(config)

        urdf_path = os.path.join(DLABSIM_ASSERT_DIR, "urdf/airbot_play_v3_gripper_fixed.urdf")
        self.arm_fik = AirbotPlayFIK(urdf_path)
    
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)
        self.rot_mat = np.array([
            [ 0., -0.,  1.],
            [ 0.,  1.,  0.],
            [-1.,  0.,  0.]
        ])
        self.tar_jq = np.zeros(self.nj)

        self.object_poses_puber = rospy.Publisher('/object_poses', PoseArray, queue_size=5)

        self.joint_state_puber = rospy.Publisher('/airbot_play/joint_states', JointState, queue_size=5)
        self.joint_state = JointState()
        self.joint_state.name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.joint_state.position = [0.] * self.nj
        self.joint_state.velocity = [0.] * self.nj
        self.joint_state.effort = [0.] * self.nj

        self.if_update_object = False
        self.update_all_cnt = 0
        rospy.Service('/update_object', Empty, self.empty_service_callback)

    def empty_service_callback(self, request:EmptyRequest):
        rospy.loginfo("Update object pose")
        self.if_update_object = True
        self.update_all_cnt = int(1. / self.config.decimation / self.mj_model.opt.timestep)
        return EmptyResponse()

    def resetState(self):
        super().resetState()

        self.tar_jq = np.zeros(self.nj)
        self.tar_end_pose = np.array([0.295, -0., 0.219])
        self.tar_end_euler = np.zeros(3)
        self.update_all_cnt = int(1. / self.config.decimation / self.mj_model.opt.timestep)

    def updateControl(self, action):
        super().updateControl(self.tar_jq)

    def teleopProcess(self):
        calc_ik = False
        if self.teleop.joy_cmd.axes[0] or self.teleop.joy_cmd.axes[1] or self.teleop.joy_cmd.axes[4]:
            calc_ik = True
            self.tar_end_pose[0] += 0.15 * self.teleop.joy_cmd.axes[1] * self.delta_t
            self.tar_end_pose[1] += 0.15 * self.teleop.joy_cmd.axes[0] * self.delta_t
            self.tar_end_pose[2] += 0.1 * self.teleop.joy_cmd.axes[4] * self.delta_t

        if self.teleop.joy_cmd.axes[3] or self.teleop.joy_cmd.axes[6] or self.teleop.joy_cmd.axes[7]:
            calc_ik = True
            self.tar_end_euler[0] += 0.01 * self.teleop.joy_cmd.axes[3]
            self.tar_end_euler[1] += 0.01 * self.teleop.joy_cmd.axes[7]
            self.tar_end_euler[2] += 0.01 * self.teleop.joy_cmd.axes[6]

        if calc_ik:
            rot = Rotation.from_euler('xyz', self.tar_end_euler).as_matrix() @ self.rot_mat
            try:
                tarjq = self.arm_fik.inverseKin(self.tar_end_pose, rot, self.jq[:6])
            except ValueError:
                tarjq = None
            if not tarjq is None:
                self.tar_jq[:6] = tarjq
            else:
                rospy.logwarn("Fail to solve inverse kinematics trans={} euler={}".format(self.tar_end_pose, self.tar_end_euler))

        if self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]:
            self.tar_jq[6] += 1. * (self.teleop.joy_cmd.axes[2] - self.teleop.joy_cmd.axes[5]) * self.delta_t
            self.tar_jq[6] = np.clip(self.tar_jq[6], 0, 1.)

    def getChangedObjectPose(self):
        obj_pose_arr = PoseArray()
        obj_pose_arr.header.stamp = rospy.Time.now()
        obj_pose_arr.header.frame_id = ""
        if self.if_update_object or self.camera_pose_changed or self.cam_id != -1:
            self.camera_pose_changed = False
            obj_pose_arr.header.frame_id += "camera;"

            trans, quat = self.getCameraPose(self.cam_id)

            camera_pose = Pose()
            camera_pose.position.x = trans[0]
            camera_pose.position.y = trans[1]
            camera_pose.position.z = trans[2]
            camera_pose.orientation.w = quat[0]
            camera_pose.orientation.x = quat[1]
            camera_pose.orientation.y = quat[2]
            camera_pose.orientation.z = quat[3]

            obj_pose_arr.poses.append(camera_pose)
        
        if self.if_update_object:
            name_list = self.sinobj_dict.keys()
        else:
            name_list = self.getObjectNameUpdate()

        for name in name_list:
            obj_pose_arr.header.frame_id += "{};".format(name)

            obj = self.sinobj_dict[name]
            p = Pose()
            p.position.x = obj.position[0]
            p.position.y = obj.position[1]
            p.position.z = obj.position[2]
            p.orientation.w = obj.quat_wxyz[0]
            p.orientation.x = obj.quat_wxyz[1]
            p.orientation.y = obj.quat_wxyz[2]
            p.orientation.z = obj.quat_wxyz[3]
            obj_pose_arr.poses.append(p)

        if len(obj_pose_arr.header.frame_id):
            obj_pose_arr.header.frame_id = obj_pose_arr.header.frame_id[:-1]
            return obj_pose_arr
        else:
            return None

    def post_physics_step(self):
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state.position = self.jq.tolist()
        self.joint_state.velocity = self.jv.tolist()
        self.joint_state.effort = [0] * len(self.joint_state.name)
        self.joint_state_puber.publish(self.joint_state)

        if self.if_update_object:
            self.update_all_cnt -= 1
            if self.update_all_cnt <= 0:
                self.if_update_object = False

        pose_arr_msg = self.getChangedObjectPose()
        if not pose_arr_msg is None:
            self.object_poses_puber.publish(pose_arr_msg)

    def getObservation(self):
        return self.obs

    def printMessage(self):
        print("-" * 100)
        print("mj_data.time = {:.3f}".format(self.mj_data.time))
        print("joint tar_q = {}".format(np.array2string(self.tar_jq, separator=', ')))
        print("joint q     = {}".format(np.array2string(self.jq, separator=', ')))
        print("joint v     = {}".format(np.array2string(self.jv, separator=', ')))

        print("target end posi  = {}".format(np.array2string(self.tar_end_pose, separator=', ')))
        print("target end euler = {}".format(np.array2string(self.tar_end_euler, separator=', ')))

        if self.cam_id == -1:
            print(self.free_camera)
        else:
            print(self.mj_data.camera(self.camera_names[self.cam_id]))

if __name__ == "__main__":
    import rospy
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    rospy.init_node('Airbot_play_mujoco_node', anonymous=True)

    cfg = AirbotPlayCfg()
    cfg.decimation = 4
    cfg.timestep   = 0.001
    cfg.use_gaussian_renderer = True
    cfg.obj_list = ["cup_blue", "cup_pink"]
    cfg.gs_model_dict.insert(0, "qz11/qz_table_new.ply")
    cfg.gs_model_dict.insert(1, "qz11/table_trans.ply")
    cfg.gs_model_dict += ["object/cup_blue.ply", "object/cup_pink.ply"]

    exec_node = AirbotPlayJoyCtl(cfg)

    while exec_node.running:
        exec_node.step()