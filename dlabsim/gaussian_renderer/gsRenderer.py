import os
import cv2
import torch
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from dlabsim.gaussian_renderer import util_gau
from dlabsim.gaussian_renderer.renderer_cuda import CUDARenderer

from dlabsim import DLABSIM_ASSERT_DIR

class GSRenderer:
    def __init__(self, models_dict:dict, render_width=1920, render_height=1080):
        self.width = render_width
        self.height = render_height

        self.camera = util_gau.Camera(self.height, self.width)

        self.update_gauss_data = False

        self.scale_modifier = 1.

        self.renderer = CUDARenderer(self.camera.w, self.camera.h)
        self.camera_tran = np.zeros(3)
        self.camera_quat = np.zeros(4)

        self.gaussians_all:dict[util_gau.GaussianData] = {}
        self.gaussians_idx = {}
        self.gaussians_size = {}
        idx_sum = 0

        gs_model_dir = Path(os.path.join(DLABSIM_ASSERT_DIR, "3dgs"))

        bg_key = "background"
        data_path = Path(os.path.join(gs_model_dir, models_dict[bg_key]))
        gs = util_gau.load_ply(data_path)
        self.gaussians_all[bg_key] = gs
        self.gaussians_idx[bg_key] = idx_sum
        self.gaussians_size[bg_key] = gs.xyz.shape[0]
        idx_sum = self.gaussians_size[bg_key]
        for i, (k, v) in enumerate(models_dict.items()):
            if k != "background":
                data_path = Path(os.path.join(gs_model_dir, v))
                gs = util_gau.load_ply(data_path)
                self.gaussians_all[k] = gs
                self.gaussians_idx[k] = idx_sum
                self.gaussians_size[k] = gs.xyz.shape[0]
                idx_sum += self.gaussians_size[k]

        self.update_activated_renderer_state(self.gaussians_all)

        for name in self.gaussians_all.keys():
            # :TODO: 找到哪里被改成torch了
            try:
                self.gaussians_all[name].R = self.gaussians_all[name].R.numpy()
            except:
                pass

    def update_camera_intrin_lazy(self):
        if self.camera.is_intrin_dirty:
            self.renderer.update_camera_intrin(self.camera)
            self.camera.is_intrin_dirty = False

    def update_activated_renderer_state(self, gaus: util_gau.GaussianData):
        self.renderer.update_gaussian_data(gaus)
        self.renderer.set_scale_modifier(self.scale_modifier)
        self.renderer.update_camera_pose(self.camera)
        self.renderer.update_camera_intrin(self.camera)
        self.renderer.set_render_reso(self.camera.w, self.camera.h)

    def set_obj_pose(self, obj_name, trans, quat_wzyx):
        if not ((self.gaussians_all[obj_name].origin_rot == quat_wzyx).all() and (self.gaussians_all[obj_name].origin_xyz == trans).all()):
            self.update_gauss_data = True
            self.gaussians_all[obj_name].origin_rot = quat_wzyx.copy()
            self.gaussians_all[obj_name].origin_xyz = trans.copy()
            self.renderer.gau_xyz_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(trans).cuda().requires_grad_(False)
            self.renderer.gau_rot_all_cu[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name],:] = torch.from_numpy(quat_wzyx).cuda().requires_grad_(False)

    def set_camera_pose(self, trans, quat_xyzw):
        if not ((self.camera_tran == trans).all() and (self.camera_quat == quat_xyzw).all()):
            self.camera_tran[:] = trans[:]
            self.camera_quat[:] = quat_xyzw[:]
            rmat = Rotation.from_quat(quat_xyzw).as_matrix()
            self.renderer.update_camera_pose_from_topic(self.camera, rmat, trans)

    def set_camera_fovy(self, fovy):
        if not fovy == self.camera.fovy:
            self.camera.fovy = fovy
            self.camera.is_intrin_dirty = True

    def render(self, render_depth=False):
        self.update_camera_intrin_lazy()
        return self.renderer.draw(render_depth)

if __name__ == "__main__":
    import rospy
    from std_srvs.srv import Empty
    from geometry_msgs.msg import PoseArray

    rospy.init_node('GaussianSplattingNode', anonymous=True)

    def pose_array_callback(msg:PoseArray):
        global g_pose_array_msg_recv, g_pose_arr
        g_pose_array_msg_recv = True
        g_pose_arr = msg

    g_pose_array_msg_recv = False
    g_pose_arr = PoseArray()
    g_pose_arr.header.frame_id = ""
    rospy.Subscriber('object_poses', PoseArray, pose_array_callback)

    while not rospy.is_shutdown():
        try:
            update_object_service = rospy.ServiceProxy('/update_object', Empty)
            response = update_object_service()
            break
        except rospy.ServiceException as e:
            rospy.logwarn_once("Service call failed: %s" % e)
            rospy.sleep(1.0)

    models_lst = [
        "qz11/table.ply",
        "qz11/qz_table.ply",
        "object/cup_blue.ply",
        "object/cup_pink.ply",
        "airbot_play/arm_base.ply",
        "airbot_play/link1.ply",
        "airbot_play/link2.ply",
        "airbot_play/link3.ply",
        "airbot_play/link4.ply",
        "airbot_play/link5.ply",
        "airbot_play/link6.ply",
        "airbot_play/left.ply",
        "airbot_play/right.ply",
    ]
    gsrender = GSRenderer(models_lst)

    cv2.namedWindow("GaussianSplattingNode", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("GaussianSplattingNode", gsrender.width, gsrender.height)
    
    import time
    stt = time.time()

    rate = rospy.Rate(50)
    delta_t = 0.02
    while cv2.getWindowProperty("GaussianSplattingNode", cv2.WND_PROP_VISIBLE):
        if g_pose_array_msg_recv:
            g_pose_array_msg_recv = False
            obj_names = g_pose_arr.header.frame_id.split(";")
            if len(obj_names):
                if len(obj_names) == len(g_pose_arr.poses):
                    for name, pose in zip(obj_names, g_pose_arr.poses):
                        trans = np.array([pose.position.x, pose.position.y, pose.position.z])
                        # quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                        quat = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
                        if name in gsrender.gaussians_all.keys():
                            gsrender.set_obj_pose(name, trans, quat)
                        elif name == "camera":
                            gsrender.set_camera_pose(trans, quat)
                            gsrender.renderer.need_rerender = True
                else:
                    print("Wrong size of poseArr len(name)={} while len(poseArr.poses)={}".format(len(obj_names), len(g_pose_arr.poses)))
        if gsrender.update_gauss_data:
            new_tmat_cu = gsrender.renderer.gau_tmats_all_cu[gsrender.renderer.gau_env_idx:] @ gsrender.renderer.gau_ori_tmats_all_cu[gsrender.renderer.gau_env_idx:]
            gsrender.renderer.gaussians.xyz[gsrender.renderer.gau_env_idx:] = new_tmat_cu[:, :3, 3]
            gsrender.renderer.gaussians.rot[gsrender.renderer.gau_env_idx:] = matrix_to_quaternion(new_tmat_cu[:, :3, :3]).requires_grad_(False)

            # self.renderer.gaussians.xyz[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name]] = self.gaussians_all[obj_name].xyz_cu
            # self.renderer.gaussians.rot[self.gaussians_idx[obj_name]:self.gaussians_idx[obj_name]+self.gaussians_size[obj_name]] = self.gaussians_all[obj_name].rot_cu

        img_rgb = gsrender.render()
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        delta_t = 0.1 * (time.time() - stt) + 0.9 * delta_t
        stt = time.time()

        cv2.putText(img, "FPS: {:.2f}".format(1.0/delta_t), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("GaussianSplattingNode", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        rate.sleep()

    cv2.destroyAllWindows()