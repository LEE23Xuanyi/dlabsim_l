import numpy as np

from dlabsim.envs import SimulatorBase
from dlabsim.gaussian_renderer import GSRenderer
from dlabsim.gaussian_renderer.util_gau import multiple_quaternion_vector3d, multiple_quaternions

class SimulatorGSBase(SimulatorBase):
    gs_renderer = None
    gaussian_render = True

    def init_gs_render(self, model_set):
        self.gs_renderer = GSRenderer(model_set, self.config.render_set["width"], self.config.render_set["height"])
        self.last_cam_id = self.cam_id
        if self.cam_id == -1:
            self.gs_renderer.set_camera_fovy(np.pi * 0.25)
        else:
            self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[self.cam_id] * np.pi / 180.0)

    def update_gs_scene(self):
        for name in self.config.obj_list + self.config.rb_link_list:
            trans, quat_wxyz = self.getObjPose(name)
            self.gs_renderer.set_obj_pose(name, trans, quat_wxyz)

        if self.gs_renderer.update_gauss_data:
            self.gs_renderer.update_gauss_data = False
            self.gs_renderer.renderer.need_rerender = True
            self.gs_renderer.renderer.gaussians.xyz[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternion_vector3d(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]) + self.gs_renderer.renderer.gau_xyz_all_cu[self.gs_renderer.renderer.gau_env_idx:]
            self.gs_renderer.renderer.gaussians.rot[self.gs_renderer.renderer.gau_env_idx:] = multiple_quaternions(self.gs_renderer.renderer.gau_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:], self.gs_renderer.renderer.gau_ori_rot_all_cu[self.gs_renderer.renderer.gau_env_idx:])

    def getDepthImg(self, cam_id):
        if self.gaussian_render:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            if self.last_cam_id != cam_id:
                if cam_id == -1:
                    self.gs_renderer.set_camera_fovy(np.pi * 0.25)
                elif cam_id > -1:
                    self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
                else:
                    return None
            self.last_cam_id = cam_id
            self.update_gs_scene()
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render(render_depth=True)
        else:
            return super().getDepthImg(cam_id)

    def getRgbImg(self, cam_id):
        if self.gaussian_render:
            if cam_id == -1:
                self.renderer.update_scene(self.mj_data, self.free_camera, self.options)
            if self.last_cam_id != cam_id:
                if cam_id == -1:
                    self.gs_renderer.set_camera_fovy(np.pi * 0.25)
                elif cam_id > -1:
                    self.gs_renderer.set_camera_fovy(self.mj_model.cam_fovy[cam_id] * np.pi / 180.0)
                else:
                    return None
            self.last_cam_id = cam_id
            self.update_gs_scene()
            trans, quat_wxyz = self.getCameraPose(cam_id)
            self.gs_renderer.set_camera_pose(trans, quat_wxyz[[1,2,3,0]])
            return self.gs_renderer.render()
        else:
            return super().getRgbImg(cam_id)
        
    def cv2WindowKeyPressCallback(self, key):
        ret = super().cv2WindowKeyPressCallback(key)
        if key == ord('g'):
            self.gaussian_render = not self.gaussian_render
            self.gs_renderer.renderer.need_rerender = True
        elif key == ord("d"):
            self.gs_renderer.renderer.need_rerender = True
        elif key == ord("m"):
            np.save("/home/tatp/Desktop/depth.npy", self.gs_renderer.renderer.depth_img.permute(1, 2, 0).cpu().numpy())
        return ret

    def printHelp(self):
        ret = super().printHelp()
        print("Press 'g' toggle gaussian render")
        print("Press 'd' toggle depth render")
        return ret
