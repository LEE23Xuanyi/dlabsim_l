import cv2
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Rotation

from dlabsim import DLABSIM_ROOT_DIR
from dlabsim.scripts.aruco_tools.pose_from_aruco import ArucoDetector

cameras_intri_dict = {
    "x1" : {
        "id" : 1,
        "camera_matrix" : [
            [1651.95478,    0.     ,  946.51219],
            [   0.     , 1652.26405,  542.23279],
            [   0.     ,    0.     ,    1.     ]
        ],
        "dist_coeffs" : [ 0.17503, -0.3395,  -0.00023, -0.00228, -0.51048]
    },
    "x2" : {
        "id" : 2,
        "camera_matrix" : [
            [3303.62945,    0.     ,  978.12415],
            [   0.     , 3312.68043,  537.40249],
            [   0.     ,    0.     ,    1.     ]
        ],
        "dist_coeffs" : [ 0.16144, -1.18852,  0.0003,   0.00305, 17.45546]
    },
    "kinect" : {
        "id" : 3,
        "camera_matrix" : [ # 手标的
            [947.44872,   0.    ,  975.80396],
            [  0.     , 930.4648,  518.00402],
            [  0.     ,   0.    ,    1.     ],
        ],
        # "camera_matrix" : [ # 从设备读出来的
        #     [919.2274169921875,   0.    ,  962.2535400390625],
        #     [  0.     , 919.3020629882812,  555.55419921875],
        #     [  0.     ,   0.    ,    1.     ],
        # ],
        "dist_coeffs" : [ 0.08584,  0.04115, -0.00389,  0.0067 , -0.20302]
    }
}

Tmat_marker_2_center = {
    0 : np.array([
        [ 0.99981,  0.01961, -0.00005, -0.00276],
        [-0.01961,  0.99981,  0.00205, -0.1881 ],
        [ 0.00009, -0.00205,  1.     , -0.00088],
        [ 0.     ,  0.     ,  0.     ,  1.     ]]),
    1 : np.array([
        [ 0.99999,  0.00245,  0.00243,  0.00242],
        [-0.00245,  1.     , -0.00142,  0.18777],
        [-0.00244,  0.00141,  1.     ,  0.00037],
        [ 0.     ,  0.     ,  0.     ,  1.     ]]),
    2 : np.array([
        [ 0.99999, -0.00462, -0.00215, -0.18298],
        [ 0.00461,  0.99999, -0.00277,  0.00153],
        [ 0.00216,  0.00276,  0.99999, -0.00068],
        [ 0.     ,  0.     ,  0.     ,  1.     ]]),
    3 : np.array([
        [ 0.99983, -0.01823, -0.00338,  0.18662],
        [ 0.01823,  0.99983,  0.00147,  0.00076],
        [ 0.00336, -0.00153,  0.99999, -0.00059],
        [ 0.     ,  0.     ,  0.     ,  1.     ]]),
}

def quaternion_to_rotation_vector(q):
    # 将四元数 (x, y, z, w) 转换为旋转向量
    q = q / np.linalg.norm(q)
    angle = 2 * np.arccos(q[3])  # w分量
    if angle == 0:
        return np.zeros(3)
    v = q[:3] / np.sin(angle / 2)
    return angle * v

def rotation_vector_to_quaternion(rv):
    # 将旋转向量转换回四元数 (x, y, z, w)
    angle = np.linalg.norm(rv)
    if angle == 0:
        return np.array([0, 0, 0, 1])
    v = rv / angle
    s = np.sin(angle / 2)
    return np.array([v[0] * s, v[1] * s, v[2] * s, np.cos(angle / 2)])

def average_quaternions(quaternions):
    # 对多个四元数 (x, y, z, w) 取平均
    rotation_vectors = [quaternion_to_rotation_vector(q) for q in quaternions]
    mean_rotation_vector = np.mean(rotation_vectors, axis=0)
    mean_quaternion = rotation_vector_to_quaternion(mean_rotation_vector)
    return mean_quaternion

if __name__ == "__main__":
    import os
    show = False
    output_debug = True
    add_file = False

    camera = "kinect"

    object_name = "jimu/3"
    save_dir = os.path.join(DLABSIM_ROOT_DIR, f"data/object/{object_name}")

    np.set_printoptions(precision=5, suppress=True, linewidth=500)

    camera_matrix = np.array(cameras_intri_dict[camera]["camera_matrix"])
    dist_coeffs = np.array(cameras_intri_dict[camera]["dist_coeffs"])
    cam_id = cameras_intri_dict[camera]["id"]

    det = ArucoDetector(cv2.aruco.DICT_4X4_50, camera_matrix, dist_coeffs, 0.053)

    img_forder = os.path.join(save_dir, "ori_images")
    if not os.path.exists(img_forder):
        img_forder = os.path.join(save_dir, "images")

    img_files = sorted(os.listdir(img_forder))
    imgs = [cv2.imread(os.path.join(img_forder, img_file)) for img_file in tqdm.tqdm(img_files)]

    width, height = imgs[0].shape[1], imgs[0].shape[0]
    cameras_intrinsic_file = os.path.join(save_dir, "cameras.txt")
    with open(cameras_intrinsic_file, 'w') as fp:
        for c in cameras_intri_dict.keys():
            cam_mat = np.array(cameras_intri_dict[c]["camera_matrix"])
            fp.write("{} PINHOLE {} {} {} {} {} {}\n".format(
                cameras_intri_dict[c]["id"],
                width, 
                height, 
                cam_mat[0,0],
                cam_mat[1,1],
                cam_mat[0,2],
                cam_mat[1,2]))

    cap_id = 0
    cameras_extrinsic_file = os.path.join(save_dir, "images.txt")
    if add_file:
        with open(cameras_extrinsic_file, "r") as fp:
            lines = fp.readlines()
            for l in lines:
                if len(l):
                    cap_id += 1
            print("images.txt : line cnt={}".format(cap_id))
    ext_fp = open(cameras_extrinsic_file, "a" if add_file else "w")

    extrinsic_dict = {}
    cameras_extrinsic_json = cameras_extrinsic_file.replace(".txt", ".json")

    mujoco_site = {
        "m2c_0" : "",
        "m2c_1" : "",
        "m2c_2" : "",
        "m2c_3" : "",
        "w2c_0" : "",
        "w2c_1" : "",
        "w2c_2" : "",
        "w2c_3" : "",
        "c2mc" : "",
        "res" : "",
    }

    Tmats_cam2marker = {
        0 : [],
        1 : [],
        2 : [],
        3 : [],
    }
    Tmats_cam2markercenter = []
    world2cam_frame = None

    for imgid, image in enumerate(tqdm.tqdm(imgs)):
        img, Rmats, tvecs, ids, _ = det.detect(image)
        if len(ids) > 1:
            cap_id += 1

            Tmats = []
            for i in range(len(ids)):
                if not ids[i][0] in Tmats_cam2marker.keys():
                    continue
                Tmat_cam2marker = np.eye(4)
                Tmat_cam2marker[:3,:3] = Rmats[i]
                Tmat_cam2marker[:3,3] = tvecs[i].T[0]
                Tmats_cam2marker[ids[i][0]].append(Tmat_cam2marker.copy())
                Tmat_cam2center = Tmat_cam2marker @ Tmat_marker_2_center[ids[i][0]]
                Tmats.append(Tmat_cam2center.copy())

                if output_debug:
                    Tcam_mat = np.linalg.inv(Tmat_cam2marker)
                    posi = Tcam_mat[:3,3]
                    quat_xyzw = Rotation.from_matrix(Tcam_mat[:3,:3]).as_quat()
                    mujoco_site["m2c_{}".format(ids[i][0])] += '<site name="m{}_2c_{}" pos="{} {} {}" quat="{} {} {} {}" size="0.001" type="sphere"/>\n'.format(
                        ids[i][0], img_files[imgid].split(".")[0], posi[0], posi[1], posi[2],
                        quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

                    Tcam_mat = np.linalg.inv(Tmat_cam2center)
                    posi = Tcam_mat[:3,3]
                    quat_xyzw = Rotation.from_matrix(Tcam_mat[:3,:3]).as_quat()
                    mujoco_site["w2c_{}".format(ids[i][0])] += '<site name="cam_m{}_{}" pos="{} {} {}" quat="{} {} {} {}" size="0.001" type="sphere"/>\n'.format(
                        ids[i][0], img_files[imgid].split(".")[0], posi[0], posi[1], posi[2],
                        quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

            quats = []
            posis = []
            for Tmat_cam2marker in Tmats:
                posi = Tmat_cam2marker[:3,3]
                quat_xyzw = Rotation.from_matrix(Tmat_cam2marker[:3,:3]).as_quat()
                quats.append(quat_xyzw)
                posis.append(posi)
            quat_cam2markercenter = average_quaternions(quats)
            posi_cam2markercenter = np.mean(posis, axis=0)

            T_c2mc = np.eye(4)
            T_c2mc[:3,:3] = Rotation.from_quat(quat_cam2markercenter).as_matrix()
            T_c2mc[:3,3] = posi_cam2markercenter
            Tmats_cam2markercenter.append(T_c2mc)
            
            Tmat_mc2cam = np.linalg.inv(T_c2mc)
            posi = Tmat_mc2cam[:3,3]
            rmat = Tmat_mc2cam[:3,:3]
            quat_xyzw = Rotation.from_matrix(rmat).as_quat()

            qvec = Rotation.from_matrix(rmat.T).as_quat()
            tvec = -rmat.T @ posi

            ext_fp.write("{} {} {} {} {} {} {} {} {} {}\n".format(
                cap_id, 
                qvec[3], qvec[0], qvec[1], qvec[2], 
                tvec[0], tvec[1], tvec[2],
                cam_id,
                img_files[imgid]))

            if output_debug:
                mujoco_site["c2mc"] += '<site name="c2mc_{}" pos="{} {} {}" quat="{} {} {} {}" size="0.001" type="sphere"/>\n'.format(
                    img_files[imgid].split(".")[0], posi_cam2markercenter[0], posi_cam2markercenter[1], posi_cam2markercenter[2],
                    quat_cam2markercenter[3], quat_cam2markercenter[0], quat_cam2markercenter[1], quat_cam2markercenter[2],
                )

                mujoco_site["res"] += '<site name="mc2c_res_{}" pos="{} {} {}" quat="{} {} {} {}" size="0.001" type="sphere"/>\n'.format(
                    img_files[imgid].split(".")[0], posi[0], posi[1], posi[2],
                    quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2],
                )

            extrinsic_dict[img_files[imgid].split(".")[0]] = {
                "position" : posi.tolist(),
                "orientation" : quat_xyzw.tolist()
            }

            if world2cam_frame is None and len(ids) == 4:
                world2cam_frame = np.eye(4)
                world2cam_frame[:3,:3] = Rotation.from_quat(quat_xyzw).as_matrix()
                world2cam_frame[:3,3] = posi

            if show:
                cv2.imshow('Image', img)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break
        else:
            print("{} nothing".format(img_files[imgid]))
            # 删除对应的图片
            os.remove(os.path.join(img_forder, img_files[imgid]))
            continue

    ext_fp.close()

    with open(cameras_extrinsic_json, "w") as fp:
        json.dump(extrinsic_dict, fp)

    if world2cam_frame is None:
        world2cam_frame = np.eye(4)
        world2cam_frame[:3,:3] = Rotation.from_quat(quat_xyzw).as_matrix()
        world2cam_frame[:3,3] = posi

    if show:
        cv2.destroyAllWindows()

    if output_debug:
        output_forder = os.path.join(save_dir, "aruco_log")
        if not os.path.exists(output_forder):
            os.makedirs(output_forder)
        else:
            for f in os.listdir(output_forder):
                os.remove(os.path.join(output_forder, f))

        for key in mujoco_site.keys():
            with open(os.path.join(output_forder, f"{key}_pose.txt"), 'w') as fp:
                fp.write(mujoco_site[key])

        for key in Tmats_cam2marker.keys():
            if len(Tmats_cam2marker[key]):
                with open(os.path.join(output_forder, f"pose_cam2marker{key}.txt"), 'w') as fp:
                    for mat in Tmats_cam2marker[key]:
                        Tobj0 = world2cam_frame @ mat
                        posi = Tobj0[:3,3]
                        quat_xyzw = Rotation.from_matrix(Tobj0[:3,:3]).as_quat()
                        fp.write('<site pos="{} {} {}" quat="{} {} {} {}" size="0.001" type="sphere"/>\n'.format(
                            posi[0], posi[1], posi[2],
                            quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2],
                        ))
