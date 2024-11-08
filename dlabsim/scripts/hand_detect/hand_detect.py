import os
import cv2
import time
import numpy as np
import mediapipe as mp

class HandDetect:
    def __init__(self, tolerance_frame=10, wait_time=1, tolerance_score=0.8):
        self.hand_poses = []

        poses_mem = []
        for npy in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)))):
            if npy.endswith('.npy'):
                poses_mem.append(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), npy)))
        if len(poses_mem) > 1:
            self.hand_poses_mem = np.vstack(poses_mem)
        elif len(poses_mem) == 1:
            self.hand_poses_mem = poses_mem[0]
        else:
            exit('No hand pose data')

        print(poses_mem[0].shape)
        print(self.hand_poses_mem.shape)

        self.flag = 'wait'
        self.count = 0
        self.tolerance_frame = tolerance_frame
        self.wait_time = wait_time
        self.tolerance_score = tolerance_score

        self.mpDraw = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            model_complexity         = 0,
            min_detection_confidence = 0.5,
            min_tracking_confidence  = 0.5)

    def pose_convert(self, raw_data):
        poses_list = []
        for data in raw_data.landmark:
            poses_list.append([data.x, data.y, data.z])
        return poses_list

    def cal_distance(self, hand_pose, poses):
        hand_pose = np.array(hand_pose)
        keypoint_index = np.array([0,1,2,3,4,5,9,13,17])
        hand_pose = hand_pose[keypoint_index]
        poses = poses[:, keypoint_index, :]
        hand_pose = np.expand_dims(hand_pose, axis=0)
        distance = hand_pose - poses
        distance_norm = np.linalg.norm(distance, axis=2)
        return distance_norm.sum(axis=1)

    def detect(self, image_bgr, draw=False):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        vaild_distance = self.tolerance_score
        if results.multi_hand_landmarks:
            if draw:
                for hand in results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(image_bgr, hand, self.mp_hands.HAND_CONNECTIONS)

            for hand_landmarks in results.multi_hand_landmarks:
                now_pose = self.pose_convert(hand_landmarks)
                distances = self.cal_distance(now_pose, self.hand_poses_mem)
                distances.sort()
                if distances[0] < vaild_distance:
                    vaild_distance = distances[0]
        
        if vaild_distance < self.tolerance_score:
            if self.flag == 'wait':
                self.vaild_time_first = time.time()
                self.flag = 'execute'
            
            self.vaild_time = time.time()
            if self.vaild_time - self.vaild_time_first > self.wait_time:
                print('Valid hand pose')
                self.count = 0
                return True
        
        elif self.flag == 'execute' and vaild_distance == self.tolerance_score:
            self.count += 1
            print(f'Invalid frame {self.count}')
            if self.count >= self.tolerance_frame:
                print('Invalid hand pose')
                self.flag = 'wait'
                self.count = 0
        
        return False
    
if __name__ == "__main__":
    import cv2
    import pyrealsense2 as rs

    def init_camera():
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        return pipeline
    pipeline = init_camera()

    detector = HandDetect(class_name=1, tolerance_frame=10, wait_time=3)
    
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_image.flags.writeable = False
        image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        print(detector.detect(image))

