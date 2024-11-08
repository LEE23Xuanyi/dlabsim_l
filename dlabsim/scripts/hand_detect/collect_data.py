import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import pyrealsense2 as rs
import numpy as np
import time

def init_camera():
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  pipeline.start(config)
  return pipeline

def pose_convert(raw_data):
  poses_list = []
  for data in raw_data.landmark:
    poses_list.append([data.x, data.y, data.z])
  return poses_list

def cal_distance(hand_pose, poses):
  hand_pose = np.expand_dims(np.array(hand_pose), axis=0)
  # distance = np.linalg.norm(hand_pose[0] - hand_pose[1])
  distance = hand_pose - poses
  distance_norm = np.linalg.norm(distance, axis=2)

  return distance_norm.sum(axis=1)


if __name__ == '__main__':

  pipeline = init_camera()
  hand_poses = []
  hand_poses_mem = np.load('hand_poses.npy')
  flag = 'wait'
  count = 0
  tolerance_frame = 10
  wait_time = 3
  class_name = 1

  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands:
    
    start_time = time.time()
    while True:
      frames = pipeline.wait_for_frames()
      color_frame = frames.get_color_frame()

      if not color_frame:
          continue

      # 将帧转换为numpy数组
      color_image = np.asanyarray(color_frame.get_data())

      color_image.flags.writeable = False
      image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)

      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              hand_landmarks,
              mp_hands.HAND_CONNECTIONS,
              mp_drawing_styles.get_default_hand_landmarks_style(),
              mp_drawing_styles.get_default_hand_connections_style())

      cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
      
      # 按下 'q' 键退出
      key = cv2.waitKey(5) & 0xFF
      if key == ord('q'):
        break
      elif key == ord('s'):
        if not results.multi_hand_landmarks:
          print('no hand detected')
          continue
        hand_pose = pose_convert(results.multi_hand_landmarks[0])
        hand_poses.append(hand_pose)
        print('save',len(hand_poses))
      elif key == ord('w'):
        hand_poses = np.array(hand_poses)
        np.save(f'hand_poses{class_name}.npy', hand_poses)
        print(f'save file hand_poses{class_name}.npy done')
        
      # vaild_distance = 1.0
      # if results.multi_hand_landmarks:
      #   for hand_landmarks in results.multi_hand_landmarks:
      #     now_pose = pose_convert(hand_landmarks)
      #     distances = cal_distance(now_pose, hand_poses_mem)
      #     distances.sort()
      #     # print(distances)
      #     if distances[0] < vaild_distance:
      #       vaild_distance = distances[0]
      #   # print(vaild_distance)

      # if vaild_distance < 1.0:
      #   if flag == 'wait':
      #     vaild_time_first = time.time()
      #     flag = 'execute'
      #   vaild_time = time.time()
      #   if vaild_time - vaild_time_first > wait_time:
      #     print('Valid hand pose')
      #     count = 0
      #     # TODO

      # elif flag == 'execute' and vaild_distance == 1.0:
      #   count += 1
      #   print(f'Invalid frame {count}')
      #   if count >= tolerance_frame:
      #     print('Invalid hand pose')
      #     flag = 'wait'
      #     count = 0
        
  cv2.destroyAllWindows()

