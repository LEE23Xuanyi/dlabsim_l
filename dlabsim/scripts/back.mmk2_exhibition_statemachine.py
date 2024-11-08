import time
import rospy
import threading
import numpy as np
from types import MethodType

from dlabsim.envs.mmk2_base import MMK2Cfg
from dlabsim.envs import SimulatorBase, SimulatorGSBase

from mmk2_exhibition import MMK2Exhibition

from slamtec.base_control import SlamtecBaseControl
from super_statemachine.super_statemachine import Box
from super_statemachine.super_statemachine import SuperStateMachine

class MMK2ExhibitionWithSM(SuperStateMachine):
    shelf_row_2_floor_height = {
        1 : 0.10,
        2 : 0.42,
        3 : 0.74,
        4 : 1.06,
    }
    last_prize_id = None

    def __init__(self, mmk2_instantiation:MMK2Exhibition):
        super().__init__()

        self.request_lock = threading.Lock()

        self.slt_ctl = SlamtecBaseControl()
        self.mmk2 = mmk2_instantiation
        self.mmk2.teleop = None
        if isinstance(self.mmk2, SimulatorGSBase):
            self.mmk2.cv2WindowKeyPressCallback = MethodType(SimulatorGSBase.cv2WindowKeyPressCallback, self.mmk2)
        else:
            self.mmk2.cv2WindowKeyPressCallback = MethodType(SimulatorBase.cv2WindowKeyPressCallback, self.mmk2)
        # with self.request_lock:
        #     self.mmk2.action_normal()

    def run(self):
        thr_stmt = threading.Thread(target=self.state_trans_start, args=())
        thr_stmt.start()

        while self.mmk2.running and not rospy.is_shutdown():
            with self.request_lock:
                self.mmk2.view()
            time.sleep(0.1)
        thr_stmt.join()

    def state_trans_start(self):
        self.request_distribute()
    
    # ------------------------------------------------------------------------------------------------------------------------- #
    def pickup_one_prize(self, prize_id):
        with self.request_lock:
            # prize_id: 0, 1, 2
            if self.last_prize_id is None or (not self.last_prize_id is None and self.last_prize_id != prize_id):
                self.mmk2.moveto_prize(prize_id)
            self.last_prize_id = prize_id
            self.mmk2.arm_action = "pick"
            self.mmk2.action_new_look()
            time.sleep(1.)
            self.mmk2.sysc_real_robot()
            for _ in range(5):
                self.mmk2.pick_prize_select_front_pose = self.mmk2.select_prize(prize_id)
                if not self.mmk2.pick_prize_select_front_pose is None:
                    self.mmk2.base_servo_pick()
                    if not self.mmk2.pick_prize_select_front_pose is None:
                        break
                time.sleep(0.5)
            else:
                print(f"<STATEMACHION> I can't find the prize {prize_id} [0,1,2]")
                raise Exception(f"<STATEMACHION> Can't find the prize {prize_id} [0,1,2]")
            time.sleep(1)
            res = self.mmk2.action_new_pickup(60) # :TODO: 等待时间60s
            if res is None:
                print(f"<STATEMACHION> Pickup prize {prize_id} failed")
                raise Exception(f"<STATEMACHION> Pickup prize {prize_id} failed")
            else:
                if res:
                    print(f"<STATEMACHION> Pickup prize {prize_id} and handover success")
                    self.mmk2.action_normal()
                else:
                    print(f"<STATEMACHION> Pickup prize {prize_id} success, but handover failed")

    # ------------------------------------------------------------------------------------------------------------------------- #
    def move_to_shelf(self):
        print("<STATEMACHION> I have arrived shelf")
        if self.holding_box is not None:
            #################################################################################################
            # 抱着空盒子
            self.last_prize_id = None
            with self.request_lock:
                target_pose = self.mmk2.box_shelf_position_map[self.holding_box.box_position[0]]
                self.mmk2.moveto_fromhome(target_pose)
                self.mmk2.moveto_shelf_servo()
            # point = self.mmk2.box_shelf_position_map[self.holding_box.box_position[0]]
            # self.slt_ctl.go_to(point[0], point[1], point[2])
            #################################################################################################
            # print("Press Enter to lay old box...")
            self.request_lay_old_box()
        else:
            #################################################################################################
            # 空手
            with self.request_lock:
                target_pose = self.mmk2.box_shelf_position_map[self.avaliable_box.box_position[0]]
                self.mmk2.moveto_fromhome(target_pose)
                self.mmk2.moveto_shelf_servo()
            # point = self.mmk2.box_shelf_position_map[self.avaliable_box.box_position[0]]
            # self.slt_ctl.go_to(point[0], point[1], point[2])
            #################################################################################################
            # print("Press Enter to pick new box...")
            self.request_pick_new_box()

    def pick_new_box(self):
        print("<STATEMACHION> I come to pick new box")
        #################################################################################################
        with self.request_lock:
            self.mmk2.action_carry_height = self.shelf_row_2_floor_height[self.avaliable_box.box_position[1]]
            self.mmk2.arm_action = "carry"
            self.mmk2.action_new_look()
            for _ in range(10):
                self.mmk2.pick_box_pose = self.mmk2.select_box_nearest(self.mmk2.action_carry_height)
                if not self.mmk2.pick_box_pose is None:
                    self.mmk2.base_servo_carry()
                    if not self.mmk2.pick_box_pose is None:
                        break
                time.sleep(0.25)
            if self.mmk2.pick_box_pose is None:
                print("<STATEMACHION> I can't find the box")
                raise Exception("<STATEMACHION> Can't find the box")
            self.mmk2.action_carry(self.mmk2.action_carry_height)
        #################################################################################################
        self.holding_box = self.avaliable_box
        self.holding_box.status = 'holding'
        for i in range(self.shelf_box_data['number']):
            if (self.avaliable_box.is_equal(self.shelf_box_data['boxes']['prize_number'][i], self.shelf_box_data['boxes']['prize_type'][i], [self.shelf_box_data['boxes']['box_column'][i], self.shelf_box_data['boxes']['box_row'][i]])):
                # remove the box from json file
                self.shelf_box_data['boxes']['prize_number'].pop(i)
                self.shelf_box_data['boxes']['prize_type'].pop(i)
                self.shelf_box_data['boxes']['box_column'].pop(i)
                self.shelf_box_data['boxes']['box_row'].pop(i)
                self.shelf_box_data['number'] -= 1
                break
        self._update_json()
        self.avaliable_box = None
        print("<STATEMACHION> I have picked new box")
        # input("Press Enter to move to desk...")
        self.request_to_desk()

    def pick_old_box(self):
        if self.desk_boxes[self.target_prize] is None:
            print("<STATEMACHION> There is no old box to pick")
        else:
            #################################################################################################
            with self.request_lock:
                self.mmk2.arm_action = "carry"
                self.mmk2.action_carry_height = self.mmk2.table_height
                self.mmk2.action_new_look()
                for _ in range(5):
                    self.mmk2.pick_box_pose = self.mmk2.select_box_nearest(self.mmk2.action_carry_height)
                    if not self.mmk2.pick_box_pose is None:
                        self.mmk2.base_servo_carry()
                        if not self.mmk2.pick_box_pose is None:
                            break
                    time.sleep(0.25)
                else:
                    print("<STATEMACHION> I can't find the box")
                    raise Exception("<STATEMACHION> Can't find the box")
                self.mmk2.action_carry(self.mmk2.action_carry_height)
            #################################################################################################
            self.holding_box = self.desk_boxes[self.target_prize]
            self.holding_box.status = 'holding'
            self.desk_boxes[self.target_prize] = None
            print("<STATEMACHION> I have picked old box")
        # input("Press Enter to move to shelf...")
        self.request_to_shelf()

    def move_to_desk(self):
        self.target_base_position = self.box_desk_position_map[self.target_prize]
        #################################################################################################
        if self.target_prize == "first_prize":
            prize_id = 0
        elif self.target_prize == "second_prize":
            prize_id = 1
        elif self.target_prize == "third_prize":
            prize_id = 2
        else:
            raise ValueError(f"<STATEMACHION> Invalid target prize: {self.target_prize}")
        with self.request_lock:
            self.mmk2.moveto_prize(prize_id)
        #################################################################################################
        print("I have arrived desk")
        # input(">>> Press Enter to lay new box...")
        self.arrived_desk()

    def lay_new_box(self):
        #################################################################################################
        with self.request_lock:
            self.mmk2.action_carry_height = self.mmk2.table_height
            self.mmk2.action_put_box(self.mmk2.action_carry_height)
        #################################################################################################
        print("<STATEMACHION> I have laid new box")
        self.desk_boxes[self.target_prize] = self.holding_box
        self.desk_boxes[self.target_prize].status = 'on_desk'
        self.holding_box = None
        # input(">>> Press Enter to distribute boxes...")
        self.request_distribute()

    def lay_old_box(self):
        #################################################################################################
        with self.request_lock:
            # [column, row]
            plat_height = self.shelf_row_2_floor_height[self.holding_box.box_position[1]]
            self.mmk2.action_put_box(plat_height)
        #################################################################################################
        print("<STATEMACHION> I have laid old box")
        self.shelf_box_data['boxes']['prize_number'].append(self.holding_box.number)
        self.shelf_box_data['boxes']['prize_type'].append(self.holding_box.prize_type)
        self.shelf_box_data['boxes']['box_column'].append(self.holding_box.box_position[0])
        self.shelf_box_data['boxes']['box_row'].append(self.holding_box.box_position[1])
        self.shelf_box_data['number'] += 1
        self._update_json()
        self.holding_box = None
        # input(">>> Press Enter to pick new box...")
        self.request_to_shelf()

    def distribute_prize(self):
        if not self.no_box_empty():
            # input(">>> Press Enter to update the prize...")
            self.request_update()
            return
        while True:
            print("I am distributing prize")
            ipt = input(">>> Enter prize box number to distribute (1,2,3)...\n")
            while True:
                if ipt == "q":
                    exit(0)
                elif ipt in {"1","2","3"}:
                    target_prize_number = int(ipt)
                    break
            print("================================================")
            # :TODO:
            if target_prize_number == 1:
                self.target_prize = 'first_prize'
            elif target_prize_number == 2:
                self.target_prize = 'second_prize'
            elif target_prize_number == 3:
                self.target_prize = 'third_prize'
            else:
                print("Invalid prize box number")
                continue
            print("target_prize :", self.target_prize)
            self.pickup_one_prize(target_prize_number-1)
            self.desk_boxes[self.target_prize].take_one()
            if self.desk_boxes[self.target_prize].is_empty():
                print("The prize box", self.target_prize, "is empty")
                # input(">>> Press Enter to update the boxes...")
                self.request_update()
            time.sleep(0.25)

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=500)
    rospy.init_node('mujoco_exhibition_pickup_statemachine_node', anonymous=True)

    cfg = MMK2Cfg()
    cfg.init_key = "pick"
    cfg.obs_camera_id = -2
    cfg.render_set["fps"] = 25
    cfg.render_set["width"] = 1440
    cfg.render_set["height"] = 1080
    cfg.mjcf_file_path = "mjcf/exhibition_conference.xml"
    # cfg.rb_link_list =  [
    #     "agv_link", "slide_link", "head_yaw_link", "head_pitch_link",
    #     "lft_arm_base", "lft_arm_link1", "lft_arm_link2",
    #     "lft_arm_link3", "lft_arm_link4", "lft_arm_link5", "lft_arm_link6",
    #     "lft_finger_left_link", "lft_finger_right_link",
    #     "rgt_arm_base", "rgt_arm_link1", "rgt_arm_link2",
    #     "rgt_arm_link3", "rgt_arm_link4", "rgt_arm_link5", "rgt_arm_link6",
    #     "rgt_finger_left_link", "rgt_finger_right_link"
    # ]
    cfg.rb_link_list = []
    cfg.obj_list = []

    exec_node = MMK2Exhibition(cfg)

    if isinstance(exec_node, SimulatorGSBase):
        models_lst = [
            "exhibition/booth.ply",

            "mmk2/mmk2_base/agv_link.ply",
            "mmk2/mmk2_base/slide_link.ply",
            "mmk2/mmk2_base/head_pitch_link.ply",
            "mmk2/mmk2_base/head_yaw_link.ply",

            "mmk2/left_arm/lft_arm_base.ply",
            "mmk2/left_arm/lft_arm_link1.ply",
            "mmk2/left_arm/lft_arm_link2.ply",
            "mmk2/left_arm/lft_arm_link3.ply",
            "mmk2/left_arm/lft_arm_link4.ply",
            "mmk2/left_arm/lft_arm_link5.ply",
            "mmk2/left_arm/lft_arm_link6.ply",
            "mmk2/left_arm/lft_finger_left_link.ply",
            "mmk2/left_arm/lft_finger_right_link.ply",
 
            "mmk2/right_arm/rgt_arm_base.ply",
            "mmk2/right_arm/rgt_arm_link1.ply",
            "mmk2/right_arm/rgt_arm_link2.ply",
            "mmk2/right_arm/rgt_arm_link3.ply",
            "mmk2/right_arm/rgt_arm_link4.ply",
            "mmk2/right_arm/rgt_arm_link5.ply",
            "mmk2/right_arm/rgt_arm_link6.ply",
            "mmk2/right_arm/rgt_finger_left_link.ply",
            "mmk2/right_arm/rgt_finger_right_link.ply"
        ]
        exec_node.init_gs_render(models_lst)

    exec_node.reset()

    top_state_machine = MMK2ExhibitionWithSM(exec_node)

    # b1 = Box(number=2, prize_type="first_prize" , status="on_desk", box_position=[1, 4])
    # b2 = Box(number=1, prize_type="second_prize", status="on_desk", box_position=[1, 3])
    # b3 = Box(number=1, prize_type="third_prize" , status="on_desk", box_position=[1, 2])
    # top_state_machine.desk_boxes = {'first_prize': b1, 'second_prize': b2, 'third_prize': b3}

    top_state_machine.run()
