
from .base_task import Base_task
from .utils import *
import sapien
import math

class dual_bottles_pick_hard(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        self.load_actors()
        self.step_lim = 400
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0

        self.together_close_gripper(save_freq=None)
        self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def load_actors(self):
        # super().setup_scene()

        qpose = [[1,0,0,0],[0.707,0.707,0,0]]
        ylim = [[-0.2,0.05],[0.03, 0.023]]
        zlim = [[0.785], [0.865]]
        rotate_lim = [(0,0,1.4), (0,0,0)]

        tag = np.random.randint(0,2)
        self.red_bottle,self.red_bottle_data = rand_create_glb(
            self.scene,
            xlim=[-0.25,-0.1],
            ylim=ylim[tag],
            zlim=zlim[tag],
            modelname="001_bottles",
            rotate_rand=True,
            rotate_lim = rotate_lim[tag],
            qpos=qpose[tag],
            model_id=13
        )

        tag = np.random.randint(0,2)
        self.green_bottle,self.green_bottle_data =rand_create_glb(
            self.scene,
            xlim=[0.1,0.25],
            ylim=ylim[tag],
            zlim=zlim[tag],
            modelname="001_bottles",
            rotate_rand=True,
            rotate_lim = rotate_lim[tag],
            qpos=qpose[tag],
            model_id=16
        )

        self.red_bottle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01
        self.green_bottle.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.01

        render_freq = self.render_freq
        self.render_freq = 0
        for _ in range(4):
            self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def play_once(self):

        if self.red_bottle.get_pose().p[2] > 0.8:
            left_pose0 = self.get_grasp_pose_w_given_direction(self.red_bottle, self.red_bottle_data,grasp_qpos=[-0.906,0,0,-0.424], pre_dis=0.1)
            left_pose1 = self.get_grasp_pose_w_given_direction(self.red_bottle, self.red_bottle_data,grasp_qpos=[-0.906,0,0,-0.424], pre_dis=0)
        else:
            left_pose1 = self.get_grasp_pose_w_labeled_direction(self.red_bottle, self.red_bottle_data,grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0)
            left_pose0 = self.get_grasp_pose_w_labeled_direction(self.red_bottle, self.red_bottle_data,grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0.1)

        if self.green_bottle.get_pose().p[2] > 0.8:
            right_pose0 = self.get_grasp_pose_w_given_direction(self.green_bottle, self.green_bottle_data,grasp_qpos=[-0.415,0,0,-0.910], pre_dis=0.11)
            right_pose1 = self.get_grasp_pose_w_given_direction(self.green_bottle, self.green_bottle_data,grasp_qpos=[-0.415,0,0,-0.910], pre_dis=0.01)
        else:
            right_pose1 = self.get_grasp_pose_w_labeled_direction(self.green_bottle, self.green_bottle_data,grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0)
            right_pose0 = self.get_grasp_pose_w_labeled_direction(self.green_bottle, self.green_bottle_data,grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0.1)


        left_target_pose = [-0.19,-0.12,0.92,1,0,0,0]
        right_target_pose = [0.19,-0.12,0.92,-0.01,0.01,0.03,-1]
        # pre_grasp
        
        self.together_move_to_pose_with_screw(left_pose0,right_pose0)

        self.together_move_to_pose_with_screw(left_pose1,right_pose1)
        self.together_close_gripper()

        left_pose1[2]+=0.1
        right_pose1[2]+=0.1
        self.together_move_to_pose_with_screw(left_pose1,right_pose1)
        self.together_move_to_pose_with_screw(left_target_pose,right_target_pose)

    def check_success(self):
        red_target = [-0.07,-0.11]
        green_target = [0.06,-0.11]
        eps = 0.03
        red_bottle_pose = self.red_bottle.get_pose().p
        green_bottle_pose = self.green_bottle.get_pose().p
        
        return abs(red_bottle_pose[0]-red_target[0])<eps and abs(red_bottle_pose[1]-red_target[1])<eps and red_bottle_pose[2]>0.9 and\
               abs(green_bottle_pose[0]-green_target[0])<eps and abs(green_bottle_pose[1]-green_target[1])<eps and green_bottle_pose[2]>0.9
