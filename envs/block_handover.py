
from .base_task import Base_task
from .utils import *
import sapien
import math

class block_handover(Base_task):
    def setup_demo(self,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        self.load_actors()
        self.step_lim = 600
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)

        self.render_freq = render_freq

    def load_actors(self):
        rand_pos = rand_pose(
            xlim=[-0.25,-0.05],
            ylim=[0.,0.25],
            zlim=[0.842],
            qpos=[-0.906,0,0,-0.424]
        )
        self.box = create_box(
            scene = self.scene,
            pose = rand_pos,
            half_size=(0.03,0.03,0.1),
            color=(1,0,0),
            name="box"
        )

        rand_pos = rand_pose(
            xlim=[0.23,0.23],
            ylim=[0.09,0.09],
            zlim=[0.74],
        )

        self.target = create_box(
            scene = self.scene,
            pose = rand_pos,
            half_size=(0.05,0.05,0.005),
            color=(0,0,1),
            name="box"
        )
        self.target.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 1
        self.box.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.1

    def play_once(self):
        left_pose0 = list(self.box.get_pose().p+[-0.14,-0.18,0.07])+[-0.906,0,0,-0.424]
        left_pose1 = list(self.box.get_pose().p+[-0.08,-0.11,0.07])+[-0.906,0,0,-0.424]
        left_target_pose = [-0.19,-0.12,0.96,1,0,0,0]
        right_pick_pre_pose = [0.191,-0.12,0.87,0,0,0,1]
        right_pick_pose = [0.09,-0.12,0.85,0,0,0,1]
        self.left_move_to_pose_with_screw(left_pose0)
        self.left_move_to_pose_with_screw(left_pose1)
        self.close_left_gripper()
        
        left_pose1[2] +=0.06
        self.left_move_to_pose_with_screw(left_pose1)
        self.together_move_to_pose_with_screw(left_target_pose,right_pick_pre_pose)

        self.right_move_to_pose_with_screw(right_pick_pose)
        
        self.close_right_gripper()
        
        self.open_left_gripper()
        
        right_pick_pose[0]+=0.05
        left_target_pose[0]-=0.1
        self.together_move_to_pose_with_screw(left_target_pose,right_pick_pose)
        right_target_pose = list(self.target.get_pose().p + [0.02,-0.13,0.11]) + [0.707,0,0,0.707]

        self.right_move_to_pose_with_screw(right_target_pose)
        right_target_pose[2] -= 0.04
        self.right_move_to_pose_with_screw(right_target_pose)

        self.open_right_gripper()
        right_target_pose[1] -= 0.1
        right_target_pose[2] += 0.1
        self.right_move_to_pose_with_screw(right_target_pose)

    def check_success(self):
        box_pos = self.box.get_pose().p
        target_pose = self.target.get_pose().p
        if box_pos[2] < 0.78:
            self.actor_pose = False
        eps = 0.02
        right_endpose = self.get_right_endpose_pose()
        endpose_target_pose = [0.241,-0.129,0.889,0,-0.7,-0.71,0]
        return abs(box_pos[0] - target_pose[0]) < eps and abs(box_pos[1] - target_pose[1]) < eps and abs(box_pos[2] - 0.85) < 0.0015 and\
               np.all(abs(np.array(right_endpose.p.tolist() + right_endpose.q.tolist()) - endpose_target_pose ) < 0.2 * np.ones(7)) and self.is_right_gripper_open()