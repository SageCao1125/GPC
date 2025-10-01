
from .base_task import Base_task
from .utils import *
import sapien

class block_hammer_beat(Base_task):

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
        pass

    def load_actors(self):
        self.hammer, self.hammer_data = create_glb(
            self.scene,
            pose=sapien.Pose([0, -0.06, 0.783],[0, 0, 0.995, 0.105]),
            modelname="020_hammer_2",
        )
        block_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.05,0.15],
            zlim=[0.76],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0,1,0],
        )

        while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2],2)) < 0.001:
            block_pose = rand_pose(
                xlim=[-0.25,0.25],
                ylim=[-0.05,0.15],
                zlim=[0.76],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0,1,0],
            )

        self.block = create_box(
            scene = self.scene,
            pose = block_pose,
            half_size=(0.025,0.025,0.025),
            color=(1,0,0),
            name="box"
        )
        self.hammer.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.001
        self.block.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.001

    def play_once(self):

        pose1 = self.get_grasp_pose_w_labeled_direction(self.hammer, self.hammer_data, pre_dis=0.1) # pre grasp pose
        pose2 = self.get_grasp_pose_w_labeled_direction(self.hammer, self.hammer_data, pre_dis=0.01) # grap pose
        if self.block.get_pose().p[0] > 0:
            # use right arm
            self.open_right_gripper() 
            self.right_move_to_pose_with_screw(pose1)
            self.right_move_to_pose_with_screw(pose2)
            self.close_right_gripper()
            pose2[2] += 0.07
            self.right_move_to_pose_with_screw(pose2)
            pose3 = self.get_target_pose_from_goal_point_and_direction(self.hammer,self.hammer_data,self.right_endpose,self.block.get_pose().p+[0,0,0.08],[-0.55,0.45,-0.45,-0.55])
            self.right_move_to_pose_with_screw(pose3)
            pose3[2] -= 0.06
            self.right_move_to_pose_with_screw(pose3)
        else:
            self.open_left_gripper()
            self.left_move_to_pose_with_screw(pose1)
            self.left_move_to_pose_with_screw(pose2)
            self.close_left_gripper()
            pose2[2] += 0.07
            self.left_move_to_pose_with_screw(pose2)
            pose3 = self.get_target_pose_from_goal_point_and_direction(self.hammer,self.hammer_data,self.left_endpose,self.block.get_pose().p+[0,0,0.08],[-0.55,0.45,-0.45,-0.55])
            self.left_move_to_pose_with_screw(pose3)
            pose3[2] -= 0.06
            self.left_move_to_pose_with_screw(pose3)
        

    def check_success(self):
        hammer_target_pose = self.get_actor_goal_pose(self.hammer,self.hammer_data)
        block_pose = self.block.get_pose().p
        eps = np.array([0.02,0.02])
        return np.all(abs(hammer_target_pose[:2] - block_pose[:2])<eps) and hammer_target_pose[2] < 0.81 and hammer_target_pose[2] > 0.78