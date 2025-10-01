
from .base_task import Base_task
from .utils import *
import sapien

class container_place(Base_task):
    def setup_demo(self,is_test = False,**kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        if is_test:
            self.id_list = [0,1,2,3,4,6,7]
        else:
            self.id_list = [8,9]
        self.load_actors()
        self.step_lim = 350
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):

        self.plate, _ = create_glb(
            self.scene,
            pose = sapien.Pose([0, -0.05, 0.753], [0.5,0.5,0.5,0.5]),
            modelname="003_plate",
            scale=[0.025,0.025,0.025],
            is_static=True,
            convex=False
        )

        container_pose = rand_pose(
            xlim=[-0.3,0.3],
            ylim=[-0.1,0.05],
            zlim=[0.8],
            rotate_rand=False,
            qpos=[0.707,0.707,0,0]
        )

        while abs(container_pose.p[0]) < 0.15:
            container_pose = rand_pose(
                xlim=[-0.3,0.3],
                ylim=[-0.1,0.05],
                zlim=[0.8],
                rotate_rand=False,
                qpos=[0.707,0.707,0,0]
            )

        self.container,self.container_data = create_glb(
            self.scene,
            pose=container_pose,
            modelname="002_container",
            model_id=np.random.choice(self.id_list),
            # model_id=self.ep_num,
            model_z_val=True
        )
        
        self.container.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.001

    def play_once(self):

        container_pose = self.container.get_pose().p
        container_edge_dis = np.array(self.container_data['extents']) * np.array(self.container_data['scale'])
        container_edge_dis = [container_edge_dis[0]/2, 0, container_edge_dis[1]/2 + 0.12]
        if container_pose[0] < 0:
            # use left arm
            container_edge_dis[0] *= -1
            pose1 = (container_pose + container_edge_dis).tolist() + [-0.5,0.5,-0.5,-0.5]
            pose1[2] += 0.08
            self.left_move_to_pose_with_screw(pose = pose1)
            pose1[2] -= 0.08
            self.left_move_to_pose_with_screw(pose = pose1)
            self.close_left_gripper(pos = -0.01)
            pose1[2] += 0.08
            self.left_move_to_pose_with_screw(pose = pose1)
            target_pose = self.get_target_pose_from_goal_point_and_direction(self.container,self.container_data,self.left_endpose,[0,-0.05,0.83],[-0.5,0.5,-0.5,-0.5])
            self.left_move_to_pose_with_screw(pose = target_pose)
            target_pose[2] -= 0.08
            self.left_move_to_pose_with_screw(pose = target_pose)
            self.open_left_gripper()
            target_pose[2] += 0.08
            self.left_move_to_pose_with_screw(pose = target_pose)
        else:
            pose1 = (container_pose + container_edge_dis).tolist() + [-0.5,0.5,-0.5,-0.5]
            pose1[2] += 0.08
            self.right_move_to_pose_with_screw(pose = pose1)
            pose1[2] -= 0.08
            self.right_move_to_pose_with_screw(pose = pose1)
            self.close_right_gripper(pos = -0.01)
            pose1[2] += 0.08
            self.right_move_to_pose_with_screw(pose = pose1)
            target_pose = self.get_target_pose_from_goal_point_and_direction(self.container,self.container_data,self.right_endpose,[0,-0.05,0.83],[-0.5,0.5,-0.5,-0.5])
            self.right_move_to_pose_with_screw(pose = target_pose)
            target_pose[2] -= 0.08
            self.right_move_to_pose_with_screw(pose = target_pose)
            self.open_right_gripper()
            target_pose[2] += 0.08
            self.right_move_to_pose_with_screw(pose = target_pose)
    

    def check_success(self):
        container_pose = self.get_actor_goal_pose(self.container,self.container_data)
        target_pose = np.array([0,-0.05, 0.74])
        eps = np.array([0.02,0.02, 0.01])
        left_gripper = self.active_joints[34].get_drive_target()[0]
        right_gripper = self.active_joints[36].get_drive_target()[0]
        endpose_z = max(self.get_right_endpose_pose().p[2], self.get_left_endpose_pose().p[2])
        return np.all(abs(container_pose - target_pose) < eps) and left_gripper > 0.04 and right_gripper > 0.04 and endpose_z > 0.98 and self.is_left_gripper_open() and self.is_right_gripper_open()