
from .base_task import Base_task
from .utils import *
import math
import sapien

class pick_apple_messy(Base_task):
    def setup_demo(self,is_test = False, **kwags):
        super()._init(**kwags)
        self.create_table_and_wall()
        self.load_robot()
        self.setup_planner()
        self.load_camera()
        self.pre_move()
        if is_test:
            self.id_list = [2*i+1 for i in range(5)]
        else:
            self.id_list = [2*i for i in range(5)]
        self.load_actors()
        self.step_lim = 250
    
    def pre_move(self):
        render_freq = self.render_freq
        self.render_freq=0
        self.together_open_gripper(save_freq=None)
        self.render_freq = render_freq

    def load_actors(self):
        self.actor_list=[]
        self.actor_data_list=[]
        file_path = './envs/utils/rand_model_data.json'
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        pose_list = []
        obj_num = 4
        obj_list = np.random.choice([i for i in range(data["number"])], size = obj_num, replace=True)

        apple_pose = rand_pose(
            xlim=[-0.25,0.25],
            ylim=[-0.1,0.1],
            zlim=[0.78],
            ylim_prop=True,
            rotate_rand=False,
        )
        pose_list.append(apple_pose)
        self.apple, self.apple_data = create_obj(
            self.scene,
            pose=apple_pose, # the z value (0.783) is important
            modelname="035_apple",
            convex=True
        )
        for i in obj_list:
            model_index = f"model{i}"
            actor_pose = rand_pose(
                xlim=data[model_index]["xlim"],
                ylim=data[model_index]["ylim"],
                zlim=data[model_index]["zlim"],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=data[model_index]["rotate_lim"],
                qpos=data[model_index]["init_qpos"]
            )

            tag = True
            while tag:
                tag = False
                for pose in pose_list:
                    if np.sum(pow(pose.p[:2] - actor_pose.p[:2],2)) < 0.0225:
                        tag = True
                        break
                if tag:
                    actor_pose = rand_pose(
                        xlim=data[model_index]["xlim"],
                        ylim=data[model_index]["ylim"],
                        zlim=data[model_index]["zlim"],
                        ylim_prop=True,
                        rotate_rand=True,
                        rotate_lim=data[model_index]["rotate_lim"],
                        qpos=data[model_index]["init_qpos"]
                    )

            pose_list.append(actor_pose)
            model, model_data = create_actor(
                self.scene,
                pose=actor_pose,
                modelname=data[model_index]["name"],
                convex=True,
                model_z_val = data[model_index]["model_z_val"],
            )

            self.actor_list.append(model)
            self.actor_data_list.append(model_data)

    def play_once(self):
        pose1 = self.get_grasp_pose_w_given_direction(self.apple, self.apple_data, grasp_qpos=[-0.5, 0.5, -0.5, -0.5], pre_dis=0.1) # apple pre grasp pose
        pose2 = self.get_grasp_pose_w_given_direction(self.apple, self.apple_data, grasp_qpos=[-0.5, 0.5, -0.5, -0.5]) # apple grasp pose
        
        if self.apple.get_pose().p[0] < 0:
            self.open_left_gripper()  # open the left gripper first
            self.left_move_to_pose_with_screw(pose1) # left arm move to the pre grasp pose
            self.left_move_to_pose_with_screw(pose2) # left arm move to the grasp pose
            self.close_left_gripper()  # close left gripper to grasp the apple
            pose2[2] += 0.05  # lift the apple up for 5 cm
            self.left_move_to_pose_with_screw(pose2) # lift the apple up for 5 cm
        else:
            self.open_right_gripper()  # open the right gripper first
            self.right_move_to_pose_with_screw(pose1) # right arm move to the pre grasp pose
            self.right_move_to_pose_with_screw(pose2) # right arm move to the grasp pose
            self.close_right_gripper()  # close right gripper to grasp the apple
            pose2[2] += 0.05  # lift the apple up for 5 cm
            self.right_move_to_pose_with_screw(pose2) # lift the apple up for 5 cm

    def check_success(self):
        apple_pose = self.apple.get_pose().p
        return apple_pose[2] > 0.81
