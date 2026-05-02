
import numpy as np
import torch
import yaml
import sys
from DP.diffusion_policy.common.pytorch_util import dict_apply

# eval_policy.py already does sys.path.append('./policy'); absolute imports work here
from DP.deploy_policy import get_model as get_dp_model
from DP3.deploy_policy import get_model as get_dp3_model

def _encode_obs_dp(observation):
    # Reuse DP encoding: head/left/right + agent_pos
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(head_cam=head_cam, left_cam=left_cam, right_cam=right_cam)
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs

def _encode_obs_dp3(observation):
    # Reuse DP3 encoding: point_cloud + agent_pos
    obs = dict()
    obs["agent_pos"] = observation["joint_action"]["vector"]
    obs["point_cloud"] = observation["pointcloud"]
    return obs

class MCDPModel:
    def __init__(self, usr_args):
        # 1) Load DP 
        self.dp = get_dp_model(dict(
            task_name=usr_args["task_name"],
            ckpt_setting=usr_args.get("dp_ckpt_setting"),
            expert_data_num=usr_args.get("dp_expert_data_num"),
            seed=usr_args.get("dp_seed"),
            checkpoint_num=usr_args.get("dp_checkpoint_num"),
            left_arm_dim=6,
            right_arm_dim=6,
        ))
        # 2) Load DP3
        self.dp3 = get_dp3_model(dict(
            task_name=usr_args["task_name"],
            ckpt_setting=usr_args.get("dp3_ckpt_setting"),
            expert_data_num=usr_args.get("dp3_expert_data_num"),
            seed=usr_args.get("dp3_seed"),
            checkpoint_num=usr_args.get("dp3_checkpoint_num"),
            config_name=usr_args.get("config_name", "robot_dp3"),
            use_rgb=usr_args.get("dp3_use_rgb", False),
        ))
        # Fusion weights
        self.dp_w = float(usr_args.get("dp_w", 0.5))
        self.dp3_w = float(usr_args.get("dp3_w", 0.5))

        # Align key hyperparams: n_action_steps / n_obs_steps / scheduler steps
        self.dp_n_action_steps = getattr(self.dp.policy, "n_action_steps", None)
        self.dp_n_obs_steps = getattr(self.dp.policy, "n_obs_steps", None)
        # DP3 policy uses the same naming as DP
        self.dp3_n_action_steps = getattr(self.dp3.policy, "n_action_steps")
        self.dp3_n_obs_steps = getattr(self.dp3.policy, "n_obs_steps")
        print("DP steps:", getattr(self.dp.policy, "n_obs_steps", None), getattr(self.dp.policy, "n_action_steps", None), getattr(self.dp.policy, "num_inference_steps", None))
        print("DP3 steps:", self.dp3.policy.n_obs_steps, self.dp3.policy.n_action_steps, getattr(self.dp3.policy, "num_inference_steps", None))
        assert self.dp_n_action_steps == self.dp3_n_action_steps, "DP and DP3 n_action_steps mismatch; cannot compose"
        assert self.dp_n_obs_steps == self.dp3_n_obs_steps, "DP and DP3 n_obs_steps mismatch; cannot compose"

        # Align inference steps
        self.dp_steps = getattr(self.dp.policy, "num_inference_steps", None)
        self.dp3_steps = getattr(self.dp3.policy, "num_inference_steps", None)
        assert self.dp_steps == self.dp3_steps, "DP and DP3 num_inference_steps mismatch; cannot compose"
        print(f"Loading MCDPModel with dp_w: {self.dp_w}, dp3_w: {self.dp3_w}")

    def reset_obs(self):
        self.dp.reset_obs()
        self.dp3.env_runner.reset_obs()

    def update_obs(self, observation):
        # Update both observation windows
        # obs_dp3 = _encode_obs_dp3(observation)
        # print("[DP3-update_obs] types:", type(obs_dp3["point_cloud"]), type(obs_dp3["agent_pos"]))
        # print("[DP3-update_obs] shapes:", obs_dp3["point_cloud"].shape, obs_dp3["agent_pos"].shape)
        self.dp.update_obs(_encode_obs_dp(observation))
        self.dp3.update_obs(_encode_obs_dp3(observation))

    # Build DP cond_data / cond_mask / global_cond / scheduler, etc.
    def _prepare_dp_infer(self):
        policy = self.dp.policy
        runner = self.dp.runner

        # Same construction as DPRunner.prepare_data
        obs = runner.get_n_steps_obs()
        device = policy.device
        np_obs_dict = dict(obs)
        obs_dict = {k: torch.from_numpy(v).to(device=device) for k, v in np_obs_dict.items()}

        with torch.no_grad():
            obs_dict_input = {}
            for k in ["head_cam", "front_cam", "left_cam", "right_cam", "agent_pos", "head_cam_depth"]:
                if k in obs_dict:
                    obs_dict_input[k] = obs_dict[k].unsqueeze(0)
            infer_data = policy.prepare_infer_data(obs_dict_input)

        return infer_data

    # Build DP3 cond_data / cond_mask / global_cond / scheduler, etc.
    def _prepare_dp3_infer(self):
        policy = self.dp3.policy
        runner = self.dp3.env_runner

        # Same construction as RobotRunner.prepare_data
        obs = runner.get_n_steps_obs()
        device = policy.device
        np_obs_dict = dict(obs)
        obs_dict = {}
        for k in ["point_cloud", "agent_pos"]:
            if k in np_obs_dict:
                obs_dict[k] = torch.from_numpy(np_obs_dict[k]).to(device=device)

        with torch.no_grad():
            obs_dict_input = {k: v.unsqueeze(0) for k, v in obs_dict.items()}
            infer_data = policy.prepare_infer_data(obs_dict_input)

        return infer_data
    
    def get_action(self, generator=None):
        # Prepare combined conditioning for both models
        data1 = self._prepare_dp_infer()
       # print("data1:", data1)
        data2 = self._prepare_dp3_infer()
        #print("data2:", data2)

        # Basic consistency checks
        assert data1["num_inference_steps"] == data2["num_inference_steps"]

        dp_model = data1["model"]; dp_sched = data1["scheduler"]
        dp3_model = data2["model"]; dp3_sched = data2["scheduler"]

        # Schedulers must match in type for synchronized stepping
        assert type(dp_sched).__name__ == type(dp3_sched).__name__, "DP/DP3 scheduler types differ"
        K = data1["num_inference_steps"]
        dp_sched.set_timesteps(K); dp3_sched.set_timesteps(K)

        # Composed sampling
        traj = torch.randn(
            size=data1["cond_data"].shape, 
            dtype=data1["cond_data"].dtype,
            device=data1["cond_data"].device,
            generator=generator)
        # printed = False

        

        for t in dp3_sched.timesteps:
            traj[data2["cond_mask"]] = data2["cond_data"][data2["cond_mask"]]
            out_dp  = dp_model(traj, t, local_cond=data1["local_cond"],  global_cond=data1["global_cond"])
            out_dp3 = dp3_model(traj, t, local_cond=data2["local_cond"], global_cond=data2["global_cond"])
            out = self.dp_w * out_dp + self.dp3_w * out_dp3
            traj = dp3_sched.step(out, t, traj).prev_sample
            # if not printed:
            #     print("out_dp norm:", out_dp.norm().item(), "out_dp3 norm:", out_dp3.norm().item())
            #     printed = True

        traj[data2["cond_mask"]] = data2["cond_data"][data2["cond_mask"]]
        naction_pred = traj[..., :data1["Da"]]
        # Unnormalize with DP action normalizer
        action_pred = data1["action_normalizer"].unnormalize(naction_pred)

        # Slice n_action_steps of actions
        To = data1["To"]
        n_action_steps = data1["n_action_steps"]
        start = To - 1
        end = start + n_action_steps
        action = action_pred[:, start:end]  # [1, n_action_steps, Da]
        result = {
            'action': action,
            'action_pred': action_pred
        }

        # device_transfer
        np_action_dict = dict_apply(result, lambda x: x.detach().to('cpu').numpy())
        action = np_action_dict['action'].squeeze(0)

        return action

def get_model(usr_args):
    # Return composed model
    return MCDPModel(usr_args)

def eval(TASK_ENV, model, observation):
    # Same as DP/DP3 eval: fill window, chunk actions, step env and refresh window
    assert model.dp_n_action_steps == model.dp3_n_action_steps, "DP and DP3 n_action_steps mismatch; cannot compose"

    model.update_obs(observation)
    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action)
        obs = TASK_ENV.get_obs()
        model.update_obs(obs)
        
def eval_vis(TASK_ENV, model, observation):
    # Same as DP/DP3 eval: fill window, chunk actions, step env and refresh window
    assert model.dp_n_action_steps == model.dp3_n_action_steps, "DP and DP3 n_action_steps mismatch; cannot compose"
    model.update_obs(observation)
    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action)
        obs = TASK_ENV.get_obs()
        model.update_obs(obs)
    return actions

def reset_model(model):
    model.reset_obs()