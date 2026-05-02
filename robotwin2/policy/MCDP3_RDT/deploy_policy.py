
import numpy as np
import torch
import yaml
import sys
from DP.diffusion_policy.common.pytorch_util import dict_apply

# eval_policy.py already does sys.path.append('./policy'); absolute imports work here
from DP3.deploy_policy import get_model as get_dp3_model
from RDT.deploy_policy import get_model as get_rdt_model
from PIL import Image


def _encode_obs_dp3(observation):
    # Reuse DP3 encoding: point_cloud + agent_pos
    obs = dict()
    obs["agent_pos"] = observation["joint_action"]["vector"]
    obs["point_cloud"] = observation["pointcloud"]
    return obs

def _encode_obs_rdt(observation):
    # Reuse RDT encoding: RGB images + agent_pos
    observation["agent_pos"] = observation["joint_action"]["vector"]
    return observation



class MCDP3_RDT_Model:
    def __init__(self, usr_args):  # DP-style bundle: policy + runner (n_obs / n_action)
        # 2) Load DP3
        self.dp3 = get_dp3_model(dict(
            task_name=usr_args["task_name"],
            ckpt_setting=usr_args.get("dp3_ckpt_setting"),
            expert_data_num=usr_args.get("dp3_expert_data_num"),
            seed=usr_args.get("dp3_seed"),
            checkpoint_num=usr_args.get("dp3_checkpoint_num"),
            config_name=usr_args.get("config_name", "robot_dp3"),
            use_rgb=usr_args.get("use_rgb", False),
        ))
        # 3) Load RDT
        self.rdt = get_rdt_model(dict(
            task_name=usr_args["task_name"],
            ckpt_setting=usr_args.get("rdt_ckpt_setting"),
            checkpoint_id=usr_args.get("rdt_checkpoint_num"),  # explicit checkpoint id key
            left_arm_dim=usr_args["left_arm_dim"],
            right_arm_dim=usr_args["right_arm_dim"],
            rdt_step=usr_args.get("rdt_step", 32),
        ))
        # Fusion weights
        self.dp3_w = float(usr_args.get("dp3_w", 0.5))
        self.rdt_w = float(usr_args.get("rdt_w", 0.5))

               # Align key hyperparams: n_action_steps / n_obs_steps / scheduler steps
        # Read from DP3
        self.dp3_n_action_steps = getattr(self.dp3.policy, "n_action_steps")
        self.dp3_n_obs_steps = getattr(self.dp3.policy, "n_obs_steps")
        
        # Read from RDT (may lack some attrs; validate)
        self.rdt_n_action_steps = getattr(self.rdt, "rdt_step", None)
        self.rdt_n_obs_steps = 2
        print("DP3 steps:", self.dp3_n_obs_steps, self.dp3_n_action_steps, getattr(self.dp3.policy, "num_inference_steps", None))
        print("RDT steps:", self.rdt_n_obs_steps, self.rdt_n_action_steps, getattr(self.rdt, "rdt_step", None))

        # Align inference steps
        self.dp3_steps = getattr(self.dp3.policy, "num_inference_steps", None)

        print(f"Loading MCDP3_RDTModel with dp3_w: {self.dp3_w}, rdt_w: {self.rdt_w}")
        print(f"[COMPOSE] dp3.n_action_steps={self.dp3.policy.n_action_steps}, rdt.rdt_step={self.rdt.rdt_step}, "
              f"dp3.num_inference_steps={self.dp3.policy.num_inference_steps}, rdt.num_inference_steps={self.rdt.policy.policy.num_inference_timesteps}")
        assert self.dp3.policy.n_action_steps == self.rdt.rdt_step, \
            f"Action steps mismatch: DP3={self.dp3.policy.n_action_steps} vs RDT={self.rdt.rdt_step}"

    def reset_obs(self):
        self.dp3.env_runner.reset_obs()
        self.rdt.reset_obsrvationwindows()

    def update_obs(self, observation):
        # Update all observation windows
        obs = _encode_obs_rdt(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        # obs_dp3 = _encode_obs_dp3(observation)
        # print("[DP3-update_obs] types:", type(obs_dp3["point_cloud"]), type(obs_dp3["agent_pos"]))
        # # print("[DP3-obs] obs_dp3:", obs_dp3)
        self.dp3.update_obs(_encode_obs_dp3(observation))
        self.rdt.update_observation_window(input_rgb_arr, input_state)


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
    
    def _prepare_rdt_infer(self):
        # wrapper: RoboticDiffusionTransformerModel
        wrapper = self.rdt.policy
        # runner: RDTRunner (noise_scheduler_sample / model / adaptors)
        runner = wrapper.policy

        device = next(runner.parameters()).device
        dtype = next(runner.parameters()).dtype

        # 1) Last two frames from RDT observation_window -> img_tokens
        ow = self.rdt.observation_window
        assert ow is not None and len(ow) >= 2, "RDT needs at least two observation frames"
        cam_names = self.rdt.config["camera_names"]  # ["cam_high", "cam_right_wrist", "cam_left_wrist"]

        imgs = [
            ow[-2]["images"][cam_names[0]],
            ow[-2]["images"][cam_names[1]],
            ow[-2]["images"][cam_names[2]],
            ow[-1]["images"][cam_names[0]],
            ow[-1]["images"][cam_names[1]],
            ow[-1]["images"][cam_names[2]],
        ]

        # Reuse agilex_model preprocessing
        image_tensor_list = []
        background_color = np.array([int(x * 255) for x in wrapper.image_processor.image_mean], dtype=np.uint8).reshape(1, 1, 3)
        background_image = (np.ones(
            (
                wrapper.image_processor.size["height"],
                wrapper.image_processor.size["width"],
                3,
            ),
            dtype=np.uint8,
        ) * background_color)
        for image in imgs:
            if image is None:
                image = Image.fromarray(background_image)
            elif isinstance(image, np.ndarray):
                image = np.clip(image, 0, 255).astype(np.uint8)
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(image)
            # pad to square
            if wrapper.args["dataset"].get("image_aspect_ratio", "pad") == "pad":
                def expand2square(pil_img, bg):
                    w, h = pil_img.size
                    if w == h:
                        return pil_img
                    if w > h:
                        result = Image.new(pil_img.mode, (w, w), tuple(int(x * 255) for x in wrapper.image_processor.image_mean))
                        result.paste(pil_img, (0, (w - h) // 2))
                        return result
                    result = Image.new(pil_img.mode, (h, h), tuple(int(x * 255) for x in wrapper.image_processor.image_mean))
                    result.paste(pil_img, ((h - w) // 2, 0))
                    return result
                image = expand2square(image, wrapper.image_processor.image_mean)
            image = wrapper.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        img_tokens = wrapper.vision_model(image_tensor).detach()
        img_tokens = img_tokens.reshape(-1, wrapper.vision_model.hidden_size).unsqueeze(0)  # [1, img_len, C]

        # 2) Language tokens (set via set_language_instruction at episode start)
        lang_tokens = self.rdt.lang_embeddings.to(device, dtype=dtype)  # [1, L, C]
        lang_attn_mask = torch.ones(lang_tokens.shape[:2], dtype=torch.bool, device=lang_tokens.device)

        # 3) state + action_mask (last-frame qpos -> unified state vector)
        proprio = ow[-1]["qpos"].unsqueeze(0).unsqueeze(0).to(device)  # shape [1, 1, Da_joint]
        states, state_elem_mask = wrapper._format_joint_to_state(proprio)  # [1,1,S], [1,S]
        states = states.to(device, dtype=dtype)[:, -1:, :]  # [1,1,S]
        action_mask_unified = state_elem_mask.to(device, dtype=dtype).unsqueeze(1)  # [1,1,S]
        ctrl_freqs = torch.tensor([wrapper.control_frequency], device=device)

        # 4) Pre-adapt static conditioning (t fixed)
        state_tokens = torch.cat([states, action_mask_unified], dim=2)  # [1,1,S+S]
        lang_cond, img_cond, state_traj = runner.adapt_conditions(
            lang_tokens=lang_tokens, img_tokens=img_tokens, state_tokens=state_tokens
        )  # -> all mapped to hidden_size

        # 5) Joint index mapping (RDT unified 128-D -> joint order) via STATE_VEC_IDX_MAPPING
        from RDT.scripts.agilex_model import STATE_VEC_IDX_MAPPING as SVM
        left_arm_dim, right_arm_dim = wrapper.left_arm_dim, wrapper.right_arm_dim
        AGILEX_STATE_INDICES = (
            [SVM[f"left_arm_joint_{i}_pos"] for i in range(left_arm_dim)]
            + [SVM["left_gripper_open"]]
            + [SVM[f"right_arm_joint_{i}_pos"] for i in range(right_arm_dim)]
            + [SVM["right_gripper_open"]]
        )
        indices = torch.as_tensor(AGILEX_STATE_INDICES, device=device)

        # 6) Per-timestep eps: embed joint traj in unified space, RDT predicts, map back
        def rdt_eps_fn(traj, t, local_cond, global_cond):
            # traj: [B, H, Da_joint] (same layout as DP3 traj)
            B, H, Da = traj.shape
            S = action_mask_unified.shape[-1]  # unified dim
            # Embed in unified state space (other dims zero)
            traj_unified = torch.zeros((B, H, S), device=device, dtype=dtype)
            traj_unified.index_copy_(2, indices, traj.to(device, dtype=dtype))
            # Expand mask to H
            am = action_mask_unified.expand(B, H, S)
            # Concat and adapt as in training
            action_traj = torch.cat([traj_unified, am], dim=2)           # [B,H,S+S]
            action_traj = runner.state_adaptor(action_traj)              # [B,H,hidden]
            state_action_traj = torch.cat([state_traj, action_traj], 1)  # [B,1+H,hidden]
            # Base diffusion output (training uses prediction_type=sample)
            pred_unified = runner.model(
                state_action_traj,
                global_cond["ctrl_freqs"],
                t.unsqueeze(-1).to(device),
                global_cond["lang_cond"],
                global_cond["img_cond"],
                lang_mask=global_cond["lang_attn_mask"],
            )  # [B,H,S]
            # Map back to joint space
            pred_joint = torch.index_select(pred_unified, dim=2, index=indices)  # [B,H,Da_joint]
            return pred_joint

        # 7) Pack infer_data
        infer_data = {
            "model": rdt_eps_fn,
            "scheduler": runner.noise_scheduler_sample,
            "num_inference_steps": runner.num_inference_timesteps,
            "cond_data": None,
            "cond_mask": None,
            "local_cond": {
                "state_traj": state_traj,
                "action_mask": action_mask_unified,
            },
            "global_cond": {
                "lang_cond": lang_cond,
                "img_cond": img_cond,
                "lang_attn_mask": lang_attn_mask,
                "ctrl_freqs": ctrl_freqs,
            },
            "Da": None,
            "To": 1,
            "n_action_steps": getattr(self, "dp3_n_action_steps", None),
        }
        return infer_data
    
    
    def get_action(self, generator=None):
        # Prepare combined conditioning for both models
        data1 = self._prepare_dp3_infer()
        data2 = self._prepare_rdt_infer()

        # Basic consistency checks
        assert data1["num_inference_steps"] == data2["num_inference_steps"]

        dp3_model = data1["model"]; dp3_sched = data1["scheduler"]
        rdt_model = data2["model"]; rdt_sched = data2["scheduler"]

        # Schedulers must match for synchronized stepping
       # assert type(dp3_sched).__name__ == type(rdt_sched).__name__, "DP3/RDT scheduler types differ"
        # print("dp3_sched:", dp3_sched, "rdt_sched:", rdt_sched)
        K = data1["num_inference_steps"]
        dp3_sched.set_timesteps(K); rdt_sched.set_timesteps(K)

        # Schedulers must be identical (time grid / type / prediction) so t is interchangeable
        assert type(dp3_sched).__name__ == type(rdt_sched).__name__, "DP3/RDT scheduler types differ"
        assert dp3_sched.config.prediction_type == rdt_sched.config.prediction_type == "sample", "prediction_type must match and be sample"
        assert dp3_sched.config.num_train_timesteps == rdt_sched.config.num_train_timesteps, "num_train_timesteps mismatch"
        try:
            import torch as _torch
            assert _torch.equal(dp3_sched.timesteps, rdt_sched.timesteps), "timesteps grid mismatch"
        except Exception:
            pass

        # Sample in RDT real joint space; traj_real shape [B, T, Da_joint]
        B, T, Da = data1["cond_data"].shape
        Da_joint = data1["Da"]
        dp3_norm = data1["action_normalizer"]
        traj_real = torch.randn(
            size=(B, T, Da_joint),
            dtype=data1["cond_data"].dtype,
            device=data1["cond_data"].device,
            generator=generator,
        )

        for t in rdt_sched.timesteps:
            # --- DP/DP3 forward: real -> normalized space (pad to Da, then normalize)
            traj_full_real = torch.zeros((B, T, Da), device=traj_real.device, dtype=traj_real.dtype)
            traj_full_real[..., :Da_joint] = traj_real
            traj_full_dp3 = dp3_norm.normalize(traj_full_real)

            # Conditioning applies only in the normalized copy
            traj_full_dp3[data1["cond_mask"]] = data1["cond_data"][data1["cond_mask"]]
            out_dp3_dp3 = dp3_model(traj_full_dp3, t, local_cond=data1["local_cond"], global_cond=data1["global_cond"])

            # Unnormalize DP3 output to real joint space (first Da_joint dims)
            out_dp3_full_real = dp3_norm.unnormalize(out_dp3_dp3)
            out_dp3_joint_real = out_dp3_full_real[..., :Da_joint]

            # --- RDT forward: real joint space in/out
            out_rdt_joint_real = rdt_model(traj_real, t, local_cond=data2["local_cond"], global_cond=data2["global_cond"])

            # --- Weighted blend; step with RDT scheduler
            out_real = self.dp3_w * out_dp3_joint_real + self.rdt_w * out_rdt_joint_real
            traj_real = rdt_sched.step(out_real, t, traj_real).prev_sample

        # traj_real is in real joint space; pad to Da for consistent slicing
        traj_full_real = torch.zeros((B, T, Da), device=traj_real.device, dtype=traj_real.dtype)
        traj_full_real[..., :Da_joint] = traj_real
        action_pred = traj_full_real  # real units

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
        # K = data1["num_inference_steps"]
        # dp3_sched.set_timesteps(K); rdt_sched.set_timesteps(K)

        # # Composed sampling
        # traj = torch.randn(
        #     size=data1["cond_data"].shape, 
        #     dtype=data1["cond_data"].dtype,
        #     device=data1["cond_data"].device,
        #     generator=generator)
        # # printed = False

        

        # for t in dp3_sched.timesteps:
        #     traj[data1["cond_mask"]] = data1["cond_data"][data1["cond_mask"]]
        #     out_dp3 = dp3_model(traj, t, local_cond=data1["local_cond"], global_cond=data1["global_cond"])
        #     out_rdt = rdt_model(traj, t, local_cond=data2["local_cond"], global_cond=data2["global_cond"])
        #     out = self.dp3_w * out_dp3 + self.rdt_w * out_rdt
        #     traj = dp3_sched.step(out, t, traj).prev_sample
        #     if True:
        #         print("out_dp3 norm:", out_dp3.norm().item(), "out_rdt norm:", out_rdt.norm().item())
        #         print("rdt_w:", self.rdt_w, "dp3_w:", self.dp3_w)
            

        # traj[data1["cond_mask"]] = data1["cond_data"][data1["cond_mask"]]
        # naction_pred = traj[..., :data1["Da"]]
        # # Unnormalize with DP action normalizer
        # action_pred = data1["action_normalizer"].unnormalize(naction_pred)

        # # Slice n_action_steps
        # To = data1["To"]
        # n_action_steps = data1["n_action_steps"]
        # start = To - 1
        # end = start + n_action_steps
        # action = action_pred[:, start:end]  # [1, n_action_steps, Da]
        # result = {
        #     'action': action,
        #     'action_pred': action_pred
        # }

        # # device_transfer
        # np_action_dict = dict_apply(result, lambda x: x.detach().to('cpu').numpy())
        # action = np_action_dict['action'].squeeze(0)

        # return action
    
    def set_language(self, instruction):
        self.rdt.set_language_instruction(instruction)

def get_model(usr_args):
    # Return composed model
    return MCDP3_RDT_Model(usr_args)

def eval(TASK_ENV, model, observation):
    if getattr(model, "lang_embeddings", None) is None:
        model.set_language(TASK_ENV.get_instruction())

    model.update_obs(observation)
    actions = model.get_action()

    for action in actions:
        TASK_ENV.take_action(action)
        obs = TASK_ENV.get_obs()
        model.update_obs(obs)


def eval_vis(TASK_ENV, model, observation):

    if getattr(model, "lang_embeddings", None) is None:
        model.set_language(TASK_ENV.get_instruction())
  
    model.update_obs(observation)
    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action)
        obs = TASK_ENV.get_obs()
        model.update_obs(obs)
    return actions

def reset_model(model):
    model.reset_obs()