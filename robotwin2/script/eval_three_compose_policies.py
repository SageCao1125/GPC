import sys, os, subprocess, yaml, argparse
import numpy as np
from datetime import datetime
import importlib
from pathlib import Path
import torch
import traceback

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
from generate_episode_instructions import generate_episode_descriptions
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

from policy.MCDP_DP3_RDT.deploy_policy import get_model as get_mcdp_dp3_rdt_model
from policy.MCDP_DP3_RDT.deploy_policy import eval as eval_composed

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)



def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e
    
def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        d = {}
        if not pairs:
            return d
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            val = pairs[i + 1]
            try:
                val = eval(val)
            except:
                pass
            d[key] = val
        return d

    if args.overrides:
        config.update(parse_override_pairs(args.overrides))
    return config


def main(cfg):
    policy_name = "MCDP_DP3_RDT"

    rdt_w = cfg.get("rdt_w", 0.34)
    dp3_w = cfg.get("dp3_w", 0.33)
    dp_w = cfg.get("dp_w", 0.33)
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = cfg["task_name"]
    task_config = cfg["task_config"]
    
    instruction_type = cfg["instruction_type"]
    print(f"task_name: {task_name}, task_config: {task_config}")
    print(f"Loading Compositional Policy (DP + DP3 + RDT)")
    save_dir = Path(f"eval_result/{task_name}/MCDP_DP3_RDT/{task_config}/dp-{dp_w}_dp3-{dp3_w}_rdt-{rdt_w}_{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load task_config and embodiment

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = task_name


    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])


    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

        # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")


    # Instantiate environment
    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = "MCDP_DP3_RDT"
    args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    # Args for three-way composed model (cfg / deploy_policy.yml)
    args["task_name"] = args.get("task_name") or cfg["task_name"]
    args["ckpt_setting"] = args.get("ckpt_setting") or cfg.get("ckpt_setting", "rgb_only")
    args["expert_data_num"] = args.get("expert_data_num") or cfg.get("expert_data_num", 50)

    # DP (from cfg, defaults below)
    args["dp_ckpt_setting"]    = cfg.get("dp_ckpt_setting")
    args["dp_expert_data_num"] = cfg.get("dp_expert_data_num", 50)
    args["dp_checkpoint_num"]  = cfg.get("dp_checkpoint_num", 600)
    args["dp_seed"]            = cfg.get("dp_seed", 0)

    # DP3 (from cfg, defaults below)
    args["dp3_ckpt_setting"]    = cfg.get("dp3_ckpt_setting")
    args["dp3_expert_data_num"] = cfg.get("dp3_expert_data_num", 50)
    args["dp3_checkpoint_num"]  = cfg.get("dp3_checkpoint_num", 3000)
    args["dp3_seed"]            = cfg.get("dp3_seed", 0)
    args["dp3_use_rgb"]         = cfg.get("dp3_use_rgb", False)

    # RDT (from cfg, defaults below)
    args["rdt_ckpt_setting"]    = f"rdt_{args['task_name']}"
    args["rdt_expert_data_num"] = cfg.get("rdt_expert_data_num", 50)
    args["rdt_checkpoint_num"]  = cfg.get("rdt_checkpoint_num", 5000)
    args["rdt_seed"]            = cfg.get("rdt_seed", 0)
    args["rdt_step"]            = cfg.get("rdt_step", 32)

    # Fusion weights (CLI overrides only)
    if "rdt_w" not in cfg or "dp3_w" not in cfg or "dp_w" not in cfg:
        raise KeyError("Missing --dp_w / --dp3_w / --rdt_w in CLI overrides")
    args["dp_w"] = float(cfg["dp_w"])
    args["dp3_w"] = float(cfg["dp3_w"])
    args["rdt_w"] = float(cfg["rdt_w"])
    print(f"dp_w: {args['dp_w']}, dp3_w: {args['dp3_w']}, rdt_w: {args['rdt_w']}")

    args["config_name"] = cfg.get("config_name", "robot_dp3")
    args["dp3_use_rgb"] = cfg.get("dp3_use_rgb", False)

    model = get_mcdp_dp3_rdt_model(args)


    # Eval loop (same style as eval_policy.py)
    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0
    seed_base = 0
    st_seed = 100000 * (1 + seed_base)

    suc_nums = []
    test_num = 100  # override via cfg if needed
    topk = 1

    st_seed, suc_num = eval_composed_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
    )
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"dp_w: {model.dp_w}\n\n")
        file.write(f"dp3_w: {model.dp3_w}\n\n")
        file.write(f"rdt_w: {model.rdt_w}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")

def eval_composed_policy(task_name,
                         TASK_ENV,
                         args,
                         model,
                         st_seed,
                         test_num=100,
                         video_size=None,
                         instruction_type=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print('-------------')
                print(traceback.format_exc())
                print('-------------')
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)

        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        reset_func(model)
     
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            # TODO: compose noise
            eval_composed(TASK_ENV, model, observation)

            if TASK_ENV.eval_success:
                succ = True
                break
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")
            
        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()
        TASK_ENV.test_num += 1
        print(f"{task_name} | MCDP_DP3_RDT |\n"
              f"Success rate: {TASK_ENV.suc}/{TASK_ENV.test_num} => {round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%, seed: {now_seed}")
        now_seed += 1
    
    return now_seed, TASK_ENV.suc

 


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    cfg = parse_args_and_config()
    main(cfg)