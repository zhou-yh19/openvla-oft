"""
run_teleavatar_eval.py

Evaluates a trained policy on a Teleavatar.

```bash
python experiments/robot/teleavatar/run_teleavatar_eval.py   --pretrained_checkpoint outputs/Teleavatar-stuffed-animal   > eval_logs/shihaoran--teleavatar--stuffed_animal--chkpt.log 2>&1 &
```
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm 
import time
import torch
from PIL import Image

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_model,
    set_seed_everywhere,
    get_image_resize_size,
)
from prismatic.vla.constants import TELEAVATAR_CONSTANTS
PROPRIO_DIM = TELEAVATAR_CONSTANTS["PROPRIO_DIM"]

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Import Robot interface
from experiments.robot.teleavatar.robot_interface import TeleavatarRobotInterface


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = "/home/lingyu/VLA-project/openvla-oft/ckpt/openvla7b+teleavatar_teleop_53_demos_30000"     # Pretrained checkpoint path
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses diffusion-based action generation (instead of autoregressive decoding)
    
    use_film: bool = True                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 3                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_action_horizon: int = 30                     # Number of actions in each chunk returned by policy
    num_open_loop_steps: int = 10                    # Number of actions to execute before querying policy again
    unnorm_key: Union[str, Path] = ""                # Action un-normalization key
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = True                       # (For OpenVLA only) Load with 4-bit quantization
    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)
    
    #################################################################################################################
    # Teleavatar runtime parameters
    #################################################################################################################
    control_frequency: float = 30                    # Control loop frequency in Hz               
    task_description: str = "tidy_up_blocks" 
                                                     # Language instruction for the robot
    num_episodes: int = 10                           # Number of episodes to run
    max_episode_steps: int = 800                     # Maximum VLA inference count per episode (0 = unlimited)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    seed: int = 7                                    # Random Seed (for reproducibility)



def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"


def initialize_model(cfg: GenerateConfig):
    """Initialize model and associated components."""
    # Load model
    model = get_model(cfg)
    # Load proprio projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        proprio_projector = get_proprio_projector(
            cfg,
            model.llm_dim,
            proprio_dim=PROPRIO_DIM,  # 14-dimensional proprio for Teleavatar
        )

    # Load action head if needed
    action_head = None
    if cfg.use_l1_regression:
        action_head = get_action_head(cfg, model.llm_dim)

    # Load noisy action projector if using diffusion
    noisy_action_projector = None

    # Get OpenVLA processor if needed
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
        check_unnorm_key(cfg, model)

    return model, action_head, proprio_projector, noisy_action_projector, processor


def check_unnorm_key(cfg: GenerateConfig, model) -> None:
    """Check that the model contains the action un-normalization key."""
    # Set the unnorm_key in cfg
    cfg.unnorm_key = list(model.norm_stats.keys())[0]



def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-Teleavatar-Grab_stuffed_animal-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file



def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def check_model_modules_on_cuda(model, log_file=None, max_report: int = 200):
    """
    Check whether every module tensor (params/buffers) is on CUDA.

    Returns:
        Tuple[bool, list]: (all_on_cuda, non_cuda_module_records)
    """
    total_modules = 0
    modules_with_tensors = 0
    non_cuda_modules = []

    for module_name, module in model.named_modules():
        total_modules += 1

        direct_tensors = []
        direct_tensors.extend([p for _, p in module.named_parameters(recurse=False)])
        direct_tensors.extend([b for _, b in module.named_buffers(recurse=False)])

        if not direct_tensors:
            continue
        modules_with_tensors += 1

        devices = sorted({str(t.device) for t in direct_tensors})
        all_cuda = all(t.device.type == "cuda" for t in direct_tensors)
        if not all_cuda:
            non_cuda_modules.append((module_name or "<root>", devices))

    all_on_cuda = len(non_cuda_modules) == 0
    log_message(
        f"[DeviceCheck] modules={total_modules}, modules_with_tensors={modules_with_tensors}, all_on_cuda={all_on_cuda}",
        log_file,
    )

    if not all_on_cuda:
        log_message(f"[DeviceCheck] non_cuda_modules={len(non_cuda_modules)}", log_file)
        for name, devices in non_cuda_modules[:max_report]:
            log_message(f"[DeviceCheck] {name}: devices={devices}", log_file)
        if len(non_cuda_modules) > max_report:
            log_message(
                f"[DeviceCheck] ... truncated {len(non_cuda_modules) - max_report} more modules",
                log_file,
            )

    return all_on_cuda, non_cuda_modules



def _get_teleavatar_chest_image(obs):
    """Get chest raw-image from ros2 interface."""
    return obs['images']['head_camera']

def _get_teleavatar_left_wrist_image(obs):
    """Get left-wrist raw-image from ros2 interface."""
    return obs['images']['left_color']

def _get_teleavatar_right_wrist_image(obs):
    """Get right-wrist raw-image from ros2 interface."""
    return obs['images']['right_color']

def _get_teleavatar_state(obs):
    """Get proprio from ros2 interface and normalize"""
    return obs['state']

def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get raw images
    chest_img = _get_teleavatar_chest_image(obs)
    _, w, _ = chest_img.shape
    if w >= 2:
        half_w = w // 2
        chest_img = chest_img[:, :half_w, :]
    left_wrist_img = _get_teleavatar_left_wrist_image(obs)
    right_wrist_img = _get_teleavatar_right_wrist_image(obs)

    chest_img = resize_image_for_policy(chest_img, resize_size)
    left_wrist_img = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img = resize_image_for_policy(right_wrist_img, resize_size)

    # Get state
    state = _get_teleavatar_state(obs)

    # Prepare observations dict
    observation = {
        "full_image": chest_img,
        "left_wrist_image": left_wrist_img,
        "right_wrist_image": right_wrist_img,
        "state": state,
    }
    return observation  


def _save_observation_images_tmp(observation: dict, episode_idx: int, step_idx: int, log_file=None):
    """Temporarily save 3 camera images for quick inspection."""
    save_dir = "tmp/teleavatar_eval_images"
    os.makedirs(save_dir, exist_ok=True)

    name_to_image = {
        "chest": observation["full_image"],
        "left_wrist": observation["left_wrist_image"],
        "right_wrist": observation["right_wrist_image"],
    }

    saved_paths = []
    for name, image in name_to_image.items():
        filename = f"ep{episode_idx:03d}_step{step_idx:04d}_{name}.png"
        save_path = os.path.join(save_dir, filename)
        Image.fromarray(image.astype(np.uint8)).convert("RGB").save(save_path)
        saved_paths.append(save_path)

    log_message(f"[ImageDump] Saved observation images: {saved_paths}", log_file)



def run_episode(
    cfg: GenerateConfig,
    task_description: str,
    robot_interface: TeleavatarRobotInterface,
    model,
    episode_idx,
    resize_size,
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    inference_time_ls = list()

    # Run episode
    try:
        t = 0
        action_publish_interval = 1.0 / cfg.control_frequency
        start_time = None

        # VLA inference count is cfg.max_episode_steps
        while t < cfg.max_episode_steps:
            
            if len(action_queue) == 0:
                inference_start_time = time.time()
                
                # If action queue is empty, requery model
                obs = robot_interface.get_observation()
                observation = prepare_observation(obs, resize_size)
                # if t == 0:
                #     _save_observation_images_tmp(
                #         observation=observation,
                #         episode_idx=episode_idx,
                #         step_idx=t,
                #         log_file=log_file,
                #     )
                #print(observation["full_image"].device)
                # Query model to get action
                actions = get_action(
                    cfg,
                    model,
                    observation,
                    task_description,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    use_film=cfg.use_film,
                )
                inference_end_time = time.time()
                inference_time_ls.append(inference_end_time - inference_start_time)

                # Add actions to queue
                action_queue.extend(actions)
                t += 1

            # Get action from queue
            action = action_queue.popleft()

            # publish action to Teleavatar via ROS2 interface
            if start_time is not None:
                stop_time = time.time()
                interval = stop_time - start_time
                if interval < action_publish_interval:
                    time.sleep(action_publish_interval - interval)
            print(f"Episode {episode_idx}, Step {t}, Action: {action}")
            robot_interface.apply_action(action)
            start_time = time.time()

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return inference_time_ls


def run_eval_runtime(
    cfg: GenerateConfig,
    robot_interface: TeleavatarRobotInterface,
    model,
    resize_size,    
    processor=None,
    action_head=None,
    proprio_projector=None,
    noisy_action_projector=None,
    log_file=None,
):
    """Run teleavatar runtime for multi episodes."""
    inference_time_ls = list()
    
    # Start Episodes
    for episode_idx in tqdm.tqdm(range(cfg.num_episodes)):
        log_message(f"Episode: {episode_idx}", log_file)

        # Run episode
        episode_inference_time_ls = run_episode(
            cfg,
            cfg.task_description.replace("_", " "),
            robot_interface,
            model,
            episode_idx,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            log_file,
        )

        inference_time_ls.extend(episode_inference_time_ls)

        log_message(f"Episode: {episode_idx} has finished!!", log_file)
    
    return inference_time_ls



@draccus.wrap()
def eval_teleavatar(cfg: GenerateConfig):
    """Main function to evaluate a trained policy on Teleavatar."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup logging
    log_file = setup_logging(cfg)
    resize_size = get_image_resize_size(cfg)
    log_message(f"Using image resize size: {resize_size}", log_file)
    # Initialize model and components
    log_message("Initializing Finetuned VLA-Adapter...", log_file)
    start_time = time.time()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    initialize_model_period = time.time() - start_time
    log_message(f"Initialize model period: {initialize_model_period:.2f} seconds", log_file)
    log_message(f"model is on {model.device}", log_file)
    all_on_cuda, _ = check_model_modules_on_cuda(model, log_file=log_file)
    if not all_on_cuda:
        log_message(
            "[DeviceCheck] Found model modules not on CUDA. "
            "If you enabled 4bit/8bit with device_map, partial non-CUDA placement can be expected.",
            log_file,
        )

    # Robot interface can initialize ros2_interface
    log_message("Initializing Teleavatar Robot Interface...", log_file)
    robot_interface = TeleavatarRobotInterface()

    # Start evaluation
    log_message("Starting Evaluation...", log_file)
    inference_time_ls = run_eval_runtime(
        cfg,
        robot_interface,
        model,
        resize_size,
        processor,
        action_head,
        proprio_projector,
        noisy_action_projector,
        log_file
    )

    # Log final results
    log_message(f"Total episodes: {cfg.num_episodes}", log_file)
    log_message(f"Inference time list: {inference_time_ls}", log_file)
    log_message(f"Max inference time: {max(inference_time_ls):.3f} seconds", log_file)
    log_message(f"Average inference time: {sum(inference_time_ls) / len(inference_time_ls):.3f} seconds", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return


if __name__ == "__main__":
    eval_teleavatar()
