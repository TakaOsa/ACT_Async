"""
run_aloha_eval.py

Evaluates a model in a real-world ALOHA environment.
"""

import logging
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import tqdm
import numpy as np
import copy
import threading
import queue

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.aloha.aloha_utils import (
    get_aloha_env,
    get_aloha_image,
    get_aloha_wrist_images,
    get_next_task_label,
    save_rollout_video,
)
from experiments.robot.openvla_utils import (
    get_action_from_server,
    resize_image_for_policy,
)
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    set_seed_everywhere,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 25                    # Number of actions to execute open-loop before requerying policy

    use_vla_server: bool = True                      # Whether to query remote VLA server for actions
    vla_server_url: Union[str, Path] = ""            # Remote VLA server URL (set to 127.0.0.1 if on same machine)

    #################################################################################################################
    # ALOHA environment-specific parameters
    #################################################################################################################
    num_rollouts_planned: int = 50                   # Number of test rollouts
    max_steps: int = 1500                            # Max number of steps per rollout
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    cac: bool = True 

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.use_vla_server, (
        "Must use VLA server (server-client interface) to query model and get actions! Please set --use_vla_server=True"
    )


def setup_logging(cfg: GenerateConfig):
    """Set up logging to file."""
    # Create run ID
    run_id = f"EVAL-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    print(message)
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def get_server_endpoint(cfg: GenerateConfig):
    """Get the server endpoint for remote inference."""
    ip_address = socket.gethostbyname(cfg.vla_server_url)
    return f"http://{ip_address}:8777/act"


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_aloha_image(obs)
    left_wrist_img, right_wrist_img = get_aloha_wrist_images(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    left_wrist_img_resized = resize_image_for_policy(left_wrist_img, resize_size)
    right_wrist_img_resized = resize_image_for_policy(right_wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "left_wrist_image": left_wrist_img_resized,
        "right_wrist_image": right_wrist_img_resized,
        "state": obs.observation["qpos"],
    }

    return observation, img_resized, left_wrist_img_resized, right_wrist_img_resized

max_action = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2])
min_action = np.array([-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 0.0, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 0.0])

max_dev = np.array([0.025, 0.025, 0.025, 0.025, 0.015, 0.015, 0.2, 0.025, 0.025, 0.025, 0.025, 0.015, 0.015, 0.2])
max_acc = np.array([0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.05, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.05])


class OverwriteQueue(queue.Queue):
    def __init__(self):
        super().__init__(maxsize=1)

    def put(self, item, block=True, timeout=None):
        while not self.empty():
            try:
                self.get_nowait() 
            except queue.Empty:
                break
        super().put(item, block, timeout)
    
    def clear_que(self):
        while not self.empty():
            try:
                self.get_nowait() 
            except queue.Empty:
                break


obs_que = OverwriteQueue()
command_que = OverwriteQueue()
stop_event = threading.Event()

def inference_query_thread(server_endpoint):
    time_step = 0
    inf_cnt = 0
    while not stop_event.is_set():
        time.sleep(0.01) # default 0.01
        if not obs_que.empty():
            # start_flag = True
            time_step, observation= obs_que.get()
            actions = get_action_from_server(observation, server_endpoint)
            print('inference thread', time_step)
            command_que.put([time_step, actions])
            inf_cnt += 1

        if time_step > 1199: #Set the maximum time step
            break
    print('end of inference thread.')


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    server_endpoint: str,
    resize_size,
    log_file=None,
    cac=False,
    temopral_agg = True
):
    """Run a single episode in the ALOHA environment."""
    # Define control frequency
    STEP_DURATION_IN_SEC = 1.0 / 25.0
    query_frequency = 8

    # Reset environment
    obs = env.reset()

    # Initialize action queue
    action_queue = deque(maxlen=cfg.num_open_loop_steps)

    # Setup
    t = 0
    curr_state = None
    replay_images = []
    replay_images_resized = []
    replay_images_left_wrist_resized = []
    replay_images_right_wrist_resized = []

    log_message("Prepare the scene, and then press Enter to begin...", log_file)
    input()

    # Reset environment again to fetch first timestep observation
    obs = env.reset()

    # Fetch initial robot state (but sleep first so that robot stops moving)
    time.sleep(2)
    curr_state = env.get_qpos()

    episode_start_time = time.time()
    total_model_query_time = 0.0

    action_list = []
    action_prev = None
    action_dev_prev = np.zeros((14))

    all_time_actions = np.zeros([cfg.max_steps, cfg.max_steps+cfg.num_open_loop_steps, 14])

    try:
        while t < cfg.max_steps:
            # Get step start time (used to compute how much to sleep between steps)
            step_start_time = time.time()

            # Get observation
            obs = env.get_observation(t=t)

            # Save raw high camera image for replay video
            replay_images.append(obs.observation["images"]["cam_high"])

            # Prepare observation
            if t % 3 == 0:
                observation, img_resized, left_wrist_resized, right_wrist_resized = prepare_observation(obs, resize_size)
                observation["instruction"] = task_description
                obs_que.put([t, observation])

            if temopral_agg:
                print(t)
                if t % query_frequency == 0:
                    log_message("Requerying model...", log_file)
                    model_query_start_time = time.time()
                    time_inf, actions = command_que.get()
                    total_model_query_time += time.time() - model_query_start_time
                    all_time_actions[time_inf, time_inf:time_inf+cfg.num_open_loop_steps] = np.array(actions)
                actions_for_curr_step = all_time_actions[:, t, :]
                mask = np.all(actions_for_curr_step !=0, axis=1)
                actions_for_curr_step = actions_for_curr_step[mask]
                k=0.01
                N = len(actions_for_curr_step)
                exp_weights = np.exp(-k * np.arange(N, dtype=actions_for_curr_step.dtype))
                exp_weights /= exp_weights.sum()
                action = (exp_weights[:, None] * actions_for_curr_step).sum(axis=0, keepdims=True)
                action = action[0]

                print(action)
                # action = (exp_weights[:, None] * actions_for_curr_step).sum(axis=0, keepdim=True) # intentional error to stop the code for debug

            else:
                # If action queue is empty, requery model
                if len(action_queue) == 0:
                    # Save processed images for replay
                    replay_images_resized.append(img_resized)
                    replay_images_left_wrist_resized.append(left_wrist_resized)
                    replay_images_right_wrist_resized.append(right_wrist_resized)
                    
                    # Query model to get action
                    log_message("Requerying model...", log_file)
                    model_query_start_time = time.time()

                    time_inf, actions = command_que.get()
                    n = t - time_inf -1
                    actions = actions[: cfg.num_open_loop_steps]
                    print('actions[0].shape', actions[0].shape, len(actions))
                    total_model_query_time += time.time() - model_query_start_time
                    action_queue.extend(actions[-n:])
                    
                # Get action from queue
                action = action_queue.popleft()

            if cac:
                if action_prev is None or des_prev is None:
                    target_qpos = action
                    des_prev = copy.deepcopy(action)
                    action_prev = copy.deepcopy(action)
                else:
                    pos_dev = 0.9 * (des_prev - action_prev) 
                    pos_dev = np.clip(pos_dev, -3.0e-3, 3.0e-3)

                    dev_pred = action - des_prev
                    dev_total = pos_dev + dev_pred
                    dev_min_acc = action_dev_prev - max_acc
                    dev_max_acc = action_dev_prev + max_acc
                    dev_total = np.clip(dev_total, dev_min_acc, dev_max_acc)
                    dev_total = np.clip(dev_total, - max_dev, max_dev)

                    target_qpos = action_prev + dev_total
                    target_qpos = np.clip(target_qpos, min_action, max_action)

                    action_list.append(target_qpos)
                    
                    action_dev_prev = target_qpos - action_prev
                    des_prev = copy.deepcopy(action)
                    action_prev = copy.deepcopy(target_qpos)

                    action = copy.deepcopy(target_qpos)


            log_message("-----------------------------------------------------", log_file)
            log_message(f"t: {t}", log_file)
            log_message(f"action: {action}", log_file)

            # Execute action in environment
            if cfg.use_relative_actions:
                # Get absolute joint angles from relative action
                rel_action = action
                target_state = curr_state + rel_action
                obs = env.step(target_state.tolist())
                # Update current state (assume it is the commanded target state)
                curr_state = target_state
            else:
                obs = env.step(action.tolist())
            t += 1

            # Sleep until next timestep
            step_elapsed_time = time.time() - step_start_time
            if step_elapsed_time < STEP_DURATION_IN_SEC:
                time_to_sleep = STEP_DURATION_IN_SEC - step_elapsed_time
                log_message(f"Sleeping {time_to_sleep} sec...", log_file)
                time.sleep(time_to_sleep)

    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            log_message("\nCaught KeyboardInterrupt: Terminating episode early.", log_file)
        else:
            log_message(f"\nCaught exception: {e}", log_file)

    episode_end_time = time.time()

    # Get success feedback from user
    user_input = input("Success? Enter 'y' or 'n': ")
    success = True if user_input.lower() == "y" else False

    # Calculate episode statistics
    episode_stats = {
        "success": success,
        "total_steps": t,
        "model_query_time": total_model_query_time,
        "episode_duration": episode_end_time - episode_start_time,
    }

    return (
        episode_stats,
        replay_images,
        replay_images_resized,
        replay_images_left_wrist_resized,
        replay_images_right_wrist_resized,
        action_list,
    )


def save_episode_videos(
    replay_images,
    replay_images_resized,
    replay_images_left_wrist,
    replay_images_right_wrist,
    episode_idx,
    success,
    task_description,
    log_file=None,
):
    """Save videos of the episode from different camera angles."""
    # Save main replay video
    save_rollout_video(replay_images, episode_idx, success=success, task_description=task_description, log_file=log_file)

    # Save processed view videos
    save_rollout_video(
        replay_images_resized,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="resized",
    )
    save_rollout_video(
        replay_images_left_wrist,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="left_wrist_resized",
    )
    save_rollout_video(
        replay_images_right_wrist,
        episode_idx,
        success=success,
        task_description=task_description,
        log_file=log_file,
        notes="right_wrist_resized",
    )


@draccus.wrap()
def eval_aloha(cfg: GenerateConfig) -> None:
    """Main function to evaluate a trained policy in a real-world ALOHA environment."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Get ALOHA environment
    env = get_aloha_env()

    # Get server endpoint for remote inference
    server_endpoint = get_server_endpoint(cfg)

    # Initialize task description
    task_description = ""

    cac = cfg.cac

    # Start evaluation
    num_rollouts_completed, total_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(cfg.num_rollouts_planned)):
        # Get task description from user
        task_description = get_next_task_label(task_description)
        log_message(f"\nTask: {task_description}", log_file)

        log_message(f"Starting episode {num_rollouts_completed + 1}...", log_file)

        t2 = threading.Thread(target=inference_query_thread, args=(server_endpoint,), daemon=True)
        t2.start()

        # Run episode
        episode_stats, replay_images, replay_images_resized, replay_images_left_wrist, replay_images_right_wrist, action_list = (
            run_episode(cfg, env, task_description, server_endpoint, resize_size, log_file, cac=cac)
        )

        stop_event.set()
        obs_que.clear_que()
        command_que.clear_que()

        # Update counters
        num_rollouts_completed += 1
        if episode_stats["success"]:
            total_successes += 1

        # Save videos
        save_episode_videos(
            replay_images,
            replay_images_resized,
            replay_images_left_wrist,
            replay_images_right_wrist,
            num_rollouts_completed,
            episode_stats["success"],
            task_description,
            log_file,
        )

        if cac:
            filename_action = f'action_cac_{episode_idx}.txt'
        else:
            filename_action = f'action_{episode_idx}.txt'
        np.savetxt(filename_action, np.concatenate(action_list))

        # Log results
        log_message(f"Success: {episode_stats['success']}", log_file)
        log_message(f"# episodes completed so far: {num_rollouts_completed}", log_file)
        log_message(f"# successes: {total_successes} ({total_successes / num_rollouts_completed * 100:.1f}%)", log_file)
        log_message(f"Total model query time: {episode_stats['model_query_time']:.2f} sec", log_file)
        log_message(f"Total episode elapsed time: {episode_stats['episode_duration']:.2f} sec", log_file)

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(num_rollouts_completed) if num_rollouts_completed > 0 else 0

    # Log final results
    log_message("\nFinal results:", log_file)
    log_message(f"Total episodes: {num_rollouts_completed}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_aloha()
