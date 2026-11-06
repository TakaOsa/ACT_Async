import torch
import numpy as np
import os
import time
import h5py
import pickle
import argparse
import copy
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

import threading
import queue

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

# from sim_env import BOX_POSE
BOX_POSE = [None] 

import IPython
e = IPython.embed

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


obs_que = OverwriteQueue()
command_que = OverwriteQueue()

def inference_thread(ckpt_dir, ckpt_name, policy_class, policy_config, seed):
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    chunk_size = policy_config['num_queries']
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats_chunk{chunk_size}_seed{seed}.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    time_step = 0
    inf_cnt = 0
    with torch.inference_mode():
        while True:
            # if start_flag is False:
            time.sleep(0.01) # default 0.01
            if not obs_que.empty():
                time_step, qpos, obs= obs_que.get()
                all_actions = policy(qpos, obs)
                print('inference thread', time_step)
                command_que.put([time_step, all_actions])
                inf_cnt += 1
                # if time_step > 498:
                #     print('inf_cnt', inf_cnt)

    del policy
    print('end of inference thread.')
    
def main(args):
    set_seed(1)
    # Command-line arguments
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    seed = args['seed']
    chunk_size = args['chunk_size']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'cac': args['cac'],
        'skip': args['skip'],
        'var_threshold': args['var_threshold'],
        'record': args['record'],
        'record_epi_id': args['record_epi_id'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best_chunk{chunk_size}_seed{seed}.ckpt'] #policy_best
        results = []
        for ckpt_name in ckpt_names:
            ckpt_dir = config['ckpt_dir']
            policy_class = config['policy_class']
            policy_config = config['policy_config']
            t2 = threading.Thread(target=inference_thread, args=(ckpt_dir, ckpt_name, policy_class, policy_config, seed), daemon=True)
            t2.start()
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)

            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats_chunk{chunk_size}_seed{seed}.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best_chunk{chunk_size}_seed{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    
    onscreen_cam = 'angle'
    seed = config['seed']
    chunk_size = policy_config['num_queries']

    cac = config['cac']
    skip_flag = config['skip']
    var_threshold = config['var_threshold']

    record = config['record']
    record_epi_id = config['record_epi_id']

    # load policy and stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats_chunk{chunk_size}_seed{seed}.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # Requires ALOHA
        from aloha_scripts.real_env import make_real_env # Requires ALOHA
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    # Number of frames to skip
    skip_num = 1 if skip_flag else 0

    step_num = 1 + skip_num

    query_frequency = int(policy_config['num_queries'])
    if temporal_agg:
        query_frequency = 2
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 1
    episode_returns = []
    highest_rewards = []
    action_prev = None
    action_dev_prev = np.zeros((14))
    des_prev = None

    pred_step_k_max = 1.0
    pred_step_k = 1.0
    
    max_action = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.2])
    min_action = np.array([-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 0.0, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 0.0])

    max_dev = np.array([0.025, 0.025, 0.025, 0.025, 0.015, 0.015, 0.2, 0.025, 0.025, 0.025, 0.025, 0.015, 0.015, 0.2])
    max_acc = np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.1, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.1]) # Experiment settings for the NeurIPS workshop
    task_duration_log = np.zeros((num_rollouts))

    # Save the dataset
    overwrite = True
    if cac:
        dataset_dir = ckpt_dir + '/cac_skip'+ str(step_num) +'_adapt_async_rollout'
    else:
        dataset_dir = ckpt_dir + '/act_async_rollout'
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    for rollout_id in range(num_rollouts):

        dataset_name = f'episode_{rollout_id + record_epi_id}'
        dataset_path = os.path.join(dataset_dir, dataset_name)
        if os.path.isfile(dataset_path) and not overwrite:
            print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
            exit()

        rollout_id += 0
        ### Set up the task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # Used when resetting the simulation
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # Used when resetting the simulation

        ts = env.reset()

        if record:
            timesteps = [ts]
            actions = []

        ### Set up on-screen rendering
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps*step_num, max_timesteps*step_num+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps*step_num, state_dim)).cuda()
        qpos_log = torch.zeros((1, max_timesteps, state_dim)).cuda()
        action_log = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        chunk_num = int(num_queries)
        task_done_flag = False
        task_duration = 0
        t_start = 0
        t_end = 0
        t_epi_end = 0
        with torch.inference_mode():
            t_start = time.time()
            t = 0
            step = 0
            while step < max_timesteps:
                ### Update the on-screen rendering and wait for DT seconds
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### Extract joint positions and collect images from the previous time step
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                obs_que.put([t, qpos, curr_image])

                if action_prev is None or des_prev is None:
                        des_prev = copy.deepcopy(qpos_numpy)
                        action_prev = copy.deepcopy(qpos_numpy)

                # if t==0:
                    # time_inf, all_actions = command_que.get()
                    # print(time_inf, all_actions_pred.shape)

                ### Query the policy
                if config['policy_class'] == "ACT":
 
                    if t % query_frequency == 0:
                        time_inf, all_actions = command_que.get()
                        all_time_actions[[time_inf], time_inf:time_inf+chunk_num] = all_actions
                        print('current:', t, ' inferred step:', time_inf)

                    # if temporal_agg:
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                    if cac:
                        actions_for_next_2step = all_time_actions[:, t + 2]
                        actions_populated_next2 = torch.all(actions_for_next_2step != 0, dim=1)
                        actions_for_next_2step = actions_for_next_2step[actions_populated_next2]

                else:
                    raise NotImplementedError

                if cac:
                    ### Post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    if action_prev is None or des_prev is None:
                        target_qpos = action
                        des_prev = copy.deepcopy(action)
                        action_prev = copy.deepcopy(action)

                    else:
                        if step_num > 0 and t < step_num * 20:
                            pred_step_k =  pred_step_k_max * t / (step_num * 20.0)
                        else:
                            pred_step_k = pred_step_k_max

                        pos_dev = 0.6 * (des_prev - action_prev) # 0.06
                        pos_dev = np.clip(pos_dev, -1.5e-3, 1.5e-3)
                        # pos_dev[np.abs(pos_dev) < 4.0e-8] = 0
                        dev_pred = action - des_prev
                        # dev_ff = 0.05 * (action_next - action)
                        dev_total = pos_dev + dev_pred
                        dev_min_acc = action_dev_prev - max_acc
                        dev_max_acc = action_dev_prev + max_acc
                        dev_total = np.clip(dev_total, dev_min_acc, dev_max_acc)
                        dev_total = np.clip(dev_total, - max_dev, max_dev)

                        target_qpos = action_prev + dev_total
                        target_qpos = np.clip(target_qpos, min_action, max_action)

                        actions_for_curr_step_nogripper = np.delete(actions_for_curr_step.cpu().numpy(), [6,13], axis=1)
                        action_vars =np.std(actions_for_curr_step_nogripper, axis=0)
                        action_var = np.linalg.norm(action_vars)

                        if action_var < var_threshold:
                            print('skip', t)
                            t += step_num
                        else:
                            t += step_num -1
                        # print('qpos_numpy', qpos_numpy[7], 'action' , action[7], 'action_prev' , action_prev[7], 'des_prev', des_prev[7], 'target_qpos', target_qpos[7])

                        action_dev_prev = target_qpos - action_prev
                        action_prev = copy.deepcopy(target_qpos)
                        des_prev = copy.deepcopy(action)
                        # print('pos_dev', pos_dev)
                else:
                    ### Post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action

                ### Step the environment
                ts = env.step(target_qpos)
                
                # store the observation and action
                if record:
                    timesteps.append(ts)
                    actions.append(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

                step +=1
                if cac:
                    print('control thread, t:', t, action_var, '  step: ', step)
                else:
                    print('control thread, t:', t)

            plt.close()

        # Finish the episode
        t_epi_end = time.time()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # Open both grippers
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(dataset_dir, f'chunk{num_queries}_seed{seed}_video{rollout_id + record_epi_id}.mp4'))

        filename_qpos = f'/chunk{num_queries}_seed{seed}_episode_{rollout_id + record_epi_id}_q.txt'
        np.savetxt(dataset_dir+filename_qpos, np.asarray(qpos_list))
        filename_action = f'/chunk{num_queries}_seed{seed}_episode_{rollout_id + record_epi_id}_action.txt'
        np.savetxt(dataset_dir+filename_action, np.asarray(target_qpos_list))

        # Save the episode duration
        filename_time = f'/chunk{num_queries}_seed{seed}_episode_{rollout_id + record_epi_id}_time.txt'
        epi_duration = t_epi_end - t_start
        np.savetxt(dataset_dir+filename_time, [epi_duration])
        print(f'episode duration:{epi_duration:.2f}sec')

        if record:
            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/observations/effort': [],
                '/action': [],
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []

            print('len(action):', len(action), ' max_timesteps:', max_timesteps)

            while actions:
                action = actions.pop(0)
                ts = timesteps.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/observations/effort'].append(ts.observation['effort'])
                data_dict['/action'].append(action)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

            # HDF5
            t0 = time.time()
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
                root.attrs['sim'] = False
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )

                _ = obs.create_dataset('qpos', (max_timesteps, 14))
                _ = obs.create_dataset('qvel', (max_timesteps, 14))
                _ = obs.create_dataset('effort', (max_timesteps, 14))
                _ = root.create_dataset('action', (max_timesteps, 14))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs')


    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # Save the success rate to a text file
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)

    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # Backpropagation
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')

        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # CAC and skipping
    parser.add_argument('--cac', action='store_true')
    parser.add_argument('--skip', action='store_true')
    parser.add_argument('--var_threshold', type=float, help='Action variability threshold for adaptive frame skipping', default=0.2)

    parser.add_argument('--record', action='store_true')
    parser.add_argument('--record_epi_id', default=0, type=int)
    
    main(vars(parser.parse_args()))
