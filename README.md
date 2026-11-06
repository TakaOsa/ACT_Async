# ACT_Async

This repository provides code to reproduce the experiments on asynchronous policy inference and robot action execution described in the following paper:

Takayuki Osa and Kazuki Ota.
**Control-Aware Command Generation for Asynchronous Control and Action Inference with Vision-Based Policies towards Fast Execution**, 8th Robot Learning Workshop @ NeurIPS 2026.

The repository currently includes:

- `imitate_episodes_async.py`: asynchronous policy inference and action execution using Action Chunking Transformer (ACT) policies.
- `run_aloha_eval_async_cac.py`: asynchronous policy inference and action execution using OpenVLA-OFT in a real-world ALOHA environment.

This repository implements Control-Aware Command Generation (CAC) and adaptive skipping, proposed in the paper to enable smooth and fast execution. CAC is implemented for both ACT and OpenVLA-OFT, while adaptive skipping is implemented for ACT.

## Asynchronous inference and action execution with ACT

1. Set up ACT by following the instructions in the [ACT repository](https://github.com/tonyzhaozh/act).

2. Copy `imitate_episodes_async.py` into the directory containing `imitate_episodes.py`.

3. Train a policy by following the instructions in the ACT repository. For example, for the `box_hand_over` task, run:

```bash
python3 imitate_episodes.py --task_name box_hand_over --ckpt_dir /home/aloha/data/box_hand_over/checkpoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 3000  --lr 1e-5 --seed 0
```

4. Execute the policy using `imitate_episodes_async.py` instead of `imitate_episodes.py`.

For asynchronous execution without CAC or adaptive skipping, run:

```bash
python3 imitate_episodes_async.py --task_name box_hand_over --ckpt_dir /home/aloha/data/box_hand_over/checkpoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 3000  --lr 1e-5 --seed 0 --eval --temporal_agg
```

For asynchronous execution with CAC, run:

```bash
python3 imitate_episodes_async.py --task_name box_hand_over --ckpt_dir /home/aloha/data/box_hand_over/checkpoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 3000  --lr 1e-5 --seed 0 --eval --temporal_agg --cac
```

For asynchronous execution with both CAC and adaptive skipping, run:

```bash
python3 imitate_episodes_async.py --task_name box_hand_over --ckpt_dir /home/aloha/data/box_hand_over/checkpoint --policy_class ACT --kl_weight 10 --chunk_size 50 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 3000  --lr 1e-5 --seed 0 --eval --temporal_agg --cac --skip
```

## Asynchronous inference and action execution with OpenVLA-OFT

1. Set up OpenVLA-OFT by following the instructions in the [OpenVLA-OFT repository](https://github.com/moojink/openvla-oft).

2. Copy `run_aloha_eval_async_cac.py` into the directory containing `run_aloha_eval.py` (`experiments/robot/aloha/`).

3. Start a policy server by following the instructions in the OpenVLA-OFT repository. For example, run:

```bash
python vla-scripts/deploy.py --pretrained_checkpoint ~/OpenVLA-oft/openvla-7b+aloha1_pen_in_drawer_demos+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--25_acts_chunk--continuous_acts--L1_regression--3rd_person_img--left_right_wrist_imgs--proprio_state--film--100000_chkpt/ --use_l1_regression True --use_film True --num_images_in_input 3 --use_proprio True --center_crop True --unnorm_key aloha1_pen_in_drawer_demos
```

4. Run the robot controller using `run_aloha_eval_async_cac.py` instead of `run_aloha_eval.py`.

For asynchronous execution with CAC, run:

```bash
python experiments/robot/aloha/run_aloha_eval_async_cac.py --center_crop True --num_open_loop_steps 25 --use_vla_server True --vla_server_url 127.0.0.1 --num_rollouts_planned 1 --max_steps 1200
```

To disable CAC, add `--cac False`:

```bash
python experiments/robot/aloha/run_aloha_eval_async_cac.py --center_crop True --num_open_loop_steps 25 --use_vla_server True --vla_server_url 127.0.0.1 --num_rollouts_planned 1 --max_steps 1200 --cac False
```

In our experience, asynchronous inference and action execution with OpenVLA-OFT can be unstable without CAC. If you disable CAC, be prepared to stop the robot at any time.
