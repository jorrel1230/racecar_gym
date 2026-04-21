import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import glob
import os
import re
import torch as th
import torch.nn as nn
import numpy as np
import gymnasium
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
import racecar_gym.envs.gym_api


class RacecarFeaturesExtractor(BaseFeaturesExtractor):
    """
    Wide-View CNN features extractor for racecar_gym.
    Processes Lidar with downsampling, normalization, and large-receptive-field convolutions.
    """
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        # 1. Lidar Wide-View CNN Branch
        # Input shape: (Batch, 1, 1080)
        lidar_shape = observation_space.spaces['lidar'].shape
        self.lidar_cnn = nn.Sequential(
            # Aggressive downsample: 1080 -> 270 (removes noise, preserves corridor geometry)
            nn.AvgPool1d(kernel_size=4, stride=4),
            # Layer 1: Wide kernel to capture large field-of-view gap structure
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            # Layer 2: Mid-range curvature / opening detection
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # Layer 3: Global context (which side has more space)
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute lidar output dimension
        with th.no_grad():
            sample_lidar = th.zeros((1, 1, lidar_shape[0]))
            lidar_output_dim = self.lidar_cnn(sample_lidar).shape[1]

        # 2. Other sensors branch (velocity, acceleration, time, progress_sensor)
        self.other_keys = ['velocity', 'acceleration', 'time', 'progress_sensor']
        other_input_dim = 0
        for key in self.other_keys:
            if key in observation_space.spaces:
                space = observation_space.spaces[key]
                other_input_dim += np.prod(space.shape) if len(space.shape) > 0 else 1

        self.other_mlp = nn.Sequential(
            nn.Linear(other_input_dim, 64),
            nn.ReLU(),
        )

        self._features_dim = lidar_output_dim + 64

    def forward(self, observations):
        # Lidar: Normalize distances (0.25m-15m) to [0, 1]
        lidar = observations['lidar'].unsqueeze(1) / 15.0
        lidar_features = self.lidar_cnn(lidar)

        # Other sensors: Normalize and concatenate
        other_data = []
        for key in self.other_keys:
            if key in observations:
                val = observations[key]
                if key == 'time':
                    val = val / 30.0
                elif key == 'velocity':
                    val = val / 14.0

                if len(val.shape) == 1:
                    val = val.unsqueeze(1)
                other_data.append(th.flatten(val, start_dim=1))

        other_input = th.cat(other_data, dim=1)
        other_features = self.other_mlp(other_input)

        return th.cat([lidar_features, other_features], dim=1)


class MultiInputRacecarEnv(gymnasium.Wrapper):
    """Filter out global pose and add progress sensor."""
    def __init__(self, env):
        super().__init__(env)
        self._act_keys = sorted(env.action_space.spaces.keys())
        act_low = np.concatenate([env.action_space[k].low.flatten() for k in self._act_keys])
        act_high = np.concatenate([env.action_space[k].high.flatten() for k in self._act_keys])
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        orig_space = env.observation_space
        new_spaces = {}
        for k, v in orig_space.spaces.items():
            if k != 'pose':
                # Scalar obs (shape ()) break SB3 replay buffer indexing — promote to (1,)
                if v.shape == ():
                    new_spaces[k] = spaces.Box(low=v.low.flatten(), high=v.high.flatten(),
                                               shape=(1,), dtype=v.dtype)
                else:
                    new_spaces[k] = v
        new_spaces['progress_sensor'] = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict(new_spaces)

        self._stuck_steps = 0
        self._last_progress = 0

    def _unflatten_action(self, action):
        action_dict = {}
        offset = 0
        for k in self._act_keys:
            size = np.prod(self.env.action_space[k].shape)
            action_dict[k] = action[offset:offset + size]
            offset += size
        # Remap motor: policy outputs [-1,1], shift so 0 = mild forward (0.3)
        if 'motor' in action_dict:
            action_dict['motor'] = np.clip(action_dict['motor'] * 0.7 + 0.3, -1.0, 1.0)
        return action_dict

    def _process_obs(self, obs, info):
        new_obs = {}
        for k, v in obs.items():
            if k != 'pose':
                v = np.asarray(v)
                new_obs[k] = v.reshape(1) if v.shape == () else v
        new_obs['progress_sensor'] = np.array([info['progress']], dtype=np.float32)
        return new_obs

    def reset(self, **kwargs):
        if 'options' not in kwargs or kwargs['options'] is None:
            kwargs['options'] = dict(mode='random_biased')
        obs, info = self.env.reset(**kwargs)
        self._stuck_steps = 0
        self._last_progress = info['lap'] + info['progress']
        return self._process_obs(obs, info), info

    def step(self, action):
        action_dict = self._unflatten_action(action)
        obs, reward, done, truncated, info = self.env.step(action_dict)

        # Stuck detection
        current_progress = info['lap'] + info['progress']
        if current_progress <= self._last_progress:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
            self._last_progress = current_progress

        if self._stuck_steps > 200:
            truncated = True

        return self._process_obs(obs, info), reward, done, truncated, info

    def get_wrapper_attr(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            return getattr(self.env, name)


def make_env(track='SingleAgentAustria-v0', render_mode=None):
    def _init():
        env = gymnasium.make(track, render_mode=render_mode)
        env = MultiInputRacecarEnv(env)
        return env
    return _init


def inject_demos(model, demo_dir):
    """Load human demo .npz files and inject into SAC replay buffer.

    Demo files have sequential (obs, action) pairs per episode.
    next_obs = obs[t+1], reward = 0 (placeholder), done = False.
    Demo transitions seed the buffer so SAC updates start with human-quality data.
    """
    files = sorted(glob.glob(os.path.join(demo_dir, 'ep_*.npz')))
    if not files:
        print(f'  [demos] No ep_*.npz found in {demo_dir}')
        return 0

    obs_space = model.observation_space
    sorted_keys = sorted(obs_space.spaces.keys())
    shapes = [obs_space.spaces[k].shape for k in sorted_keys]
    sizes = [int(np.prod(s)) for s in shapes]

    n_envs = model.n_envs
    total = 0
    for f in files:
        d = np.load(f)
        obs_flat = d['observations']  # (T, flat_dim)
        actions = d['actions']        # (T, 2) — already in policy space

        for t in range(len(obs_flat) - 1):
            obs_dict = _reconstruct_obs(obs_flat[t], sorted_keys, shapes, sizes)
            next_obs_dict = _reconstruct_obs(obs_flat[t + 1], sorted_keys, shapes, sizes)
            # Buffer expects (n_envs, *shape) — tile single transition across all env slots
            model.replay_buffer.add(
                obs={k: np.stack([v] * n_envs) for k, v in obs_dict.items()},
                next_obs={k: np.stack([v] * n_envs) for k, v in next_obs_dict.items()},
                action=np.tile(actions[t], (n_envs, 1)),
                reward=np.zeros(n_envs),
                done=np.zeros(n_envs, dtype=bool),
                infos=[{}] * n_envs,
            )
            total += 1

    print(f'  [demos] Injected {total} transitions from {len(files)} episodes into replay buffer.')
    return total


def _reconstruct_obs(flat, sorted_keys, shapes, sizes):
    obs_dict = {}
    offset = 0
    for key, shape, size in zip(sorted_keys, shapes, sizes):
        obs_dict[key] = flat[offset:offset + size].reshape(shape)
        offset += size
    return obs_dict


class CheckpointCallback(BaseCallback):
    """Periodic checkpointing with tqdm progress."""
    def __init__(self, eval_freq, checkpoint_dir, total_timesteps, n_envs, step_offset=0, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.checkpoint_dir = checkpoint_dir
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.step_offset = step_offset
        self.pbar = None
        self._last_pbar_update = 0

    def _on_training_start(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.pbar = tqdm(total=self.total_timesteps, desc='Training', unit='step')

    def _on_step(self) -> bool:
        current = self.num_timesteps
        self.pbar.update(current - self._last_pbar_update)
        self._last_pbar_update = current
        if self.num_timesteps % self.eval_freq < self.n_envs:
            step = self.num_timesteps + self.step_offset
            path = os.path.join(self.checkpoint_dir, f'sac_step_{step}')
            self.model.save(path)
            if self.verbose:
                self.pbar.write(f'[Step {step}] Saved checkpoint: {path}')
        return True

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAC on racecar_gym')
    parser.add_argument('--num-cpu', type=int, default=1,
                        help='Parallel envs (SAC is off-policy; 1-2 usually sufficient)')
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--checkpoint-freq', type=int, default=10_000)
    parser.add_argument('--track', type=str, default='SingleAgentAustria-v0')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint .zip to resume from')
    parser.add_argument('--render', action='store_true', help='Show PyBullet GUI during training')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--buffer-size', type=int, default=300_000,
                        help='Replay buffer size (reduce if OOM)')
    parser.add_argument('--demos', type=str, default=None, metavar='DIR',
                        help='Directory of human demo .npz files to seed replay buffer')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    render_mode = 'human' if args.render else None

    if args.render and args.num_cpu > 1:
        print('WARNING: Rendering with multiple CPUs opens multiple PyBullet windows.')
        print('Use --num-cpu 1 --render instead.\n')

    if args.num_cpu > 1:
        train_env = SubprocVecEnv([make_env(args.track, render_mode) for _ in range(args.num_cpu)])
    else:
        train_env = DummyVecEnv([make_env(args.track, render_mode)])

    step_offset = 0
    if args.resume:
        match = re.search(r'sac_step_(\d+)', args.resume)
        if match:
            step_offset = int(match.group(1))
        print(f'  Resuming from: {args.resume} (step offset: {step_offset})')
        model = SAC.load(args.resume, env=train_env)
        model.learning_rate = args.learning_rate
    else:
        policy_kwargs = dict(
            features_extractor_class=RacecarFeaturesExtractor,
            net_arch=[256, 256],
        )

        model = SAC(
            'MultiInputPolicy',
            train_env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=512,
            tau=0.005,           # soft target update
            gamma=0.99,
            train_freq=1,        # update every env step
            gradient_steps=2,
            ent_coef='auto',     # automatic entropy tuning
            learning_starts=1000,
            device='auto',
        )

    if args.demos:
        print(f'Seeding replay buffer with human demos from {args.demos}')
        inject_demos(model, args.demos)

    callback = CheckpointCallback(
        eval_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        total_timesteps=args.total_timesteps,
        n_envs=args.num_cpu,
        step_offset=step_offset,
    )

    print(f'Training SAC on {args.track}')
    print(f'  {args.num_cpu} env(s) | {args.total_timesteps} steps | buffer {args.buffer_size}')
    print(f'  Device: {model.device}')

    model.learn(total_timesteps=args.total_timesteps, callback=callback,
                reset_num_timesteps=not args.resume)

    final_step = args.total_timesteps + step_offset
    final_path = os.path.join(args.checkpoint_dir, f'sac_step_{final_step}')
    model.save(final_path)
    print(f'\nDone. Final model: {final_path}')

    train_env.close()
