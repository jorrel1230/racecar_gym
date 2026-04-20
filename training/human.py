"""Human keyboard control for racecar_gym.

Controls:
    W / Up    - Throttle forward
    S / Down  - Brake
    A / Left  - Steer left
    D / Right - Steer right
    R         - Reset episode
    Q         - Quit

Usage:
    python human.py [--track TRACK] [--mode grid|random_biased] [--record ./demos/]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from time import sleep

import numpy as np
import pybullet as p
import gymnasium
import racecar_gym.envs.gym_api
from train import MultiInputRacecarEnv

# PyBullet key codes
KEY_UP    = p.B3G_UP_ARROW
KEY_DOWN  = p.B3G_DOWN_ARROW
KEY_LEFT  = p.B3G_LEFT_ARROW
KEY_RIGHT = p.B3G_RIGHT_ARROW
KEY_W = ord('w')
KEY_S = ord('s')
KEY_A = ord('a')
KEY_D = ord('d')
KEY_R = ord('r')
KEY_Q = ord('q')

KEY_IS_DOWN = p.KEY_IS_DOWN


def get_raw_action(keys):
    """Human input → (motor_raw, steering_raw) in {-1, 0, +1}."""
    motor = 0.0
    steering = 0.0
    if keys.get(KEY_UP, 0) & KEY_IS_DOWN or keys.get(KEY_W, 0) & KEY_IS_DOWN:
        motor = 1.0
    if keys.get(KEY_DOWN, 0) & KEY_IS_DOWN or keys.get(KEY_S, 0) & KEY_IS_DOWN:
        motor = -1.0
    if keys.get(KEY_LEFT, 0) & KEY_IS_DOWN or keys.get(KEY_A, 0) & KEY_IS_DOWN:
        steering = -1.0
    if keys.get(KEY_RIGHT, 0) & KEY_IS_DOWN or keys.get(KEY_D, 0) & KEY_IS_DOWN:
        steering = 1.0
    return motor, steering


def raw_to_policy(motor_raw, steering_raw):
    """Invert MultiInputRacecarEnv motor remap so recorded targets match policy output space.

    Wrapper applies: motor_raw = policy * 0.7 + 0.3
    Inverse:         policy    = (motor_raw - 0.3) / 0.7
    """
    motor_policy = np.clip((motor_raw - 0.3) / 0.7, -1.0, 1.0)
    return np.array([motor_policy, steering_raw], dtype=np.float32)


def run(args):
    base_env = gymnasium.make(args.track, render_mode='human')
    env = MultiInputRacecarEnv(base_env)

    recording = args.record is not None
    if recording:
        os.makedirs(args.record, exist_ok=True)
        print(f'Recording demos → {args.record}')

    print(f'\nTrack: {args.track}')
    print('Controls: W/Up=throttle  S/Down=brake  A/Left=left  D/Right=right  R=reset  Q=quit')
    if recording:
        print('  [RECORDING] Each episode saved on reset/quit/done.')
    print()

    episode = 0
    while True:
        obs, info = env.reset(options=dict(mode=args.mode))
        episode += 1
        total_reward = 0.0
        step = 0

        ep_obs = []
        ep_actions = []

        vehicle_id = env.unwrapped._scenario.agent.vehicle_id

        reset_requested = False
        done = False

        while not (done or reset_requested):
            keys = p.getKeyboardEvents()

            if keys.get(KEY_Q, 0) & KEY_IS_DOWN:
                if recording:
                    _save_episode(args.record, episode, ep_obs, ep_actions)
                env.close()
                return

            if keys.get(KEY_R, 0) & KEY_IS_DOWN:
                reset_requested = True
                break

            motor_raw, steering_raw = get_raw_action(keys)

            # Policy-space action (what BC will learn to output)
            action_policy = raw_to_policy(motor_raw, steering_raw)

            obs, reward, done, truncated, info = env.step(action_policy)
            total_reward += reward
            step += 1

            if recording:
                # Flatten Dict obs to 1D for storage
                ep_obs.append(_flatten_obs(obs))
                ep_actions.append(action_policy.copy())

            pos, orn = p.getBasePositionAndOrientation(vehicle_id)
            _, _, yaw = p.getEulerFromQuaternion(orn)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=np.degrees(yaw) - 90,
                cameraPitch=-25,
                cameraTargetPosition=pos,
            )

            print(
                f'\r  ep={episode} step={step:4d} | '
                f'motor={motor_raw:+.1f} steer={steering_raw:+.1f} | '
                f'reward={reward:+7.2f} total={total_reward:+8.2f} | '
                f'progress={info["progress"]:.3f} lap={info["lap"]}',
                end='', flush=True,
            )

            sleep(0.02)

            if truncated:
                done = True

        print(f'\n  Episode {episode} ended: total_reward={total_reward:.2f} steps={step}')

        if recording and ep_obs:
            _save_episode(args.record, episode, ep_obs, ep_actions)


def _flatten_obs(obs_dict):
    """Flatten ordered Dict obs to 1D numpy array. Keys sorted for consistency."""
    return np.concatenate([obs_dict[k].flatten() for k in sorted(obs_dict.keys())])


def _save_episode(record_dir, episode, obs_list, action_list):
    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(action_list, dtype=np.float32)
    path = os.path.join(record_dir, f'ep_{episode:04d}.npz')
    np.savez_compressed(path, observations=obs_arr, actions=act_arr)
    print(f'  Saved {len(obs_list)} steps → {path}')


def main():
    parser = argparse.ArgumentParser(description='Human keyboard control for racecar_gym')
    parser.add_argument('--track', type=str, default='SingleAgentAustria-v0')
    parser.add_argument('--mode', type=str, default='grid',
                        choices=['grid', 'random', 'random_biased'])
    parser.add_argument('--record', type=str, default=None, metavar='DIR',
                        help='Directory to save demonstration .npz files')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
