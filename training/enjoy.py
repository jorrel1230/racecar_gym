"""Watch trained model drive with the new Multi-Input CNN architecture.

Usage:
    python enjoy.py <model_path> [--track TRACK] [--episodes N]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from time import sleep

import gymnasium
import numpy as np
import pybullet as p
from stable_baselines3 import SAC, PPO
import racecar_gym.envs.gym_api
from train import MultiInputRacecarEnv as _MultiInputRacecarEnv


class MultiInputRacecarEnv(_MultiInputRacecarEnv):
    """Override reset to use grid start for evaluation."""
    def reset(self, **kwargs):
        if 'options' not in kwargs or kwargs['options'] is None:
            kwargs['options'] = dict(mode='grid')
        return super().reset(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Watch trained model drive',
        epilog='Example: python enjoy.py ./checkpoints/ppo_final.zip --track SingleAgentAustria-v0',
    )
    parser.add_argument('model', type=str, help='Path to saved model (.zip)')
    parser.add_argument('--track', type=str, default='SingleAgentAustria-v0')
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()

    # Use render_mode='human' for follow-cam visualization
    env = gymnasium.make(args.track, render_mode='human')
    env = MultiInputRacecarEnv(env)
    
    # Auto-detect SAC vs PPO from filename
    loader = SAC if 'sac' in os.path.basename(args.model).lower() else PPO
    model = loader.load(args.model)
    print(f'Loaded {loader.__name__} model from {args.model}')

    for ep in range(args.episodes):
        obs, _ = env.reset(options=dict(mode='grid'))
        
        # Get vehicle ID for follow cam after reset
        vehicle_id = env.unwrapped._scenario.agent.vehicle_id
        
        done = False
        total_reward = 0.0
        t = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # Follow camera logic
            pos, orn = p.getBasePositionAndOrientation(vehicle_id)
            _, _, yaw = p.getEulerFromQuaternion(orn)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=np.degrees(yaw) - 90,
                cameraPitch=-25,
                cameraTargetPosition=pos
            )
            
            print(f'  step={t} reward={reward:.4f} total={total_reward:.2f} action={action}')
            sleep(0.01)
            if t % 30 == 0:
                env.render()
            t += 1
            if truncated:
                break

        print(f'Episode {ep+1}: reward={total_reward:.2f}')

    env.close()


if __name__ == '__main__':
    main()
