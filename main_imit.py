import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from imitation.data.types import TrajectoryWithRew
from imitation.algorithms.bc import BC

from rl_gym import SimRobot  # Ensure this import is correct
from misc_scripts.weight_getter import *  # Ensure this import is correct

# Define your environment parameters
state_count = 10
state_size = 5  # w x z theta t (except last state has no t)
obv_state_dimension = state_count * state_size - 1

# Create and monitor your custom environment
env = SimRobot(obv_state_dimension, 3, state_count, state_size)
env = Monitor(env, "monitoring/", allow_early_resets=True)
env.reset()

# Function to load demonstrations from a file
def load_demonstrations(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Convert lists back to numpy arrays where needed
    for demonstration in data:
        demonstration['observations'] = np.array(demonstration['observations'])
        demonstration['actions'] = np.array(demonstration['actions'])
    
    return data

# Function to add noise to demonstrations
def add_noise_to_demonstrations(demonstrations, noise_level=0.05, num_copies=10):
    noisy_demonstrations = []
    for demo in demonstrations:
        for _ in range(num_copies):
            noisy_observations = demo["observations"] + noise_level * np.random.randn(*demo["observations"].shape)
            noisy_actions = demo["actions"] + noise_level * np.random.randn(*demo["actions"].shape)
            noisy_actions = noisy_actions[:-1]
            rewards = demo["rewards"][:-1]
            dones = demo["dones"][:-1]
            noisy_demo = {
                "observations": noisy_observations,
                "actions": noisy_actions,
                "rewards": rewards,  # Keep rewards the same
                "dones": dones
            }
            noisy_demonstrations.append(noisy_demo)
    return noisy_demonstrations

# Function to convert demonstrations to TrajectoryWithRew format
def convert_to_trajectories(demonstrations):
    trajectories = []
    for demo in demonstrations:
        trajectories.append(
            TrajectoryWithRew(
                obs=np.array(demo["observations"]),
                acts=np.array(demo["actions"]),
                rews=np.array(demo["rewards"]),
                infos=None,
                terminal=demo["dones"][-1]
            )
        )
    return trajectories

# Load demonstrations from file
demonstrations = load_demonstrations('demonstrations.txt')
noisy_demos = add_noise_to_demonstrations(demonstrations)
# Convert the noisy demonstrations
trajectories = convert_to_trajectories(noisy_demos)

policy_kwargs = {
    "net_arch": [64, 64, 64],  # Example architecture, adjust as necessary
}

# Initialize the PPO model with your custom environment
ppo = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
# ppo = PPO.load("ppo_bc_model", env)
print("Model ready, beginning behavioral cloning")
print(ppo.policy)

# Initialize Behavioral Cloning
rng = np.random.default_rng()
bc_trainer = BC(
    observation_space=env.observation_space, 
    action_space=env.action_space, 
    demonstrations=trajectories,
    rng=rng
)
# Train the BC model
bc_trainer.train(n_epochs=1)

# Save the BC model weights
bc_trainer.policy.save("bc_policy")

print(bc_trainer.policy)

# Note: Stable-Baselines3 models do not have a straightforward `load` method for policies. 
# Instead, consider using the trained BC policy as an initial policy for PPO or directly copying weights if compatible.
ppo.policy = bc_trainer.policy

# Define callbacks
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='ppo_model')
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=500, deterministic=True, render=False)

print("Behavioral cloning done, begin learning")
env.reset()

# Train the PPO model
ppo.learn(total_timesteps=500, callback=[checkpoint_callback, eval_callback])

# Save the trained model
ppo.save("ppo_bc_model")
print("Learning finished! ")