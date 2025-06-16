import os
import sys
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from MADDPG_agent import MADDPGAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to sys.path for local imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from env import Env_MAPPO  # Import custom environment

# Configuration Parameters
# Environment settings
SCENARIO = "MADDPG_UAV_UGV_Cooperation"
env = Env_MAPPO()
NUM_AGENTS = env.num_uavs

# Training loop settings
NUM_EPISODES = 10000
PRINT_EVERY = 50
REWARD_AVG_WINDOW = 100
BUFFER_SIZE = 1e6
BATCH_SIZE = 128
LEARNING_START = BATCH_SIZE * 25
UPDATES_PER_STEP = 1

# MADDPG algorithm hyperparameters
GAMMA = 0.99
TARGET_UPDATE_TAU = 0.005
LEARNING_RATE_ACTOR = 3e-5
LEARNING_RATE_CRITIC = 1e-4

# Exploration noise parameters
NOISE_DECAY_STEPS = NUM_EPISODES * env.max_steps * 0.8
INITIAL_NOISE_SCALE = 0.3
FINAL_NOISE_SCALE = 0.02
NOISE_DECAY = True

# Neural network hyperparameters
HIDDEN_DIM = 256

# Device settings
DEVICE = torch.device("cpu")  # Force CPU

# Model saving settings
SAVE_MODELS = True
MODEL_SAVE_DIR = "models_maddpg"
BEST_MODEL_THRESHOLD = -np.inf

# Initialization
print(f"Using device: {DEVICE}")
print(f"Starting MADDPG training: {SCENARIO}, {NUM_AGENTS} agents, {NUM_EPISODES} episodes.")
if NOISE_DECAY:
    print(
        f"Noise decay enabled: Linear decay from {INITIAL_NOISE_SCALE:.2f} to {FINAL_NOISE_SCALE:.2f} over {NOISE_DECAY_STEPS} steps."
    )
else:
    print(f"Noise decay disabled. Using fixed noise scale: {INITIAL_NOISE_SCALE:.2f}")

# Environment information
STATE_DIM = env.state_dimension
ACTION_DIM = env.action_dimension
MAX_ACTION = 1.0  # Assuming actor output is in [-1, 1]
print(f"Single agent observation dimension: {STATE_DIM}")
print(f"Single agent action dimension: {ACTION_DIM}")
print(f"Action range: [-{MAX_ACTION}, {MAX_ACTION}]")
print(f"Max steps per episode: {env.max_steps}")

# Directory setup for saving models
model_dir = None
timestamp = time.strftime("%Y%m%d-%H%M%S")
if SAVE_MODELS:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(script_dir, MODEL_SAVE_DIR, SCENARIO)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path_prefix = os.path.join(model_dir, f"MADDPG_{SCENARIO}_{timestamp}")  # Model filename prefix
    print(f"Models will be saved in directory: {model_dir}")
else:
    print("Model saving disabled.")

# Agent setup
agent = MADDPGAgent(
    num_agents=NUM_AGENTS,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    hidden_dim=HIDDEN_DIM,
    lr_actor=LEARNING_RATE_ACTOR,
    lr_critic=LEARNING_RATE_CRITIC,
    gamma=GAMMA,
    tau=TARGET_UPDATE_TAU,
    device=DEVICE,
    buffer_capacity=BUFFER_SIZE,
    initial_noise_scale=INITIAL_NOISE_SCALE,
    final_noise_scale=FINAL_NOISE_SCALE,
    noise_decay_steps=NOISE_DECAY_STEPS if NOISE_DECAY else 0,
    max_action=MAX_ACTION,
)

# Tracking variables
reward_history = []  # Record shared total reward per episode
total_steps_elapsed = 0  # Record total timesteps
best_avg_reward = -float("inf")
current_lr_actor = agent.initial_lr_actor
current_lr_critic = agent.initial_lr_critic
current_noise = agent.initial_noise_scale

# Training Loop
# Record time to reach reward thresholds
reward_time_dict = {}  # {threshold: (episode, elapsed_time)}
last_recorded_threshold = None
threshold_step = 50

start_time = time.time()
time_1000_episodes = None
time_5000_episodes = None

for episode in range(1, NUM_EPISODES + 1):
    states_list = env.reset()  # Get initial states list
    episode_reward_sum = 0  # Cumulative shared reward for the episode
    episode_steps = 0  # Timesteps in the episode
    is_terminal = False  # Shared terminal flag for the episode

    while not is_terminal and episode_steps < env.max_steps:
        # Agent gets actions (with noise) based on states and current step
        actions_list = agent.get_action(states_list, total_steps_elapsed, evaluate=False)

        # Environment executes actions
        next_states_list, rewards_list, dones_list, infos_list = env.step(actions_list)

        # Process environment returns (assuming shared reward and done state)
        shared_reward = rewards_list[0] if rewards_list else 0.0
        is_terminal = dones_list[0] if dones_list else True

        # Store experience in replay buffer (once per timestep)
        agent.store_transition(states_list, actions_list, shared_reward, next_states_list, is_terminal)

        # Update states
        states_list = next_states_list
        # Accumulate reward
        episode_reward_sum += shared_reward
        # Increment total and episode timesteps
        total_steps_elapsed += 1
        episode_steps += 1

        # Check if learning should start and perform updates
        if total_steps_elapsed >= LEARNING_START:
            for _ in range(UPDATES_PER_STEP):
                current_lr_actor, current_lr_critic, current_noise = agent.update(BATCH_SIZE)

        # If environment terminates early, break loop
        if is_terminal:
            break

    # End of episode
    reward_history.append(episode_reward_sum)
    current_avg_reward = (
        np.mean(reward_history[-REWARD_AVG_WINDOW:])
        if len(reward_history) >= REWARD_AVG_WINDOW
        else np.mean(reward_history) if reward_history else -float("inf")
    )

    # Record time to reach reward threshold
    # Consider multiples of threshold_step (positive and negative)
    threshold = int(np.floor(episode_reward_sum / threshold_step)) * threshold_step
    if (last_recorded_threshold is None or threshold != last_recorded_threshold) and threshold not in reward_time_dict:
        elapsed_time = time.time() - start_time
        reward_time_dict[threshold] = (episode, elapsed_time)
        last_recorded_threshold = threshold

    # Record time for 1000 episodes
    if episode == 1000 and time_1000_episodes is None:
        time_1000_episodes = time.time() - start_time
    # Record time for 5000 episodes
    if episode == 5000 and time_5000_episodes is None:
        time_5000_episodes = time.time() - start_time

    # Logging
    if episode % PRINT_EVERY == 0 or episode == NUM_EPISODES:
        elapsed_time = time.time() - start_time
        # Use initial values for LR and Noise if learning hasn't started
        log_lr_a = current_lr_actor if total_steps_elapsed >= LEARNING_START else agent.initial_lr_actor
        log_lr_c = current_lr_critic if total_steps_elapsed >= LEARNING_START else agent.initial_lr_critic
        log_noise = current_noise if total_steps_elapsed >= LEARNING_START else agent.initial_noise_scale

        print(
            f"Episode {episode}/{NUM_EPISODES} | Steps: {episode_steps} | Total Steps: {total_steps_elapsed} | "
            f"Ep. Reward: {episode_reward_sum:.2f} | Avg. Reward ({min(len(reward_history), REWARD_AVG_WINDOW)}): {current_avg_reward:.2f} | "
            f"LR (A/C): {log_lr_a:.1e}/{log_lr_c:.1e} | "
            f"Noise Scale: {log_noise:.1e} | "
            f"Buffer: {len(agent.replay_buffer)}/{int(BUFFER_SIZE)} | "
            f"Elapsed: {elapsed_time:.1f}s"
        )

# End of Training
print(f"\nTraining finished. Total {NUM_EPISODES} episodes, {total_steps_elapsed} total timesteps.")

final_avg_reward = (
    np.mean(reward_history[-REWARD_AVG_WINDOW:])
    if len(reward_history) >= REWARD_AVG_WINDOW
    else np.mean(reward_history) if reward_history else -float("inf")
)
print(f"Average reward over last {min(len(reward_history), REWARD_AVG_WINDOW)} episodes: {final_avg_reward:.2f}")

# Save final model
if SAVE_MODELS and model_dir is not None:
    final_model_prefix = os.path.join(model_dir, f"MADDPG_{SCENARIO}_{timestamp}_final_reward_{final_avg_reward:.2f}")
    actor_final_path = final_model_prefix + "_actor.pth"
    critic_final_path = final_model_prefix + "_critic.pth"
    try:
        agent.save_model(actor_final_path, critic_final_path)
        # Agent prints save message
    except Exception as e:
        print(f"Error saving final model: {e}")

# Cleanup (environment doesn't need explicit close here)

# Plot the reward curve
if SAVE_MODELS and model_dir is not None:
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(reward_history, label="Episode Rewards", alpha=0.6)

        if len(reward_history) >= REWARD_AVG_WINDOW:
            try:
                import pandas as pd  # Optional: for smooth moving average

                moving_avg = pd.Series(reward_history).rolling(REWARD_AVG_WINDOW, min_periods=1).mean()
                plt.plot(moving_avg, label=f"Moving Average ({REWARD_AVG_WINDOW} Episodes)", color="red", linewidth=2)
            except ImportError:
                print("Pandas not found. Skipping plotting the moving average line.")
                avg_rew = np.mean(reward_history[-REWARD_AVG_WINDOW:])
                plt.axhline(
                    avg_rew,
                    color="red",
                    linestyle="--",
                    label=f"Average of the last {REWARD_AVG_WINDOW} Episodes: {avg_rew:.2f}",
                )
        elif reward_history:  # If less than REWARD_AVG_WINDOW episodes but some history exists
            avg_rew = np.mean(reward_history)
            plt.axhline(avg_rew, color="red", linestyle="--", label=f"Overall Average Reward: {avg_rew:.2f}")

        plt.title(f"MADDPG Training Rewards - {SCENARIO} ({NUM_AGENTS} Agents)")
        plt.xlabel("Episodes")
        plt.ylabel("Cumulative Shared Rewards")
        plt.legend()
        plt.grid(True)

        # Save the plot with a filename that includes the final reward
        plot_path = final_model_prefix + "_rewards.png"
        plt.savefig(plot_path)
        print(f"Reward curve plot saved to: {plot_path}")
        plt.close()
    except ImportError:  # Matplotlib or Pandas might not be installed
        print("Matplotlib or Pandas not found. Skipping plotting the reward curve.")
    except Exception as e:
        print(f"Error occurred while plotting the reward curve: {e}")

# Save reward_history to file
reward_file_path = os.path.join(os.path.dirname(__file__), "DDPGreward.txt")
try:
    with open(reward_file_path, "w") as f:
        for reward in reward_history:
            f.write(f"{reward}\n")
    print(f"Reward history saved to: {reward_file_path}")  # English print
except Exception as e:
    print(f"Error saving reward history: {e}")  # English print

# Save time_cost.txt
time_cost_path = os.path.join(os.path.dirname(__file__), "time_cost.txt")
try:
    with open(time_cost_path, "w") as f:
        for th in sorted(reward_time_dict.keys()):  # Iterate sorted thresholds
            ep, t = reward_time_dict[th]
            f.write(f"reward>={th}: episode={ep}, time={t:.2f}s\n")
        if time_1000_episodes is not None:
            f.write(f"\ncost_1000_episodes: {time_1000_episodes:.2f}s\n")  # Clarified key
        if time_5000_episodes is not None:
            f.write(f"cost_5000_episodes: {time_5000_episodes:.2f}s\n")  # Clarified key
    print(f"time_cost.txt saved to: {time_cost_path}")  # English print
except Exception as e:
    print(f"Error saving time_cost.txt: {e}")  # English print

print("Script execution completed.")
