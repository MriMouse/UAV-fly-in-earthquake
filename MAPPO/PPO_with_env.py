import os
import sys
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from MAPPO_agent import MAPPOAgent

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow duplicate KMP libraries

# ==================================
#          Configuration Parameters
# ==================================

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from env import Env_MAPPO  # Import environment


# Environment settings
SCENARIO = "MAPPO_UAV_UGV_Cooperation"  # Scenario name
env = Env_MAPPO()  # Instantiate MAPPO environment
NUM_AGENTS = env.num_uavs  # Get number of agents from environment

# Training loop settings
NUM_EPISODES = 10000  # Total training episodes
PRINT_EVERY = 50  # Print log every N episodes
REWARD_AVG_WINDOW = 100  # Window size for averaging rewards

# MAPPO algorithm hyperparameters
UPDATE_INTERVAL = 2048  # Agent update frequency (in timesteps)
PPO_EPOCHS = 10  # PPO training epochs per update
MINIBATCH_SIZE = 128  # PPO minibatch size (in timesteps)
GAMMA = 0.99  # Discount factor
LAMBDA = 0.95  # Lambda for GAE (Generalized Advantage Estimation)
PPO_EPSILON = 0.15  # PPO clipping epsilon
LEARNING_RATE_ACTOR = 1e-5  # Actor network learning rate
LEARNING_RATE_CRITIC = 1e-4  # Critic network learning rate

# --- Learning rate decay ---
TOTAL_TRAINING_STEPS_FOR_DECAY = NUM_EPISODES * env.max_steps  # Total steps for LR decay
LR_DECAY = True  # Enable learning rate decay
FINAL_LR_FACTOR = 0.1  # Final LR factor relative to initial LR

# --- Entropy decay parameters ---
ENTROPY_DECAY = True  # Enable entropy decay
INITIAL_ENTROPY_COEF = 0.02  # Initial entropy coefficient
FINAL_ENTROPY_COEF = 0.001  # Final entropy coefficient
# ---

# Neural network hyperparameters
HIDDEN_DIM = 256  # Hidden layer dimension

# Device settings (CPU or GPU)
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
DEVICE = torch.device("cpu")  # Force CPU usage

# Model saving settings
SAVE_MODELS = True  # Save final models after training
MODEL_SAVE_DIR = "models_MAPPO"  # Root directory for saving models

print(f"Using device: {DEVICE}")
print(f"Starting MAPPO training: {SCENARIO}, {NUM_AGENTS} agents, {NUM_EPISODES} episodes.")
if LR_DECAY:
    print(
        f"Learning rate decay enabled: Linear decay to factor {FINAL_LR_FACTOR} over {TOTAL_TRAINING_STEPS_FOR_DECAY} steps"
    )
else:
    print("Learning rate decay disabled.")
if ENTROPY_DECAY:
    print(
        f"Entropy decay enabled: Linear decay from {INITIAL_ENTROPY_COEF} to {FINAL_ENTROPY_COEF} over {TOTAL_TRAINING_STEPS_FOR_DECAY} steps"
    )
else:
    print(f"Entropy decay disabled. Using fixed coefficient: {INITIAL_ENTROPY_COEF}")

# --- Environment information ---
STATE_DIM = env.state_dimension  # Single agent state space dimension
ACTION_DIM = env.action_dimension  # Single agent action space dimension
print(f"Single agent observation dimension: {STATE_DIM}")
print(f"Single agent action dimension: {ACTION_DIM}")
print(f"Max steps per episode: {env.max_steps}")

# --- Directory settings ---
model_dir = None
timestamp = time.strftime("%Y%m%d-%H%M%S")  # Timestamp for model saving
if SAVE_MODELS:
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get script directory
    model_dir = os.path.join(script_dir, MODEL_SAVE_DIR, SCENARIO)  # Define model directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Create model directory if it doesn't exist
    print(f"Models will be saved in directory: {model_dir}")
else:
    print("Model saving disabled.")


# --- Agent settings ---
agent = MAPPOAgent(  # Initialize MAPPO agent
    num_agents=NUM_AGENTS,
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    hidden_dim=HIDDEN_DIM,
    lr_actor=LEARNING_RATE_ACTOR,
    lr_critic=LEARNING_RATE_CRITIC,
    gamma=GAMMA,
    gae_lambda=LAMBDA,
    ppo_epsilon=PPO_EPSILON,
    ppo_epochs=PPO_EPOCHS,
    minibatch_size=MINIBATCH_SIZE,
    device=DEVICE,
    total_training_steps=TOTAL_TRAINING_STEPS_FOR_DECAY,
    lr_decay=LR_DECAY,
    final_lr_factor=FINAL_LR_FACTOR,
    initial_entropy_coef=INITIAL_ENTROPY_COEF,
    final_entropy_coef=FINAL_ENTROPY_COEF,
    entropy_decay=ENTROPY_DECAY,
    action_scale=1.0,
    max_grad_norm=0.5,
)

# --- Tracking variables ---
reward_history = []  # Record shared total reward per episode
total_steps_elapsed = 0  # Record total timesteps
current_lr_actor = agent.initial_lr_actor  # Current actor learning rate
current_lr_critic = agent.initial_lr_critic  # Current critic learning rate
current_entropy_coef = agent.initial_entropy_coef  # Current entropy coefficient

# ==================================
#          Training Loop
# ==================================

start_time = time.time()  # Record training start time

reward_time_dict = {}  # Dictionary to store {threshold: (episode, elapsed_time)}
last_recorded_threshold = None  # Last recorded reward threshold
threshold_step = 50  # Step for reward thresholds
time_1000_episodes = None  # Time taken for 1000 episodes
start_time = time.time()  # Record training start time again (this might be redundant)
time_5000_episodes = None  # Time taken for 5000 episodes

for episode in range(1, NUM_EPISODES + 1):  # Loop through episodes
    states_list = env.reset()  # Reset environment, get initial states
    episode_reward_sum = 0  # Cumulative shared reward for current episode
    episode_steps = 0  # Timesteps in current episode
    is_terminal = False  # Shared terminal flag for the episode

    while not is_terminal and episode_steps < env.max_steps:  # Loop within an episode
        # 1. Agent selects actions based on current states
        #    Also get centralized value estimate and log probabilities of actions
        actions_list, centralized_value, log_probs_list = agent.get_action(states_list)

        # 2. Environment executes actions, returns new states, rewards, dones, etc.
        next_states_list, rewards_list, dones_list, infos_list = env.step(actions_list)

        # 3. Process environment returns (assuming shared reward and done state)
        shared_reward = rewards_list[0]  # Use first reward as shared reward
        is_terminal = dones_list[0]  # Use first done flag as shared done flag

        # 4. Store experience in replay buffer (once per timestep)
        #    Store: states_list before action, actions_list, shared_reward,
        #    centralized_value before action, and is_terminal after action
        agent.replay_buffer.add_memo(states_list, actions_list, shared_reward, centralized_value, is_terminal)

        # Update states
        states_list = next_states_list
        # Accumulate shared reward
        episode_reward_sum += shared_reward
        # Accumulate total timesteps
        total_steps_elapsed += 1
        # Accumulate episode timesteps
        episode_steps += 1

        # 5. Check if it's time to update the agent (buffer size, by timesteps)
        if agent.replay_buffer.size() >= UPDATE_INTERVAL:
            last_centralized_value = 0.0  # Value of last state is 0 if episode ended during update
            # If episode not ended, estimate centralized value of the last state for GAE
            if not is_terminal:
                with torch.no_grad():  # No gradient calculation needed
                    # Prepare centralized state tensor
                    states_tensor_list = [torch.FloatTensor(s).unsqueeze(0).to(DEVICE) for s in states_list]
                    centralized_state_tensor = torch.cat(states_tensor_list, dim=1)
                    # Estimate value using Critic network
                    last_centralized_value = agent.critic(centralized_state_tensor).cpu().item()

            # 6. Call agent's update method for training
            current_lr_actor, current_lr_critic, current_entropy_coef = agent.update(
                last_centralized_value, total_steps_elapsed
            )
            # update method clears the buffer internally

        # If terminated due to environment reasons (e.g., out of battery), break loop
        if is_terminal:
            break

    # --- End of episode ---
    reward_history.append(episode_reward_sum)  # Record cumulative shared reward for this episode
    current_avg_reward = (
        np.mean(reward_history[-REWARD_AVG_WINDOW:]) if reward_history else -float("inf")
    )  # Calculate current average reward

    # --- Record time to reach reward thresholds ---
    threshold = int(np.floor(episode_reward_sum / threshold_step)) * threshold_step  # Calculate current threshold
    if (last_recorded_threshold is None or threshold != last_recorded_threshold) and threshold not in reward_time_dict:
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        reward_time_dict[threshold] = (episode, elapsed_time)  # Store episode and time for threshold
        last_recorded_threshold = threshold  # Update last recorded threshold

    # --- Record time for 1000 episodes ---
    if episode == 1000 and time_1000_episodes is None:
        time_1000_episodes = time.time() - start_time

    # --- Record time for 5000 episodes ---
    if episode == 5000 and time_5000_episodes is None:
        time_5000_episodes = time.time() - start_time

    # --- Log printing ---
    if episode % PRINT_EVERY == 0 or episode == NUM_EPISODES:  # Print every PRINT_EVERY episodes or at the last episode
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(
            f"Episode {episode}/{NUM_EPISODES} | Steps: {episode_steps} | Total Steps: {total_steps_elapsed} | "
            f"Episode Reward: {episode_reward_sum:.2f} | Avg Reward ({min(len(reward_history), REWARD_AVG_WINDOW)}): {current_avg_reward:.2f} | "
            f"LR (A/C): {current_lr_actor:.1e}/{current_lr_critic:.1e} | "
            f"Entropy Coef: {current_entropy_coef:.1e} | "
            f"Elapsed Time: {elapsed_time:.1f}s"
        )


print(f"\nTraining finished. Total {NUM_EPISODES} episodes, {total_steps_elapsed} total timesteps.")

final_avg_reward = (  # Calculate final average reward
    np.mean(reward_history[-REWARD_AVG_WINDOW:])
    if len(reward_history) >= REWARD_AVG_WINDOW
    else np.mean(reward_history) if reward_history else -float("inf")
)
print(f"Average reward of last {min(len(reward_history), REWARD_AVG_WINDOW)} episodes: {final_avg_reward:.2f}")

# --- Save final model ---
if SAVE_MODELS and model_dir is not None:  # If model saving is enabled and directory exists
    final_model_prefix = os.path.join(
        model_dir, f"MAPPO_{SCENARIO}_{timestamp}_final_reward_{final_avg_reward:.2f}"
    )  # Define model file prefix
    actor_final_path = final_model_prefix + "_actor.pth"  # Actor model path
    critic_final_path = final_model_prefix + "_critic.pth"  # Critic model path
    try:
        agent.save_model(actor_final_path, critic_final_path)  # Save models
        # print(f"Final models saved: Actor '{actor_final_path}', Critic '{critic_final_path}'") # Already printed in save_model
    except Exception as e:
        print(f"Error saving final models: {e}")

# --- Cleanup ---
# Environment does not need explicit closing

# --- Plot the reward curve ---
if SAVE_MODELS and model_dir is not None:  # If model saving is enabled and directory exists
    try:
        plt.figure(figsize=(12, 6))  # Create a new figure
        plt.plot(reward_history, label="Episode Rewards", alpha=0.6)  # Plot episode rewards

        if len(reward_history) >= REWARD_AVG_WINDOW:  # If enough history for moving average
            try:
                import pandas as pd  # Try importing pandas for moving average

                moving_avg = (
                    pd.Series(reward_history).rolling(REWARD_AVG_WINDOW, min_periods=1).mean()
                )  # Calculate moving average
                plt.plot(
                    moving_avg, label=f"Moving Average ({REWARD_AVG_WINDOW} Episodes)", color="red", linewidth=2
                )  # Plot moving average
            except ImportError:
                print("Pandas not found. Skipping plotting the moving average line.")  # Pandas not found
                avg_rew = np.mean(reward_history[-REWARD_AVG_WINDOW:])  # Calculate average of last window
                plt.axhline(  # Plot average line
                    avg_rew,
                    color="red",
                    linestyle="--",
                    label=f"Average of the last {REWARD_AVG_WINDOW} Episodes: {avg_rew:.2f}",
                )
        elif reward_history:  # If some reward history exists but not enough for window
            avg_rew = np.mean(reward_history)  # Calculate overall average
            plt.axhline(
                avg_rew, color="red", linestyle="--", label=f"Overall Average Reward: {avg_rew:.2f}"
            )  # Plot overall average

        plt.title(f"MAPPO Training Rewards - {SCENARIO} ({NUM_AGENTS} Agents)")  # Set plot title
        plt.xlabel("Episodes")  # Set x-axis label
        plt.ylabel("Cumulative Shared Rewards")  # Set y-axis label
        plt.legend()  # Show legend
        plt.grid(True)  # Show grid

        plot_path = final_model_prefix + "_rewards.png"  # Define plot save path
        plt.savefig(plot_path)  # Save plot
        print(f"Reward curve plot saved to: {plot_path}")
        plt.close()  # Close plot figure
    except ImportError:
        print("Matplotlib or Pandas not found. Skipping plotting the reward curve.")  # Matplotlib/Pandas not found
    except Exception as e:
        print(f"Error occurred while plotting the reward curve: {e}")  # Other plotting error

# Save reward_history to file
reward_file_path = os.path.join(os.path.dirname(__file__), "PPOreward.txt")  # Define reward history file path
try:
    with open(reward_file_path, "w") as f:  # Open file for writing
        for reward in reward_history:  # Write each reward to a new line
            f.write(f"{reward}\n")
    print(f"Reward history saved to: {reward_file_path}")
except Exception as e:
    print(f"Error saving reward history: {e}")


# Save time_cost.txt, format consistent with MADDPG
time_cost_path = os.path.join(os.path.dirname(__file__), "time_cost.txt")  # Define time cost file path
try:
    with open(time_cost_path, "w") as f:  # Open file for writing
        f.write("# Episodes and time (seconds) to reach reward thresholds (step of 50)\n")  # Header
        for th in sorted(reward_time_dict.keys()):  # Iterate through sorted reward thresholds
            ep, t = reward_time_dict[th]  # Get episode and time
            f.write(f"reward>={th}: episode={ep}, time={t:.2f}s\n")  # Write threshold info
        if time_1000_episodes is not None:  # If time for 1000 episodes recorded
            f.write(f"\nTime for 1000 episodes: {time_1000_episodes:.2f}s\n")
        if time_5000_episodes is not None:  # If time for 5000 episodes recorded
            f.write(f"Time for 5000 episodes: {time_5000_episodes:.2f}s\n")
    print(f"time_cost.txt saved to: {time_cost_path}")
except Exception as e:
    print(f"Error saving time_cost.txt: {e}")

print("Script execution completed.")
