#!/usr/bin/env python
# coding: utf-8

# # CartPole  (standard)

# In[38]:


import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[39]:


class CartPoleV1(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(CartPoleV1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# In[40]:


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs]).to(device)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.cpu().detach().numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        is_done = terminated or truncated

        episode_steps.append({"observation": obs, "action": action})
        episode_reward += reward

        if is_done:
            batch.append({"reward": episode_reward, "steps": episode_steps})
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = [ep["reward"] for ep in batch]
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for ep in batch:
        if ep["reward"] < reward_bound:
            continue
        train_obs.extend([step["observation"] for step in ep["steps"]])
        train_act.extend([step["action"] for step in ep["steps"]])

    obs_v = torch.FloatTensor(train_obs).to(device)
    act_v = torch.LongTensor(train_act).to(device)
    return obs_v, act_v, reward_bound, reward_mean


def evaluate_policy(net, env_name="CartPole-v1", episodes=5, render=True, save_gif=False,
                    gif_path="cartpole_evaluation.gif"):
    if save_gif:
        from PIL import Image
        frames = []

    env = gym.make(env_name, render_mode="rgb_array" if save_gif else "human")
    net.eval()

    episode_rewards = []

    print(f"Starting evaluation with {episodes} episodes...")

    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            # Render frame
            if render and not save_gif:
                env.render()

            # Save frame for GIF
            if save_gif:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            # Get action from policy
            with torch.no_grad():
                obs_v = torch.FloatTensor([obs]).to(device)
                action_probs = torch.softmax(net(obs_v), dim=1)
                action = torch.argmax(action_probs, dim=1).item()

            # Take action
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            is_done = terminated or truncated

            if is_done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Reward = {episode_reward:.1f}, Steps = {step_count}")

    env.close()

    # Save GIF if requested
    if save_gif and frames:
        # Add extra frames at the end to make it pause
        for _ in range(10):
            frames.append(frames[-1])

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # milliseconds per frame
            loop=0  # infinite loop
        )
        print(f"GIF saved to: {gif_path}")

    avg_reward = np.mean(episode_rewards)
    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Rewards: {[f'{r:.1f}' for r in episode_rewards]}")

    return episode_rewards


# ## Training loop

# In[41]:


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
MAX_ITER = 100

# setup
env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

net = CartPoleV1(obs_size, HIDDEN_SIZE, n_actions).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# history
episode_rewards = []
episode_losses = []


# In[42]:


try:
    print("Training started...")
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        optimizer.zero_grad()
        action_scores = net(obs_v)
        loss = criterion(action_scores, acts_v)
        loss.backward()
        optimizer.step()

        episode_rewards.append(reward_m)
        episode_losses.append(loss.item())

        print(f"Iter {iter_no}: loss={loss.item():.3f}, "
              f"reward_mean={reward_m:.1f}, reward_bound={reward_b:.1f}")

        if iter_no >= MAX_ITER:
            print("Max iterations reached")
            break
finally:
    env.close()

    ## Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: Reward
    color = 'tab:blue'
    ax1.set_xlabel('Iteration (Batch)')
    ax1.set_ylabel('Mean Reward', color=color)
    ax1.plot(episode_rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Right y-axis: Loss
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(episode_losses, color=color, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and layout
    plt.title('Training Progress: Reward and Loss vs Iteration')
    fig.tight_layout()
    plt.show()


# ## Evaluate
# 
# ![](video/cartpole_trained_agent.gif)

# In[43]:


evaluate_policy(net, episodes=2, render=True, save_gif=True, gif_path="video/cartpole_trained_agent.gif")


# # CartPole (start with rand position)

# In[45]:


# CartPole  (standard)
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[46]:


class CartPoleV2(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(CartPoleV2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# In[47]:


def reset_with_random_angle(env, max_angle_deg=10.0, max_position=0.5):
    obs, info = env.reset()
    max_angle_rad = np.deg2rad(max_angle_deg)
    state = env.unwrapped.state.copy()  # [cart_pos, cart_vel, pole_angle, pole_ang_vel]

    # Randomize position and angle
    state[0] = np.random.uniform(-max_position, max_position)  # cart position
    state[2] = np.random.uniform(-max_angle_rad, max_angle_rad)  # pole angle

    # Keep velocities zero (standard practice)
    state[1] = 0.0  # cart velocity
    state[3] = 0.0  # pole angular velocity

    env.unwrapped.state = state
    obs = state.copy()

    return obs, info


def iterate_batches(env, net, batch_size, max_angle_deg=10.0):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = reset_with_random_angle(env, max_angle_deg=max_angle_deg)
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.FloatTensor([obs]).to(device)
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.cpu().detach().numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        is_done = terminated or truncated

        episode_steps.append({"observation": obs, "action": action})
        episode_reward += reward

        if is_done:
            batch.append({"reward": episode_reward, "steps": episode_steps})
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = reset_with_random_angle(env, max_angle_deg=max_angle_deg)
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = [ep["reward"] for ep in batch]
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for ep in batch:
        if ep["reward"] < reward_bound:
            continue
        train_obs.extend([step["observation"] for step in ep["steps"]])
        train_act.extend([step["action"] for step in ep["steps"]])

    obs_v = torch.FloatTensor(train_obs).to(device)
    act_v = torch.LongTensor(train_act).to(device)
    return obs_v, act_v, reward_bound, reward_mean


def evaluate_policy(net, env_name="CartPole-v1", episodes=5, render=True, save_gif=False,
                    gif_path="cartpole_evaluation.gif", max_angle_deg=12.0):
    if save_gif:
        from PIL import Image
        frames = []

    env = gym.make(env_name, render_mode="rgb_array" if save_gif else "human")
    net.eval()

    episode_rewards = []

    print(f"Starting evaluation with {episodes} episodes...")

    for ep in range(episodes):
        obs, _ = reset_with_random_angle(env, max_angle_deg=max_angle_deg)
        episode_reward = 0
        step_count = 0

        while True:
            if render and not save_gif:
                env.render()

            if save_gif:
                frame = env.render()
                if frame is not None:
                    frames.append(Image.fromarray(frame))

            with torch.no_grad():
                obs_v = torch.FloatTensor([obs]).to(device)
                action_probs = torch.softmax(net(obs_v), dim=1)
                action = torch.argmax(action_probs, dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            is_done = terminated or truncated

            if is_done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: Reward = {episode_reward:.1f}, Steps = {step_count}")

    env.close()

    # Save GIF
    if save_gif and frames:
        # Add extra frames at the end to make it pause
        for _ in range(10):
            frames.append(frames[-1])

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=50,  # milliseconds per frame
            loop=0  # infinite loop
        )
        print(f"GIF saved to: {gif_path}")

    avg_reward = np.mean(episode_rewards)
    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Rewards: {[f'{r:.1f}' for r in episode_rewards]}")

    return episode_rewards


# ## Training loop

# In[48]:


HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70
MAX_ITER = 100
TRAIN_MAX_ANGLE = 10.0
EVAL_MAX_ANGLE = 15.0

# setup
env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

net = CartPoleV2(obs_size, HIDDEN_SIZE, n_actions).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# history
episode_rewards = []
episode_losses = []


# In[49]:


try:
    print("Training started...")
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE, max_angle_deg=TRAIN_MAX_ANGLE)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        optimizer.zero_grad()
        action_scores = net(obs_v)
        loss = criterion(action_scores, acts_v)
        loss.backward()
        optimizer.step()

        episode_rewards.append(reward_m)
        episode_losses.append(loss.item())

        print(f"Iter {iter_no}: loss={loss.item():.3f}, "
              f"reward_mean={reward_m:.1f}, reward_bound={reward_b:.1f}")

        if iter_no >= MAX_ITER:
            print("Max iterations reached")
            break
finally:
    env.close()

    ## Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: Reward
    color = 'tab:blue'
    ax1.set_xlabel('Iteration (Batch)')
    ax1.set_ylabel('Mean Reward', color=color)
    ax1.plot(episode_rewards, color=color, label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Right y-axis: Loss
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(episode_losses, color=color, label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and layout
    plt.title('Training Progress: Reward and Loss vs Iteration')
    fig.tight_layout()
    plt.show()


# ## Evaluate
# 
# ![](video/cartpole_trained_agent-2.gif)
# ![](video/cartpole_trained_agent-2b.gif)

# In[51]:


evaluate_policy(net, episodes=2, render=True, save_gif=True, gif_path="video/cartpole_trained_agent-2.gif", max_angle_deg=EVAL_MAX_ANGLE)


# In[52]:


evaluate_policy(net, episodes=2, render=True, save_gif=True, gif_path="video/cartpole_trained_agent-2b.gif", max_angle_deg=EVAL_MAX_ANGLE*2)


# # CartPole (start from bottom)

# In[ ]:




