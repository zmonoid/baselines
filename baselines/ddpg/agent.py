import copy
import numpy as np
from collections import deque

import torch
import torch.nn.functional as F


class RunningStat:
    def __init__(self, xrange=None):
        self.sum = 0
        self.sumsq = 1e-2
        self.count = 1e-2
        self.xrange = xrange

    def std(self):
        return np.sqrt(np.clip(self.sumsq / self.count - self.mean() ** 2, 1e-2, np.inf))

    def mean(self):
        return self.sum / self.count

    def update(self, x):
        self.sum += x
        self.sumsq += np.square(x)
        self.count += 1

    def normalize(self, x):
        x = (x - self.mean()) / self.std()
        if self.xrange is not None:
            x = np.clip(x, self.xrange[0], self.xrange[1])
        return x

    def denormalize(self, x):
        if self.xrange is not None:
            x = np.clip(x, self.xrange[0], self.xrange[1])
        return x * self.std() + self.mean()


class Agent(object):
    def __init__(self, actor, critic, memory, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1., cuda=True):

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.cuda = cuda

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.critic_l2_reg = critic_l2_reg


        self.critic = critic
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.critic.eval()
        self.actor.eval()
        self.critic_target.eval()
        self.actor_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.running_obs = RunningStat(observation_range)

    def step(self, obs):
        obs = self.running_obs.normalize(obs)

        obs = torch.from_numpy(obs).float()
        if self.cuda:
            obs = obs.cuda()

        action = self.actor(obs)

        if self.param_noise is not None:

            self.actor.set_sigma(self.param_noise.current_stddev)
            action_noisy = self.actor(obs)
            self.actor.set_sigma(0)

            action = action_noisy

        q = self.critic(torch.cat([obs, action], dim=-1))
        q = q.detach().cpu().numpy()

        if self.action_noise is not None:

            noise = torch.from_numpy(self.action_noise.noise())
            if self.cuda:
                noise = noise.cuda()

            action += noise

        action = action.detach().cpu().numpy()
        action = np.clip(action, *self.action_range)

        return action, q, None, None

    def train(self):

        self.critic.train()
        self.actor.train()

        batch = self.memory.sample(self.batch_size)

        obs0 = self.running_obs.normalize(batch['obs0'])
        obs1 = self.running_obs.normalize(batch['obs1'])
        obs0 = torch.from_numpy(obs0).float()
        obs1 = torch.from_numpy(obs1).float()

        terminals = torch.from_numpy(batch['terminals'])
        actions = torch.from_numpy(batch['actions'])
        rewards = torch.from_numpy(batch['rewards'])

        if self.cuda:
            obs0 = obs0.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            obs1 = obs1.cuda()
            terminals = terminals.cuda()

        action_value = self.actor(obs0)
        state_action_value = self.critic(torch.cat([obs0, actions], dim=-1)).clamp(*self.return_range)
        state_action_value_from_actor = self.critic(torch.cat([obs0, action_value], dim=-1)).clamp(*self.return_range)

        target_action_value = self.actor_target(obs1)
        target_state_action_value = self.critic_target(torch.cat([obs1, target_action_value], dim=-1))

        rhs = rewards + (1 - terminals) * self.gamma * target_state_action_value
        value_loss = F.mse_loss(state_action_value, rhs.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.clip_norm)
        self.critic_optimizer.step()

        policy_loss = 0 - state_action_value_from_actor.mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.clip_norm)
        self.actor_optimizer.step()

        self.soft_sync(self.actor_target, self.actor, self.tau)
        self.soft_sync(self.critic_target, self.critic, self.tau)

        self.critic.eval()
        self.actor.eval()

        return value_loss.item(), policy_loss.item()

    def store_transition(self, obs0, action, reward, obs1, terminal):
        reward *= self.reward_scale
        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal[b])
            if self.normalize_observations:
                self.running_obs.update(obs0[b])

    def soft_sync(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def adapt_actor_param_noise(self):
        batch = self.memory.sample(batch_size=self.batch_size)
        obs0 = self.running_obs.normalize(batch['obs0'])
        obs0 = torch.from_numpy(obs0).float()
        if self.cuda:
            obs0 = obs0.cuda()

        action = self.actor(obs0)
        self.actor.set_sigma(self.param_noise.current_stddev)
        action_noisy = self.actor(obs0)
        self.actor.set_sigma(0)

        dist = (action_noisy - action).pow(2).mean().sqrt().item()
        self.param_noise.adapt(dist)
        return dist


    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()


    def get_stats(self):
        out = {}
        out['obs_rms_mean'] = self.running_obs.mean().mean()
        out['obs_rms_std'] = self.running_obs.std().mean()
        out['ref_action_mean'] = 0
        out['param_noise_stddev'] = self.param_noise.current_stddev
        return out


