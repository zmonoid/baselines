"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= act (in case of parameter noise) ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon to a new value, if negative no update happens
        (default: no update)
    reset_ph: bool
        reset the perturbed policy by sampling a new perturbation
    update_param_noise_threshold_ph: float
        the desired threshold for the difference between non-perturbed and perturbed policy
    update_param_noise_scale_ph: bool
        whether or not to update the scale of the noise for the next time it is re-perturbed

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import torch
import torch.nn.functional as F
import copy
import numpy as np


class Agent:
    def __init__(self, q_model, actions_space, lr, scale=255.0):
        self.actions_space = actions_space
        self.q_model = q_model
        self.noisy_q_model = None
        self.target_q_model = copy.deepcopy(q_model)
        self.cuda = torch.cuda.is_available()
        self.scale = scale
        self.sigma = 0.01

        if self.cuda:
            self.q_model.cuda()
            self.target_q_model.cuda()

        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=lr)

    def act(self, obs, epsilon):
        if np.random.rand() > epsilon:
            obs = np.array([obs])
            if obs.ndim == 4:
                obs = obs.transpose(0, 3, 1, 2)
            obs = torch.from_numpy(obs).float() / self.scale
            if self.cuda:
                obs = obs.cuda()
            action = self.q_model(obs).argmax().item()

        else:
            action = self.actions_space.sample()

        return action


    def act_with_param_noise(self, obs, update_param_noise_threshold, update_param_noise_scale):

        obs = np.array([obs])
        if obs.ndim == 4:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float() / self.scale
        if self.cuda:
            obs = obs.cuda()


        q_val = self.q_model(obs)
        self.q_model.set_sigma(self.sigma)
        noise_q_val = self.q_model(obs)
        self.q_model.set_sigma(0)

        action = noise_q_val.argmax().item()

        if update_param_noise_scale and update_param_noise_threshold > 0:
            kld = F.softmax(q_val) * (F.log_softmax(q_val) - F.log_softmax(noise_q_val))
            kld = kld.sum()
            if kld.item() > update_param_noise_threshold:
                self.sigma /= 1.01
            else:
                self.sigma *= 1.01
        return action

    def one_hot(self, y):
        y_onehot = torch.zeros(y.size(0), self.actions_space.n)
        if self.cuda:
            y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1)
        return y_onehot

    def update(self, inputs, double_q=True, gamma=0.99, grad_norm_clip=10):
        inputs = [torch.from_numpy(np.array(x, dtype=np.float32)) for x in inputs]
        if self.cuda:
            inputs = [x.cuda() for x in inputs]
        obses_t, actions, rewards, obses_tp1, dones, weights = inputs

        if obses_t.ndimension() == 4:
            obses_t = obses_t.permute(0, 3, 1, 2) / self.scale
            obses_tp1 = obses_tp1.permute(0, 3, 1, 2) / self.scale

        q_t = self.q_model(obses_t)
        q_tp1 = self.target_q_model(obses_tp1)
        q_t_selected = torch.sum(q_t * self.one_hot(actions), dim=1)

        if double_q:
            q_tp1_using_online_net = self.q_model(obses_tp1)
            q_tp1_best_using_online_net = q_tp1_using_online_net.max(dim=1)[1]
            q_tp1_best = torch.sum(q_tp1 * self.one_hot(q_tp1_best_using_online_net), dim=1)
        else:
            q_tp1_best = q_tp1.max(dim=1)[0]

        rhs = rewards.squeeze() + gamma * (1 - dones) * q_tp1_best.detach()
        td_err = q_t_selected.detach() - rhs.detach()

        loss = F.smooth_l1_loss(q_t_selected, rhs, reduction='none')
        loss_weighted = torch.mean(loss * weights)

        self.optimizer.zero_grad()
        loss_weighted.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), grad_norm_clip)
        self.optimizer.step()

        if self.noisy_q_model is not None:
            self.noisy_q_model = copy.deepcopy(self.q_model)


        return td_err.detach().cpu().numpy()

    def sync(self):
        self.target_q_model = copy.deepcopy(self.q_model)

    def save(self, model_file):
        data = {
            'q_model': self.q_model.state_dict(),
            'target_model': self.target_q_model.state_dict()}
        torch.save(data, model_file)

    def load(self, model_file):
        data = torch.load(model_file)
        self.q_model.load_state_dict(data['q_model'])
        self.target_q_model.load_state_dict(data['target_model'])