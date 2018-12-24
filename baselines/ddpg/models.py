


import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super(NoiseLinear, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.sigma = 0.0

    def forward(self, x):
        noise_w = torch.randn_like(self.linear.weight).to(x.device) * self.sigma
        noise_b = torch.randn_like(self.linear.bias).to(x.device) * self.sigma
        x = F.linear(x, self.linear.weight + noise_w, self.linear.bias + noise_b)
        return x


class MLP(nn.Module):
    def __init__(self, hiddens, input_size=4, actions=4, act='relu', output_act=None):
        super(MLP, self).__init__()
        layers = []
        for hidden_size in hiddens:

            layers.append(NoiseLinear(input_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))

            if act == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

            input_size = hidden_size

        layers.append(NoiseLinear(input_size, actions))
        if output_act is not None:
            layers.append(nn.Tanh())
        self.mlp = nn.Sequential(*layers)
        self.hiddens = hiddens

    def forward(self, input):
        return self.mlp(input)


    def set_sigma(self, sigma):
        for m in self.modules():
            if isinstance(m, NoiseLinear):
                m.sigma = sigma


def build_actor_critic(network, hiddens=[64, 64], input_shape=(1, 11), actions=4, output_act=None):

    if network == 'mlp_only':
        batch_size, input_size = input_shape
        model = MLP(hiddens, input_size, actions=actions, act='tanh', output_act=output_act)
        return model

    else:
        raise ValueError("No such model")






