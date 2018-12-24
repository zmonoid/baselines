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




class CNN(nn.Module):
    def __init__(self, convs, input_channel=4):
        super(CNN, self).__init__()
        layers = []
        for output_channel, kernel_size, stride in convs:
            layers.append(nn.Conv2d(input_channel, output_channel, kernel_size, stride))
            layers.append(nn.ReLU())
            input_channel = output_channel
        self.cnns = nn.Sequential(*layers)
        self.convs = convs

    def forward(self, input):
        return self.cnns(input)


class MLP(nn.Module):
    def __init__(self, hiddens, input_size=4, act='relu'):
        super(MLP, self).__init__()
        layers = []
        for hidden_size in hiddens:

            layers.append(NoiseLinear(input_size, hidden_size))

            if act == 'relu':
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())

            input_size = hidden_size
        self.mlp = nn.Sequential(*layers)
        self.hiddens = hiddens

    def forward(self, input):
        return self.mlp(input)



class QModel(nn.Module):
    def __init__(self, cnn, mlp, actions=4, dueling=False):
        super(QModel, self).__init__()
        self.cnn = cnn
        self.mlp = mlp
        self.dueling = dueling

        self.action_head = NoiseLinear(mlp.hiddens[-1], actions)

        if dueling:
            self.state_head = NoiseLinear(mlp.hiddens[-1], 1)

    def forward(self, x):
        if self.cnn is not None:
            x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        action_score = self.action_head(x)

        if self.dueling:
            state_score = self.state_head(x)
            action_score = action_score - action_score.mean(dim=-1, keepdim=True)
            action_score = state_score + action_score

        return action_score

    def set_sigma(self, sigma):
        for m in self.modules():
            if isinstance(m, NoiseLinear):
                m.sigma = sigma

    def print_sigma(self):
        for m in self.modules():
            if isinstance(m, NoiseLinear):
                print('sss', m.sigma)

def build_q_model(network, hiddens=[256], dueling=True, input_shape=(32, 4, 84, 84), actions=4, convs=None):

    if network == 'cnn_mlp':
        batch_size, height, width, input_channel = input_shape
        input_shape = (batch_size, input_channel, height, width)
        var = torch.randn(*input_shape)
        cnn = CNN(convs, input_channel)

        var = cnn(var).view(var.size(0), -1)
        mlp = MLP(hiddens, var.size(1))
        model = QModel(cnn, mlp, actions, dueling)
        return model

    elif network == 'mlp_only':
        batch_size, input_size = input_shape
        mlp = MLP(hiddens, input_size, act='tanh')
        model = QModel(None, mlp, actions, dueling)
        return model


    else:
        raise ValueError("No such model")

